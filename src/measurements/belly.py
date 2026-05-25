"""
Belly mesh, volume estimation, and belly-button localization.

Pipeline:
    1. Apply SAM3 belly mask to AMB3R/VGGT point cloud → belly point cloud
    2. Statistical outlier removal
    3. Generate watertight mesh by closing the back of the belly with a fitted plane
    4. Compute volume of closed mesh
    5. Estimate belly button location:
        - PCA to find protrusion direction
        - Find apex (max projection onto protrusion direction)
        - Verify by gradient-flow convergence on the mesh
        - Optional: refine with mean curvature near apex
    6. Compute belly-button-to-feet/ground distance using 2D pose ankle keypoints

Outputs (in metric, assuming the input point cloud has been calibrated via
scale_picker.py + manual_scale.py):
    - belly_pointcloud.ply, belly_mesh.ply
    - volume_cm3
    - belly_button_3d (x, y, z)
    - distance_belly_to_midfeet_cm (point-to-point 3D, ankle-based)
"""

import os
import json
from collections import defaultdict
from typing import Optional

import numpy as np
import open3d as o3d
from PIL import Image as PILImage


# ─── Step 1-2: Build belly point cloud from masks + outlier removal ──

def _transform_mask_to_recon_space(orig_mask_array, transform, target_h, target_w):
    """Apply the same preprocessing (pad or crop) that VGGT/AMB3R applied to
    the input image, so that the mask aligns with the point map pixel-for-pixel.

    Args:
        orig_mask_array: (H_orig, W_orig) uint8 mask, original image resolution.
        transform: dict from VGGT meta with keys depending on mode:
            - mode='pad': orig_size, scaled_size, padded_size, pad_left, pad_top
            - mode='crop': orig_size, scaled_size, cropped_size, crop_left, crop_top
        target_h, target_w: final point-map resolution (from NPZ).

    Returns:
        bool array of shape (target_h, target_w) aligned with the point map.
    """
    mode = transform.get("mode", "pad")
    new_w, new_h = transform["scaled_size"]

    # Step 1: scale original mask down to scaled_size
    mask_pil = PILImage.fromarray(orig_mask_array)
    scaled = mask_pil.resize((new_w, new_h), PILImage.NEAREST)
    scaled = np.array(scaled)

    # Step 2: pad or crop to match the recon canvas
    if mode == "pad":
        pad_left = transform.get("pad_left", 0)
        pad_top = transform.get("pad_top", 0)
        canvas_w, canvas_h = transform.get("padded_size", (new_w, new_h))
        canvas = np.zeros((canvas_h, canvas_w), dtype=scaled.dtype)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = scaled
    else:  # crop
        crop_left = transform.get("crop_left", 0)
        crop_top = transform.get("crop_top", 0)
        cw, ch = transform.get("cropped_size", (new_w, new_h))
        # If the scaled image was larger than the crop window, take the crop;
        # otherwise the scaled image already matches.
        ymax = min(crop_top + ch, scaled.shape[0])
        xmax = min(crop_left + cw, scaled.shape[1])
        canvas = scaled[crop_top:ymax, crop_left:xmax]

    # Step 3: resize from canvas to target point-map size (usually 1:1, but
    # AMB3R may have a different output resolution).
    if (canvas.shape[0], canvas.shape[1]) != (target_h, target_w):
        canvas_pil = PILImage.fromarray(canvas)
        canvas = np.array(canvas_pil.resize((target_w, target_h), PILImage.NEAREST))

    return canvas > 127


def build_belly_pointcloud(amb3r_npz_path, segmentation_dir,
                           output_dir, conf_pct_keep=75,
                           recon_meta_path=None):
    """Apply SAM3 belly masks to AMB3R/VGGT point cloud, return belly-only points.

    Args:
        amb3r_npz_path: Path to point_cloud.npz from AMB3R/VGGT.
        segmentation_dir: Directory with SAM3 segmentation results (segmentation.json).
        output_dir: Directory to save outputs.
        conf_pct_keep: Keep top X% confidence points (rest are noise).
        recon_meta_path: Optional path to reconstruction_meta.json containing
            preprocess_transforms — needed for accurate mask alignment when the
            reconstruction model padded or cropped the input images.

    Returns:
        Dict with paths to PLY and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    npz = np.load(amb3r_npz_path, allow_pickle=True)
    pts_per_frame = npz["points_per_frame"]   # (T, H, W, 3)
    conf_per_frame = npz["conf_per_frame"]    # (T, H, W) or (T, H, W, 1)
    imgs_per_frame = npz["images_per_frame"]  # (T, H, W, 3)

    T, H_amb, W_amb, _ = pts_per_frame.shape
    if conf_per_frame.ndim == 4 and conf_per_frame.shape[-1] == 1:
        conf_per_frame = conf_per_frame[..., 0]

    # Load reconstruction metadata to apply matching transforms to masks.
    # Without this, masks (in original-image space) won't align with the
    # point map (in cropped/padded space).
    transforms_by_filename = {}
    image_files_in_order = None
    if recon_meta_path and os.path.exists(recon_meta_path):
        with open(recon_meta_path) as f:
            recon_meta = json.load(f)
        for t in recon_meta.get("preprocess_transforms", []):
            transforms_by_filename[t["filename"]] = t
        # Canonical ordering used by VGGT/AMB3R — must match T-axis of NPZ
        image_files_in_order = recon_meta.get("image_files_in_order")
        print(f"  Loaded {len(transforms_by_filename)} preprocess transforms "
              f"(mode={recon_meta.get('preprocess_mode', '?')})")

    seg_path = os.path.join(segmentation_dir, "segmentation.json")
    with open(seg_path) as f:
        seg = json.load(f)

    # Use the recon_meta ordering when available — this is the order VGGT/AMB3R
    # actually processed the frames, so the T-axis of the NPZ matches index-by-index.
    # Falling back to alphabetical sort would silently mis-align if the recon
    # subset doesn't start with frame_000.
    if image_files_in_order is not None:
        ordered_frames = list(image_files_in_order)
    else:
        ordered_frames = sorted(seg.keys())

    belly_pts = []
    belly_colors = []
    belly_conf = []

    for frame_idx in range(min(T, len(ordered_frames))):
        img_name = ordered_frames[frame_idx]
        if img_name not in seg:
            print(f"  Frame {frame_idx} ({img_name}): no SAM3 entry, skipping")
            continue
        mask_path = seg[img_name].get("combined_mask_path")
        if not mask_path or not os.path.exists(mask_path):
            print(f"  Frame {frame_idx} ({img_name}): no belly mask, skipping")
            continue

        mask_pil = PILImage.open(mask_path).convert("L")
        orig_mask = np.array(mask_pil)

        # Apply the same preprocessing transform as the reconstruction model
        # so the mask aligns with the point map.
        transform = transforms_by_filename.get(img_name)
        if transform is not None:
            mask = _transform_mask_to_recon_space(orig_mask, transform, H_amb, W_amb)
        else:
            # Legacy fallback: simple stretch (only correct if no crop/pad happened)
            mask = (np.array(mask_pil.resize((W_amb, H_amb), PILImage.NEAREST))
                    > 127)

        if mask.sum() < 50:
            print(f"  Frame {frame_idx}: mask too small ({mask.sum()} px), skipping")
            continue

        pts = pts_per_frame[frame_idx][mask]   # (N, 3)
        col = imgs_per_frame[frame_idx][mask]  # (N, 3)
        cf = conf_per_frame[frame_idx][mask]   # (N,)

        valid = np.linalg.norm(pts, axis=-1) > 0.001
        if valid.sum() == 0:
            continue

        pts = pts[valid]; col = col[valid]; cf = cf[valid]

        # Confidence filter: keep top X% within this frame
        thr = np.percentile(cf, 100 - conf_pct_keep)
        keep = cf >= thr
        belly_pts.append(pts[keep])
        belly_colors.append(col[keep])
        belly_conf.append(cf[keep])

        print(f"  Frame {frame_idx} ({img_name}): kept {keep.sum()}/{len(pts)} points")

    if not belly_pts:
        raise RuntimeError("No belly points recovered from any frame")

    belly_pts = np.concatenate(belly_pts, axis=0)
    belly_colors = np.concatenate(belly_colors, axis=0)
    belly_conf = np.concatenate(belly_conf, axis=0)

    # Build initial point cloud and apply outlier removal in two passes:
    #   (1) statistical outlier removal — drops points whose mean distance to
    #       the k nearest neighbors is > mean + std_ratio·std. Conservative.
    #   (2) Mahalanobis-style distance-from-centroid trim — drops points whose
    #       distance from the centroid exceeds the 99th percentile by a factor.
    #       This catches extreme outliers (e.g., depth bleed at mask edges
    #       that look like points way off the balloon) that SOR can miss.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(belly_pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(belly_colors, 0, 1))

    n0 = len(belly_pts)
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    n1 = len(cl.points)

    # Distance-from-centroid trim — drops the most-extreme stragglers.
    cl_pts = np.asarray(cl.points)
    centroid = cl_pts.mean(axis=0)
    d = np.linalg.norm(cl_pts - centroid, axis=1)
    # Robust threshold: 95th percentile × 1.20 (keeps the bulk, drops far stragglers)
    thresh = np.percentile(d, 95) * 1.20
    keep = d < thresh
    cl_pts_filt = cl_pts[keep]
    cl_cols_filt = np.asarray(cl.colors)[keep] if len(cl.colors) else None
    cl = o3d.geometry.PointCloud()
    cl.points = o3d.utility.Vector3dVector(cl_pts_filt)
    if cl_cols_filt is not None:
        cl.colors = o3d.utility.Vector3dVector(cl_cols_filt)
    n2 = len(cl.points)
    print(f"  Outlier removal: {n0} → {n1} (SOR) → {n2} (centroid-trim, "
          f"thresh={thresh*100:.1f}cm)")

    out_ply = os.path.join(output_dir, "belly_pointcloud.ply")
    o3d.io.write_point_cloud(out_ply, cl)

    return {
        "ply": out_ply,
        "n_points_raw": int(len(belly_pts)),
        "n_points_clean": int(len(cl.points)),
    }


# ─── Step 3: Watertight mesh by closing back with fitted plane ────────

def fit_back_plane(points, camera_position=None):
    """Fit a plane to the 'back' of the belly (least-protruding portion).

    Strategy:
        1. PCA on the points → smallest eigenvector is the surface-normal axis
           (perpendicular to the locally-flat captured surface).
        2. PCA gives direction only up to a sign — disambiguate using the
           camera position: the protrusion direction should point FROM the
           captured surface TOWARD the camera (i.e., out of the body, into
           free space). With VGGT/AMB3R world coords, the reference camera
           is at the origin, so we use that as default.
        3. Plane sits at the 5th-percentile height behind the centroid (the
           "back" of the captured belly) along the protrusion direction.

    Why the camera-based disambiguation:
        For a balloon/belly captured from the front with camera arc > 180°,
        the cloud wraps slightly past the equator. The naive "flip if
        |h.max| < |h.min|" heuristic FAILS in that case because the bigger
        extent is on the far-from-camera side, making the protrusion vector
        point INTO the body. Using the camera's location is unambiguous.

    Args:
        points: (N, 3) belly point cloud (in VGGT/AMB3R world frame).
        camera_position: (3,) camera position; defaults to origin (which is
            where VGGT/AMB3R place the reference camera).

    Returns:
        plane_point: a point on the back plane (3,).
        protrusion_direction: outward-pointing unit normal of the plane (3,)
            pointing FROM the back of the belly TOWARD the camera.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Smallest eigenvector → least spread direction → perpendicular to surface
    protrusion_dir = eigvecs[:, 0]

    # Disambiguate sign using the camera location. The protrusion direction
    # should point from the centroid TOWARD the camera (i.e., toward the open
    # half-space the camera is observing from).
    if camera_position is None:
        camera_position = np.zeros(3)
    to_camera = np.asarray(camera_position) - centroid
    to_camera_norm = np.linalg.norm(to_camera)
    if to_camera_norm > 1e-6:
        to_camera_unit = to_camera / to_camera_norm
        if protrusion_dir @ to_camera_unit < 0:
            protrusion_dir = -protrusion_dir
    else:
        # Degenerate fallback — the original spread-based heuristic
        h_tmp = centered @ protrusion_dir
        if abs(h_tmp.max()) < abs(h_tmp.min()):
            protrusion_dir = -protrusion_dir

    # Plane sits at the 5th percentile of heights along protrusion_dir (the
    # "back" of the captured belly — away from camera).
    h = centered @ protrusion_dir
    back_h = np.percentile(h, 5)
    plane_point = centroid + back_h * protrusion_dir

    return plane_point, protrusion_dir


def build_belly_mesh(belly_ply_path, output_dir, poisson_depth=8):
    """Generate a watertight mesh from the belly point cloud.

    Approach:
        1. Estimate normals on the belly point cloud
        2. Run screened Poisson surface reconstruction (gives a closed mesh,
           though the back will be a curved approximation)
        3. Trim by density to remove spurious geometry
        4. Crop to the bounding box of the actual belly points

    Returns:
        mesh_path, mesh, plane_point, protrusion_dir
    """
    pcd = o3d.io.read_point_cloud(belly_ply_path)
    points = np.asarray(pcd.points)
    if len(points) < 200:
        raise RuntimeError(f"Too few belly points to mesh: {len(points)}")

    plane_point, protrusion_dir = fit_back_plane(points)

    # Estimate normals — orient them in the protrusion direction (outward)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    normals = np.asarray(pcd.normals)
    # Flip any normal that points into the body
    flips = (normals @ protrusion_dir) < 0
    normals[flips] = -normals[flips]
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, scale=1.1, linear_fit=False
    )
    densities = np.asarray(densities)

    # Remove low-density vertices (spurious geometry far from actual data)
    threshold = np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(densities < threshold)

    # Crop to AABB of actual belly points (with small padding)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb = aabb.scale(1.05, aabb.get_center())
    mesh = mesh.crop(aabb)

    mesh.compute_vertex_normals()

    out_path = os.path.join(output_dir, "belly_mesh.ply")
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"  Belly mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    return out_path, mesh, plane_point, protrusion_dir


# ─── Step 4: Volume estimation ────────────────────────────────────────

def estimate_volume_above_plane(mesh, plane_point, plane_normal):
    """Compute the volume of the mesh region in front of (above) the back plane.

    Uses the divergence theorem on the closed convex hull of the mesh, sliced
    and capped at the back plane. The convex hull is always watertight, so
    Open3D / trimesh can compute its volume exactly via the divergence theorem.

    Why we don't use the raw Poisson mesh:
        Poisson surface reconstruction often produces slightly non-manifold
        meshes (duplicate triangles, tiny self-intersections, holes that
        don't fully close). These break any closed-mesh volume integral,
        and the previous "sum of (projected_area × avg_height)" formula
        was an unsigned integral that double-counts both sides of any
        closed shape — bulge volume on a closed balloon mesh comes out
        roughly 2× the true volume.

    For a balloon (convex): hull = true balloon shape → exact volume.
    For a pregnant belly (slightly concave around navel): hull bridges
        the navel dimple, overestimating by a few percent. Acceptable.

    Args:
        mesh: open3d.geometry.TriangleMesh
        plane_point: a point on the back plane (3,)
        plane_normal: outward-pointing unit normal (3,) (toward belly front)

    Returns:
        Volume (in mesh units cubed). If mesh is in meters → result in m³.
    """
    import trimesh
    plane_normal = np.asarray(plane_normal, dtype=float)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    if len(V) == 0 or len(F) == 0:
        return 0.0

    tm = trimesh.Trimesh(vertices=V, faces=F, process=True)
    hull = tm.convex_hull  # always watertight

    # Slice the hull by the back plane (kept side: where (v - origin) · normal >= 0)
    # and cap the cut so the result is closed.
    try:
        clipped = trimesh.intersections.slice_mesh_plane(
            hull,
            plane_normal=plane_normal,
            plane_origin=plane_point,
            cap=True,
        )
        if clipped is not None and clipped.is_watertight and clipped.volume > 0:
            return float(clipped.volume)
    except Exception:
        pass

    # Fallback if slicing fails: use whole hull volume (also watertight)
    return float(hull.volume)


def _fit_sphere_least_squares(points):
    """Algebraic least-squares sphere fit.

    Solves ||p - c||² = r² → ||p||² = 2p·c + (r² - ||c||²) = 2p·c + d
    Linear in c and d. Returns (center, radius).
    For a near-spherical object captured even partially, this recovers
    the FULL sphere — much better than convex hull of a partial capture.
    """
    pts = np.asarray(points)
    if len(pts) < 4:
        return None, None, None
    A = np.hstack([2 * pts, np.ones((len(pts), 1))])
    b = np.sum(pts**2, axis=1)
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None
    c = x[:3]
    d = x[3]
    r_sq = d + c @ c
    if r_sq <= 0:
        return None, None, None
    r = float(np.sqrt(r_sq))
    # Residual stats — fraction of points within 1 cm tells us how spherical
    # the data actually is. 80%+ = trust the sphere fit. <50% = data isn't
    # spherical, prefer hull/ellipsoid estimates.
    dists = np.linalg.norm(pts - c, axis=1)
    res = dists - r
    frac_inlier_1cm = float((np.abs(res) < 0.01).sum() / len(pts))
    return c.tolist(), r, frac_inlier_1cm


def _ellipsoid_volume_from_points(points, sigma_mult=2.0):
    """Fit an axis-aligned-by-PCA ellipsoid to the points and return its volume.

    Strategy:
        1. Center the points and compute the covariance matrix.
        2. PCA: eigenvalues are variances along principal axes.
        3. Axes lengths = sigma_mult × sqrt(variance) — a sigma_mult of 2.0
           captures ~95% of point mass for a Gaussian; for a balloon-like
           shell, 1.0 ≈ surface diameter, 2.0 ≈ ~2× too big.
        4. For a uniformly-sampled surface of an ellipsoid the surface points
           sit at exactly 1 sigma along each axis (because variance of points
           on a thin shell of radius r equals r²/3 in 1D × … actually = r²
           uniformly in 1D for a 1-D segment of length 2r, not for a sphere).
           In practice the empirical sigma_mult that matches a balloon volume
           tightly is ≈ √3 ≈ 1.73. We use 1.73 as default — the volume of an
           ellipsoid whose semi-axes equal √3 × σ along each PCA axis
           matches a sphere's volume when the points are uniformly sampled
           on its surface (variance of surface points in any direction =
           r²/3 → r = √3 σ).
    """
    if len(points) < 4:
        return None, None
    pts = np.asarray(points)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    axes = sigma_mult * np.sqrt(eigvals)
    volume = (4.0 / 3.0) * np.pi * axes[0] * axes[1] * axes[2]
    return float(volume), [float(a) for a in axes]


def compute_belly_volume(mesh, plane_point, protrusion_dir, points=None):
    """Compute multiple volume estimates of the bulge.

    Primary estimate (most trustworthy):
      `bulge_volume_cm3` = convex hull of the CLEANED point cloud, sliced and
      capped at the back plane. Tighter than the mesh hull (which uses the
      Poisson-extrapolated 145k+ vertices).

    Secondary estimates for cross-check:
      - `mesh_hull_volume_cm3`: hull of the Poisson mesh vertices (looser).
      - `ellipsoid_volume_cm3`: PCA-fit ellipsoid (tightest, best for balloons).
      - `convex_hull_volume_cm3`: full hull of point cloud (no slicing).
      - `obb_volume_cm3`: oriented bounding box (rough upper bound).

    For a (mostly-)convex object like a balloon: all four should be of the
    same order of magnitude. A big spread suggests the point cloud is noisy
    (e.g. mask bleed into surrounding clothing/hand) and the smallest
    estimate (ellipsoid) is likely closer to truth.
    """
    import trimesh
    protrusion_dir = np.asarray(protrusion_dir, dtype=float)
    protrusion_dir = protrusion_dir / (np.linalg.norm(protrusion_dir) + 1e-12)

    # === PRIMARY: clipped hull of the CLEANED point cloud ===
    bulge_m3 = None
    pc_hull_full_m3 = None
    if points is not None and len(points) >= 4:
        try:
            pc = trimesh.PointCloud(np.asarray(points))
            pc_hull = pc.convex_hull
            pc_hull_full_m3 = float(pc_hull.volume)
            clipped = trimesh.intersections.slice_mesh_plane(
                pc_hull, plane_normal=protrusion_dir, plane_origin=plane_point,
                cap=True,
            )
            if clipped is not None and clipped.is_watertight and clipped.volume > 0:
                bulge_m3 = float(clipped.volume)
            else:
                bulge_m3 = pc_hull_full_m3
        except Exception:
            pass

    # Fallback: existing mesh-hull-based method
    if bulge_m3 is None:
        bulge_m3 = estimate_volume_above_plane(mesh, plane_point, protrusion_dir)

    # === SECONDARY: ellipsoid fit on cleaned points ===
    ellipsoid_m3, ellipsoid_axes_m = None, None
    if points is not None and len(points) >= 4:
        ellipsoid_m3, ellipsoid_axes_m = _ellipsoid_volume_from_points(
            np.asarray(points), sigma_mult=np.sqrt(3.0),
        )

    # === SECONDARY: hull of the mesh vertices ===
    mesh_hull_m3 = None
    try:
        hull_mesh, _ = mesh.compute_convex_hull()
        mesh_hull_m3 = float(hull_mesh.get_volume())
    except Exception:
        pass

    # === SECONDARY: oriented bounding box ===
    obb = mesh.get_oriented_bounding_box()
    obb_volume_m3 = float(obb.volume())

    # === SECONDARY: least-squares SPHERE fit ===
    # Best estimate for a balloon (≈ sphere). Recovers the FULL sphere even
    # from a partial (hemisphere) capture. `sphere_inlier_fraction` indicates
    # how spherical the data really is — if > 0.7, trust this number.
    sphere_volume_m3 = None
    sphere_radius_m = None
    sphere_center_m = None
    sphere_inlier_frac = None
    if points is not None and len(points) >= 4:
        center, radius, inlier_frac = _fit_sphere_least_squares(np.asarray(points))
        if center is not None:
            sphere_center_m = center
            sphere_radius_m = radius
            sphere_volume_m3 = (4.0 / 3.0) * np.pi * radius ** 3
            sphere_inlier_frac = inlier_frac

    return {
        # Primary (sliced cleaned-point hull above back plane)
        "bulge_volume_m3": bulge_m3,
        "bulge_volume_cm3": bulge_m3 * 1e6,
        "bulge_volume_liters": bulge_m3 * 1000,
        "method": "cleaned_pointcloud_hull_clipped",

        # Comparisons
        "point_cloud_hull_full_cm3": (pc_hull_full_m3 * 1e6) if pc_hull_full_m3 else None,
        "mesh_hull_volume_cm3": (mesh_hull_m3 * 1e6) if mesh_hull_m3 else None,
        "ellipsoid_volume_cm3": (ellipsoid_m3 * 1e6) if ellipsoid_m3 else None,
        "ellipsoid_axes_cm": ([a*100 for a in ellipsoid_axes_m]
                                if ellipsoid_axes_m else None),
        "obb_volume_m3": obb_volume_m3,
        "obb_volume_cm3": obb_volume_m3 * 1e6,

        # Sphere fit (ideal for balloons / mostly-spherical objects)
        "sphere_volume_cm3": (sphere_volume_m3 * 1e6) if sphere_volume_m3 else None,
        "sphere_radius_cm": (sphere_radius_m * 100) if sphere_radius_m else None,
        "sphere_center_cm": ([c*100 for c in sphere_center_m]
                                if sphere_center_m else None),
        "sphere_inlier_fraction_within_1cm": sphere_inlier_frac,

        # Backwards-compat alias
        "convex_hull_volume_m3": pc_hull_full_m3 or mesh_hull_m3,
        "convex_hull_volume_cm3": ((pc_hull_full_m3 or mesh_hull_m3 or 0) * 1e6),
    }


# ─── Step 5: Belly-button localization ────────────────────────────────

def build_vertex_adjacency(triangles, n_vertices):
    """Build a mapping vertex_idx → set of neighboring vertex indices."""
    adj = defaultdict(set)
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    return {k: list(v) for k, v in adj.items()}


def find_belly_button(mesh, plane_point, protrusion_dir,
                     verify_with_gradient_flow=True):
    """Locate the belly button on the belly mesh.

    Algorithm (described by user):
        - The belly is approximately ellipsoidal
        - Define a height field h(v) = (v - plane_point) · protrusion_dir
        - The apex of protrusion is the vertex with maximum h
        - Verify by gradient-flow convergence: from each vertex, walk
          to the highest-h neighbor until no neighbor is higher (local max).
          For an ellipsoidal surface, all paths converge to the same apex.
        - The belly button is approximately at this apex.

    Args:
        mesh: open3d.geometry.TriangleMesh
        plane_point, protrusion_dir: from fit_back_plane()
        verify_with_gradient_flow: run flow-convergence check

    Returns:
        Dict with 3D position, protrusion height, convergence stats.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    n = len(V)
    if n == 0:
        return {"error": "empty mesh"}

    # Height field
    h = (V - plane_point) @ protrusion_dir
    apex_idx = int(np.argmax(h))
    apex_pos = V[apex_idx].tolist()

    result = {
        "position_3d": apex_pos,
        "protrusion_direction": protrusion_dir.tolist(),
        "protrusion_height_cm": float((h[apex_idx] - h.min()) * 100),
        "method": "pca_apex",
    }

    if not verify_with_gradient_flow:
        return result

    # Gradient-flow convergence check
    adj = build_vertex_adjacency(F, n)
    convergence = np.full(n, -1, dtype=int)
    for start in range(n):
        cur = start
        for _ in range(2000):
            neighbors = adj.get(cur, [])
            if not neighbors:
                break
            best = max(neighbors, key=lambda x: h[x])
            if h[best] <= h[cur]:
                break
            cur = best
        convergence[start] = cur

    # Find the most popular convergence point (mode)
    unique, counts = np.unique(convergence[convergence >= 0], return_counts=True)
    if len(counts) > 0:
        mode_idx = int(unique[np.argmax(counts)])
        mode_count = int(counts.max())
        result.update({
            "gradient_apex_idx": mode_idx,
            "gradient_apex_position_3d": V[mode_idx].tolist(),
            "convergence_count": mode_count,
            "convergence_fraction": mode_count / float(n),
            "n_local_maxima": int(len(unique)),
        })
        # Use the gradient-converged point as the belly button (more robust)
        result["position_3d"] = V[mode_idx].tolist()
        result["method"] = "gradient_flow_convergence"

    return result


# ─── Step 6: Belly button to feet / ground distance ───────────────────

def find_feet_3d(amb3r_npz_path, pose_results_path):
    """Locate left and right ankle 3D positions using pose keypoints.

    Args:
        amb3r_npz_path: point_cloud.npz from AMB3R/VGGT (calibrated to metric).
        pose_results_path: pose_results.json from running pose on AMB3R-resolution
            images (so pixel coords map 1:1 to point map).

    Returns:
        Dict with left_ankle_3d, right_ankle_3d, midfoot_3d (mean of two).
    """
    npz = np.load(amb3r_npz_path, allow_pickle=True)
    pts_per_frame = npz["points_per_frame"]  # (T, H, W, 3)
    T, H, W, _ = pts_per_frame.shape

    with open(pose_results_path) as f:
        pose = json.load(f)

    # Best frame = highest combined ankle confidence
    image_names = sorted(pose.keys())
    best = None
    best_score = -1.0
    for frame_idx, img_name in enumerate(image_names):
        if frame_idx >= T:
            break
        persons = pose[img_name].get("persons", [])
        if not persons:
            continue
        lk = persons[0].get("leg_keypoints", {})
        l_ankle = lk.get("left_ankle", {"score": 0})
        r_ankle = lk.get("right_ankle", {"score": 0})
        score = l_ankle.get("score", 0) + r_ankle.get("score", 0)
        if score > best_score:
            best_score = score
            best = (frame_idx, l_ankle, r_ankle)

    if best is None:
        return None

    frame_idx, l_ankle_kp, r_ankle_kp = best

    def lookup(px, py):
        ix = max(0, min(int(round(px)), W - 1))
        iy = max(0, min(int(round(py)), H - 1))
        patch = pts_per_frame[frame_idx,
                              max(0, iy-2):iy+3,
                              max(0, ix-2):ix+3]
        valid = np.linalg.norm(patch, axis=-1) > 0.001
        if valid.sum() == 0:
            return None
        return patch[valid].mean(axis=0)

    l_3d = lookup(l_ankle_kp.get("x", 0), l_ankle_kp.get("y", 0))
    r_3d = lookup(r_ankle_kp.get("x", 0), r_ankle_kp.get("y", 0))

    if l_3d is None or r_3d is None:
        return None

    midfoot = (np.array(l_3d) + np.array(r_3d)) / 2.0
    return {
        "left_ankle_3d": l_3d.tolist(),
        "right_ankle_3d": r_3d.tolist(),
        "midfoot_3d": midfoot.tolist(),
        "frame_used": frame_idx,
        "ankle_confidence_sum": float(best_score),
    }


def compute_distance_to_feet(belly_button_3d, feet_info):
    """Compute the 3D Euclidean distance from the belly button to the
    midpoint of the two ankles.

    We deliberately do NOT compute a "distance to ground" here:
    estimating ground from the point cloud (taking an extreme percentile
    along a derived vertical axis) was found to be unreliable — it
    fluctuates with whatever happens to be the lowest captured surface
    (shoes, mat, partial floor coverage) and with the chosen vertical
    direction. Ankle-to-belly-button is anatomically anchored and stable.

    Returns:
        Dict with distance_belly_to_midfeet_cm + the 3D positions used.
        Or `{"error": ...}` if pose couldn't find the feet.
    """
    bb = np.array(belly_button_3d)
    if feet_info is None:
        return {"error": "no feet detected", "belly_button_3d": belly_button_3d}

    midfoot = np.array(feet_info["midfoot_3d"])
    d_feet_m = float(np.linalg.norm(bb - midfoot))

    return {
        "belly_button_3d": belly_button_3d,
        "midfoot_3d": midfoot.tolist(),
        "left_ankle_3d": feet_info["left_ankle_3d"],
        "right_ankle_3d": feet_info["right_ankle_3d"],
        "distance_belly_to_midfeet_cm": d_feet_m * 100,
        "frame_used_for_feet": feet_info["frame_used"],
    }


# Backwards-compat alias — old callers may still import this name. Drops
# the unused `point_cloud_npz_path` and `protrusion_dir` args.
def compute_distance_to_ground(belly_button_3d, feet_info,
                                 point_cloud_npz_path=None, protrusion_dir=None):
    return compute_distance_to_feet(belly_button_3d, feet_info)


# ─── Top-level orchestration helper ───────────────────────────────────

def run_belly_pipeline(amb3r_npz_path, segmentation_dir, output_dir,
                       pose_results_path=None, conf_pct_keep=75,
                       poisson_depth=8, recon_meta_path=None):
    """Run the complete belly analysis pipeline.

    Args:
        amb3r_npz_path: Path to point_cloud.npz from AMB3R/VGGT
            (assumed already calibrated to true metric units).
        segmentation_dir: Directory with SAM3 belly segmentation.
        output_dir: Where to save belly outputs.
        pose_results_path: Optional pose_results.json on AMB3R-res images,
            for distance-to-feet computation.
        conf_pct_keep: Top X% confidence points to keep.
        poisson_depth: Poisson reconstruction octree depth.

    Returns:
        Dict with all results, also saved as belly_results.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\n[Belly] Step 1: building belly point cloud...")
    pc_info = build_belly_pointcloud(
        amb3r_npz_path, segmentation_dir, output_dir,
        conf_pct_keep=conf_pct_keep,
        recon_meta_path=recon_meta_path,
    )
    results["pointcloud"] = pc_info

    print("\n[Belly] Step 2: building belly mesh...")
    mesh_path, mesh, plane_point, protrusion_dir = build_belly_mesh(
        pc_info["ply"], output_dir, poisson_depth=poisson_depth,
    )
    results["mesh_path"] = mesh_path
    results["plane_point"] = plane_point.tolist()
    results["protrusion_direction"] = protrusion_dir.tolist()

    print("\n[Belly] Step 3: estimating volume...")
    # Load the CLEANED belly points (the file build_belly_pointcloud saved
    # after SOR + centroid-trim). Volume is computed from these directly —
    # the mesh hull is also reported for comparison but uses Poisson's
    # extrapolated vertices, which can be looser.
    _bp_pcd = o3d.io.read_point_cloud(pc_info["ply"])
    _bp_pts = np.asarray(_bp_pcd.points)
    vol = compute_belly_volume(mesh, plane_point, protrusion_dir,
                                 points=_bp_pts)
    results["volume"] = vol
    print(f"  Volume estimates (different methods, in cm³):")
    print(f"    [primary]   clipped point-cloud hull above plane: "
          f"{vol['bulge_volume_cm3']:7.0f}")
    if vol.get("sphere_volume_cm3"):
        inl = vol.get("sphere_inlier_fraction_within_1cm", 0) or 0
        rad = vol.get("sphere_radius_cm") or 0
        trust = "TRUST" if inl > 0.70 else "low-trust" if inl < 0.50 else "ok"
        print(f"    [sphere]    LS sphere fit (r={rad:.1f}cm, "
              f"{inl*100:.0f}% inliers w/in 1cm, {trust}): "
              f"{vol['sphere_volume_cm3']:7.0f}")
    if vol.get("ellipsoid_volume_cm3"):
        ax = vol.get("ellipsoid_axes_cm") or [0, 0, 0]
        print(f"    [ellipsoid] PCA-fit ellipsoid (axes ≈ "
              f"{ax[0]:.1f}×{ax[1]:.1f}×{ax[2]:.1f}cm): "
              f"{vol['ellipsoid_volume_cm3']:7.0f}")
    if vol.get("point_cloud_hull_full_cm3"):
        print(f"    [hull]      full point-cloud hull (no plane clip):    "
              f"{vol['point_cloud_hull_full_cm3']:7.0f}")
    if vol.get("mesh_hull_volume_cm3"):
        print(f"    [mesh]      Poisson mesh's convex hull:               "
              f"{vol['mesh_hull_volume_cm3']:7.0f}")
    if vol.get("obb_volume_cm3"):
        print(f"    [obb]       oriented bounding box (sanity ceiling):   "
              f"{vol['obb_volume_cm3']:7.0f}")
    print(f"  → primary in liters: {vol['bulge_volume_liters']:.2f} L")
    print(f"  → For a balloon, [sphere] is usually the best estimate of "
          f"the true (water-displacement) volume.")
    print(f"  (convex hull: {vol['convex_hull_volume_cm3']:.1f} cm³, "
          f"OBB: {vol['obb_volume_cm3']:.1f} cm³)")

    print("\n[Belly] Step 4: locating belly button via gradient flow...")
    bb = find_belly_button(mesh, plane_point, protrusion_dir,
                           verify_with_gradient_flow=True)
    results["belly_button"] = bb
    if "position_3d" in bb:
        print(f"  Belly button at: {[round(x*100,1) for x in bb['position_3d']]} cm")
        print(f"  Method: {bb.get('method', '?')}, "
              f"convergence fraction: {bb.get('convergence_fraction', 0):.2%}")

    if pose_results_path and os.path.exists(pose_results_path):
        print("\n[Belly] Step 5: distance from belly button to feet (ankle-based)...")
        feet = find_feet_3d(amb3r_npz_path, pose_results_path)
        dist = compute_distance_to_feet(bb["position_3d"], feet)
        results["distances"] = dist
        if "distance_belly_to_midfeet_cm" in dist:
            print(f"  Belly button → midfoot (3D Euclidean): "
                  f"{dist['distance_belly_to_midfeet_cm']:.1f} cm")
        elif "error" in dist:
            print(f"  (no distance computed: {dist['error']})")

    out_json = os.path.join(output_dir, "belly_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Belly] Saved results: {out_json}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Belly analysis pipeline")
    parser.add_argument("--amb3r_npz", required=True, help="point_cloud.npz from AMB3R/VGGT")
    parser.add_argument("--segmentation_dir", required=True, help="SAM3 segmentation dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pose_results", default=None,
                        help="pose_results.json on AMB3R-res images (for feet distance)")
    parser.add_argument("--conf_pct_keep", type=int, default=75,
                        help="Keep top X%% confidence points (default 75)")
    parser.add_argument("--poisson_depth", type=int, default=8)
    args = parser.parse_args()

    run_belly_pipeline(
        args.amb3r_npz, args.segmentation_dir, args.output_dir,
        pose_results_path=args.pose_results,
        conf_pct_keep=args.conf_pct_keep,
        poisson_depth=args.poisson_depth,
    )
