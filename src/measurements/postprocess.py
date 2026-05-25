"""
Point cloud post-processing for noise reduction and person segmentation.

Applies the noise mitigation stack described in the pipeline design:
1. SAM3 mask-based person segmentation (removes background from point cloud)
2. Statistical outlier removal
3. Optional surface reconstruction (Screened Poisson)
4. Normal estimation

Runs in the 'leg_pipeline' environment (open3d required).
"""

import os
import json
import numpy as np
import open3d as o3d
from PIL import Image as PILImage


def load_point_cloud(path):
    """Load a point cloud from PLY or NPZ file."""
    if path.endswith(".ply"):
        return o3d.io.read_point_cloud(path)
    elif path.endswith(".npz"):
        data = np.load(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data["points"])
        if "colors" in data:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(data["colors"], 0, 1))
        return pcd
    else:
        raise ValueError(f"Unsupported file format: {path}")


def statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers from point cloud.

    Points whose mean distance to k-nearest neighbors exceeds
    mean + std_ratio * std are removed.

    Args:
        pcd: Open3D PointCloud.
        nb_neighbors: Number of neighbors to consider.
        std_ratio: Standard deviation multiplier threshold.

    Returns:
        Cleaned PointCloud, inlier indices.
    """
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return cl, ind


def radius_outlier_removal(pcd, nb_points=16, radius=0.05):
    """Remove radius outliers from point cloud.

    Points with fewer than nb_points neighbors within radius are removed.
    """
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return cl, ind


def estimate_normals(pcd, radius=0.05, max_nn=30):
    """Estimate point normals for surface reconstruction."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    return pcd


def poisson_reconstruction(pcd, depth=9, scale=1.1, linear_fit=False):
    """Screened Poisson surface reconstruction.

    Converts noisy point cloud into a smooth, watertight mesh.

    Args:
        pcd: Open3D PointCloud with normals.
        depth: Octree depth (higher = more detail, more memory).
        scale: Scale factor for the bounding box.
        linear_fit: Use linear interpolation for the Poisson field.

    Returns:
        mesh: Open3D TriangleMesh.
        densities: Per-vertex density values (for filtering low-confidence areas).
    """
    if not pcd.has_normals():
        pcd = estimate_normals(pcd)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit
    )

    return mesh, np.asarray(densities)


def filter_mesh_by_density(mesh, densities, quantile=0.1):
    """Remove low-density vertices from Poisson mesh.

    The Poisson reconstruction can create spurious geometry in areas
    with sparse points. Removing low-density vertices cleans this up.
    """
    threshold = np.quantile(densities, quantile)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh


def filter_pointcloud_by_masks(
    points_per_frame,
    colors_per_frame,
    conf_per_frame,
    segmentation_dir,
    amb3r_images_dir,
):
    """Filter 3D point cloud using SAM3 person segmentation masks.

    For each frame, loads the corresponding person mask, resizes it to match
    the AMB3R point map resolution, and keeps only points where the mask is
    positive (i.e., the person's body).

    Args:
        points_per_frame: (T, H, W, 3) AMB3R world points.
        colors_per_frame: (T, H, W, 3) or None — if provided, filter colors too.
        conf_per_frame: (T, H, W, 1) AMB3R confidence.
        segmentation_dir: Path to SAM3 segmentation output directory.
        amb3r_images_dir: Path to AMB3R-resolution images (for filename matching).

    Returns:
        Filtered (points, colors, confidence) as flat arrays, plus per-frame masks.
    """
    seg_json_path = os.path.join(segmentation_dir, "segmentation.json")
    if not os.path.exists(seg_json_path):
        print("  WARNING: segmentation.json not found, skipping mask filtering")
        return None

    with open(seg_json_path) as f:
        seg_results = json.load(f)

    T, H_amb, W_amb, _ = points_per_frame.shape
    all_person_masks = np.zeros((T, H_amb, W_amb), dtype=bool)

    # Try to match original image masks to AMB3R frames
    # The segmentation was run on original images, so we need to apply masks
    # to the point map grid (which is at AMB3R resolution).
    seg_images = sorted(seg_results.keys())

    for frame_idx in range(min(T, len(seg_images))):
        img_name = seg_images[frame_idx]
        seg_data = seg_results[img_name]

        mask_path = seg_data.get("combined_mask_path")
        if mask_path is None or not os.path.exists(mask_path):
            print(f"  Frame {frame_idx}: no mask, keeping all points")
            all_person_masks[frame_idx] = True
            continue

        # Load mask and resize to AMB3R resolution
        mask_img = PILImage.open(mask_path).convert("L")
        mask_resized = mask_img.resize((W_amb, H_amb), PILImage.NEAREST)
        mask_array = np.array(mask_resized) > 127  # bool (H_amb, W_amb)

        mask_fraction = mask_array.sum() / (H_amb * W_amb)
        print(f"  Frame {frame_idx}: person mask covers {mask_fraction*100:.1f}% of frame")

        if mask_fraction < 0.01:
            print(f"  WARNING: Mask too small (<1%), keeping all points for this frame")
            all_person_masks[frame_idx] = True
        else:
            all_person_masks[frame_idx] = mask_array

    # Apply masks to filter points
    pts_list = []
    col_list = []
    conf_list = []

    for t in range(T):
        mask = all_person_masks[t]  # (H, W)
        pts = points_per_frame[t][mask]  # (N_valid, 3)

        # Handle confidence shape: may be (H, W) or (H, W, 1)
        cf = conf_per_frame[t]
        if cf.ndim == 3 and cf.shape[-1] == 1:
            cf = cf[:, :, 0]
        conf_vals = cf[mask]  # (N_valid,)

        # Filter out zero/invalid points
        valid = np.linalg.norm(pts, axis=-1) > 0.01

        # Also filter by confidence: keep only top 75% of confidence values
        # This removes noisy low-confidence points that create wild outliers
        if valid.sum() > 0:
            conf_valid = conf_vals[valid]
            conf_threshold_auto = np.percentile(conf_valid, 25)
            conf_ok = conf_vals >= conf_threshold_auto
            valid = valid & conf_ok

        pts_list.append(pts[valid])
        conf_list.append(conf_vals[valid])

        if colors_per_frame is not None:
            col = colors_per_frame[t][mask]
            col_list.append(col[valid])

    filtered_pts = np.concatenate(pts_list, axis=0)
    filtered_conf = np.concatenate(conf_list, axis=0)
    filtered_colors = np.concatenate(col_list, axis=0) if col_list else None

    return {
        "points": filtered_pts,
        "colors": filtered_colors,
        "confidence": filtered_conf,
        "person_masks": all_person_masks,
    }


def full_postprocess(
    input_path,
    output_dir,
    statistical_nb=20,
    statistical_std=2.0,
    do_poisson=True,
    poisson_depth=9,
    density_quantile=0.1,
    segmentation_dir=None,
    amb3r_npz_path=None,
    amb3r_images_dir=None,
    skip_outlier_removal=False,
):
    """Full post-processing pipeline for AMB3R point cloud output.

    Args:
        input_path: Path to point_cloud.ply from AMB3R.
        output_dir: Directory to save processed outputs.
        statistical_nb: Neighbors for statistical outlier removal.
        statistical_std: Std ratio for statistical outlier removal.
        do_poisson: Whether to run Poisson surface reconstruction.
        poisson_depth: Octree depth for Poisson reconstruction.
        density_quantile: Quantile threshold for mesh density filtering.
        segmentation_dir: Path to SAM3 segmentation output (None to skip).
        amb3r_npz_path: Path to point_cloud.npz from AMB3R (needed for mask filtering).
        amb3r_images_dir: Path to AMB3R-resolution images dir.

    Returns:
        Dict with paths to all output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}

    # Step 0: SAM3 mask-based person segmentation (if available)
    if segmentation_dir and amb3r_npz_path and os.path.exists(amb3r_npz_path):
        print("Applying SAM3 person segmentation masks to point cloud...")
        amb3r_data = np.load(amb3r_npz_path, allow_pickle=True)
        pts_per_frame = amb3r_data["points_per_frame"]  # (T, H, W, 3)
        conf_per_frame = amb3r_data["conf_per_frame"]  # (T, H, W, 1)
        imgs_per_frame = amb3r_data["images_per_frame"]  # (T, H, W, 3)

        mask_result = filter_pointcloud_by_masks(
            pts_per_frame, imgs_per_frame, conf_per_frame,
            segmentation_dir, amb3r_images_dir,
        )

        if mask_result is not None:
            # Build a new point cloud from mask-filtered points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mask_result["points"])
            if mask_result["colors"] is not None:
                pcd.colors = o3d.utility.Vector3dVector(
                    np.clip(mask_result["colors"], 0, 1)
                )
            n_original = len(pcd.points)
            print(f"After SAM3 person filtering: {n_original} points")

            # Save person-only point cloud
            person_ply = os.path.join(output_dir, "point_cloud_person.ply")
            o3d.io.write_point_cloud(person_ply, pcd)
            outputs["person_ply"] = person_ply

            # Save updated NPZ with mask info
            mask_npz = os.path.join(output_dir, "person_masks.npz")
            np.savez_compressed(mask_npz, person_masks=mask_result["person_masks"])
            outputs["person_masks_npz"] = mask_npz
        else:
            print("Mask filtering failed, falling back to unfiltered point cloud")
            pcd = load_point_cloud(input_path)
            n_original = len(pcd.points)
    else:
        # Load without mask filtering
        print(f"Loading point cloud: {input_path}")
        pcd = load_point_cloud(input_path)
        n_original = len(pcd.points)

    print(f"Original: {n_original} points")

    # Step 1: Statistical outlier removal
    if skip_outlier_removal:
        print("Skipping statistical outlier removal (--no_outlier_removal)")
        pcd_clean = pcd
        n_clean = n_original
    else:
        print(f"Statistical outlier removal (nb={statistical_nb}, std={statistical_std})...")
        pcd_clean, inlier_idx = statistical_outlier_removal(
            pcd, nb_neighbors=statistical_nb, std_ratio=statistical_std
        )
        n_clean = len(pcd_clean.points)
        print(f"After cleaning: {n_clean} points ({n_original - n_clean} removed)")

    clean_path = os.path.join(output_dir, "point_cloud_clean.ply")
    o3d.io.write_point_cloud(clean_path, pcd_clean)
    outputs["clean_ply"] = clean_path

    # Step 2: Normal estimation
    print("Estimating normals...")
    pcd_clean = estimate_normals(pcd_clean)

    # Step 3: Poisson surface reconstruction
    if do_poisson:
        print(f"Poisson surface reconstruction (depth={poisson_depth})...")
        mesh, densities = poisson_reconstruction(pcd_clean, depth=poisson_depth)
        n_vertices = len(mesh.vertices)
        print(f"Mesh: {n_vertices} vertices, {len(mesh.triangles)} triangles")

        # Filter low-density regions
        mesh = filter_mesh_by_density(mesh, densities, quantile=density_quantile)
        n_filtered = len(mesh.vertices)
        print(f"After density filter: {n_filtered} vertices")

        mesh_path = os.path.join(output_dir, "surface_mesh.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        outputs["mesh_ply"] = mesh_path

    # Save metadata
    meta = {
        "n_original": n_original,
        "n_clean": n_clean,
        "n_removed": n_original - n_clean,
        "statistical_nb": statistical_nb,
        "statistical_std": statistical_std,
        "poisson_depth": poisson_depth if do_poisson else None,
    }
    if do_poisson:
        meta["n_mesh_vertices"] = len(mesh.vertices)
        meta["n_mesh_triangles"] = len(mesh.triangles)

    meta_path = os.path.join(output_dir, "postprocess_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    outputs["meta"] = meta_path

    print(f"Post-processing complete. Outputs in: {output_dir}")
    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-process AMB3R point cloud")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to point_cloud.ply from AMB3R",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed outputs",
    )
    parser.add_argument("--statistical_nb", type=int, default=20)
    parser.add_argument("--statistical_std", type=float, default=2.0)
    parser.add_argument("--no_poisson", action="store_true")
    parser.add_argument("--no_outlier_removal", action="store_true",
                        help="Skip statistical outlier removal")
    parser.add_argument("--poisson_depth", type=int, default=9)
    parser.add_argument("--segmentation_dir", type=str, default=None,
                        help="SAM3 segmentation output dir for person filtering")
    parser.add_argument("--amb3r_npz", type=str, default=None,
                        help="Path to point_cloud.npz from AMB3R")
    parser.add_argument("--amb3r_images_dir", type=str, default=None,
                        help="Path to AMB3R-resolution images dir")
    args = parser.parse_args()

    full_postprocess(
        args.input,
        args.output_dir,
        statistical_nb=args.statistical_nb,
        statistical_std=args.statistical_std,
        do_poisson=not args.no_poisson,
        poisson_depth=args.poisson_depth,
        skip_outlier_removal=args.no_outlier_removal,
        segmentation_dir=args.segmentation_dir,
        amb3r_npz_path=args.amb3r_npz,
        amb3r_images_dir=args.amb3r_images_dir,
    )
