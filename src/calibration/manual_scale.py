"""
Helper for loading and applying manual scale calibration from scale_picker.py.

The scale_calibration.json produced by scale_picker.py is the primary
ground-truth metric scale source — it has a separate cm/pixel for each image
based on the user clicking known-distance points.

This module provides:
    - load_manual_scale(path) → dict of image_name → scale_cm_per_pixel
    - compute_3d_scale_from_points(points_per_frame, p1, p2, real_distance_cm)
        → factor to multiply 3D points by to get true metric units
    - apply_3d_scale_to_pointcloud(npz_path, scale_factor)
"""

import os
import json
import numpy as np


def transform_point_to_recon_space(point_xy, transform):
    """Apply the same VGGT/AMB3R preprocessing transform to a single point.

    For pad mode (default VGGT belly):
        scaled_xy = original_xy * scale
        recon_xy  = scaled_xy + (pad_left, pad_top)
    For crop mode:
        scaled_xy = original_xy * scale
        recon_xy  = scaled_xy - (crop_left, crop_top)

    Args:
        point_xy: (x, y) in original image pixel coordinates.
        transform: dict from recon_meta.preprocess_transforms[i].

    Returns:
        (x_recon, y_recon) — pixel coords in the reconstruction's canvas.
    """
    if transform is None:
        return point_xy
    scale = transform.get("scale", 1.0)
    sx = float(point_xy[0]) * scale
    sy = float(point_xy[1]) * scale
    if transform.get("mode") == "pad":
        sx += transform.get("pad_left", 0)
        sy += transform.get("pad_top", 0)
    else:  # crop
        sx -= transform.get("crop_left", 0)
        sy -= transform.get("crop_top", 0)
    return (sx, sy)


def load_manual_scale(path):
    """Load per-image scale calibration JSON.

    Args:
        path: Path to scale_calibration.json from scale_picker.py.

    Returns:
        Dict of {image_basename: scale_cm_per_pixel} or empty dict if missing.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: v["scale_cm_per_pixel"] for k, v in data.items()
            if "scale_cm_per_pixel" in v}


def load_manual_scale_full(path):
    """Load full per-image calibration (with click points and metadata)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def compute_3d_scale_factor(amb3r_npz_path, scale_calib_path,
                             image_names_in_order=None,
                             recon_meta_path=None):
    """Compute a global 3D scale factor by using clicked points to find the
    true distance and comparing against the AMB3R/VGGT 3D distance.

    For each image with calibration:
      1. Apply VGGT/AMB3R's preprocessing transform (scale + pad/crop) to
         click pixel coordinates so they land on the right point-map pixel.
      2. Look up the 3D positions of p1 and p2 in the per-frame point map.
      3. Compute 3D Euclidean distance.
      4. Compute scale = real_distance_meters / 3D_distance_in_amb3r_units.
    Median across all calibrated images for robustness.

    Args:
        amb3r_npz_path: Path to point_cloud.npz from AMB3R/VGGT.
        scale_calib_path: Path to scale_calibration.json.
        image_names_in_order: List of input image basenames in the EXACT order
            VGGT/AMB3R processed them. If recon_meta_path is supplied this is
            ignored (overridden by recon_meta.image_files_in_order).
        recon_meta_path: Path to reconstruction_meta.json. PREFERRED — provides
            the canonical frame ordering AND per-frame preprocess transforms,
            so click coordinates are mapped to the recon canvas correctly.

    Returns:
        Dict with scale_factor (median), mean, std, per-frame details.
    """
    if not os.path.exists(amb3r_npz_path) or not os.path.exists(scale_calib_path):
        return None

    npz = np.load(amb3r_npz_path, allow_pickle=True)
    pts_per_frame = npz["points_per_frame"]  # (T, H, W, 3)
    T, H_amb, W_amb, _ = pts_per_frame.shape

    # Prefer recon_meta for both ordering and pad/crop transforms
    transforms_by_filename = {}
    if recon_meta_path and os.path.exists(recon_meta_path):
        with open(recon_meta_path) as f:
            recon_meta = json.load(f)
        for t in recon_meta.get("preprocess_transforms", []):
            transforms_by_filename[t["filename"]] = t
        meta_order = recon_meta.get("image_files_in_order")
        if meta_order:
            image_names_in_order = list(meta_order)

    if image_names_in_order is None:
        # Last-resort fallback: use the calibration JSON's keys in sorted order.
        with open(scale_calib_path) as f:
            calibs_tmp = json.load(f)
        image_names_in_order = sorted(calibs_tmp.keys())

    with open(scale_calib_path) as f:
        calibs = json.load(f)

    factors = []
    details = []

    for frame_idx, img_name in enumerate(image_names_in_order):
        if frame_idx >= T:
            break
        if img_name not in calibs:
            continue

        calib = calibs[img_name]
        p1_orig = calib["p1"]
        p2_orig = calib["p2"]
        real_distance_cm = calib["real_distance_cm"]

        # ── Apply pad/crop-aware projection ─────────────────────────────
        # If we have the recon_meta transform, use it (correct for pad mode).
        # Otherwise fall back to a naive stretch (only correct when no padding
        # or cropping was applied — i.e., the input was already 518×518).
        transform = transforms_by_filename.get(img_name)
        if transform is not None:
            p1_recon = transform_point_to_recon_space(p1_orig, transform)
            p2_recon = transform_point_to_recon_space(p2_orig, transform)
        else:
            # Naive scale fallback — only correct when no preprocess transform
            try:
                from PIL import Image as PILImage
                possible_dirs = [
                    os.path.dirname(amb3r_npz_path).replace("/reconstruction", "/../input"),
                    os.path.dirname(amb3r_npz_path).replace("reconstruction", "input"),
                    os.path.join(os.path.dirname(amb3r_npz_path), "..", "recon_frames"),
                ]
                orig_path = None
                for d in possible_dirs:
                    p = os.path.join(d, img_name)
                    if os.path.exists(p):
                        orig_path = p
                        break
                if orig_path:
                    W_orig, H_orig = PILImage.open(orig_path).size
                else:
                    W_orig, H_orig = W_amb, H_amb
            except Exception:
                W_orig, H_orig = W_amb, H_amb
            sx_n = W_amb / W_orig
            sy_n = H_amb / H_orig
            p1_recon = (p1_orig[0] * sx_n, p1_orig[1] * sy_n)
            p2_recon = (p2_orig[0] * sx_n, p2_orig[1] * sy_n)

        p1_amb = (int(round(p1_recon[0])), int(round(p1_recon[1])))
        p2_amb = (int(round(p2_recon[0])), int(round(p2_recon[1])))

        # Clamp
        p1_amb = (max(0, min(p1_amb[0], W_amb - 1)),
                  max(0, min(p1_amb[1], H_amb - 1)))
        p2_amb = (max(0, min(p2_amb[0], W_amb - 1)),
                  max(0, min(p2_amb[1], H_amb - 1)))

        # Get 3D points (averaged in 3x3 patch for robustness)
        def lookup_3d(px, py):
            patch = pts_per_frame[frame_idx,
                                  max(0, py-1):py+2,
                                  max(0, px-1):px+2]  # (h, w, 3)
            valid = np.linalg.norm(patch, axis=-1) > 0.001
            if valid.sum() == 0:
                return None
            return patch[valid].mean(axis=0)

        pt1 = lookup_3d(p1_amb[0], p1_amb[1])
        pt2 = lookup_3d(p2_amb[0], p2_amb[1])

        if pt1 is None or pt2 is None:
            details.append({
                "frame": frame_idx, "image": img_name,
                "error": "3D lookup failed (zero points)"
            })
            continue

        d3 = float(np.linalg.norm(pt2 - pt1))
        if d3 < 1e-6:
            details.append({
                "frame": frame_idx, "image": img_name,
                "error": "3D distance zero"
            })
            continue

        # AMB3R outputs in meters; real_distance is cm; we want cm-units output
        # so factor = real_cm / d3_m / 100 * 100 = real_cm / (d3 * 100)
        # But to keep units consistent: multiply 3D coords (in meters) by a factor
        # to get meters-true-scale, OR convert to cm and apply factor.
        # We'll output a factor that scales meters → meters-true-scale:
        #   true_d3_m = real_distance_cm / 100
        #   factor = true_d3_m / d3_m
        true_d3_m = real_distance_cm / 100.0
        factor = true_d3_m / d3

        factors.append(factor)
        details.append({
            "frame": frame_idx,
            "image": img_name,
            "p1_orig": p1_orig,
            "p2_orig": p2_orig,
            "p1_amb3r": list(p1_amb),
            "p2_amb3r": list(p2_amb),
            "amb3r_3d_distance_m": round(d3, 6),
            "real_distance_cm": real_distance_cm,
            "scale_factor_to_apply": round(factor, 6),
        })

    if not factors:
        return None

    factors_arr = np.array(factors)
    median_f = float(np.median(factors_arr))
    mean_f = float(np.mean(factors_arr))
    std_f = float(np.std(factors_arr))

    # Use the MEDIAN, not the mean. With ≤5 frames and any single bad lookup
    # (e.g. clicked pixel landed on an occluded surface or a depth hole), the
    # mean gets dragged off by an order of magnitude. Median is robust to one
    # or two bad frames out of ≤5.
    chosen = median_f

    # Warn loudly if the spread is large — usually means one or more frames
    # had bad 3D lookups and you should inspect the per-frame `details`.
    if mean_f > 0 and abs(mean_f - median_f) / max(median_f, 1e-6) > 0.20:
        print(f"  WARNING: per-frame scale factors are inconsistent "
              f"(mean={mean_f:.4f}, median={median_f:.4f}, std={std_f:.4f}). "
              f"Using median. Inspect details:")
        for d in details:
            if "scale_factor_to_apply" in d:
                print(f"    frame {d['frame']} ({d['image']}): "
                      f"factor={d['scale_factor_to_apply']:.4f}, "
                      f"3D_dist={d['amb3r_3d_distance_m']*1000:.2f}mm")
            elif "error" in d:
                print(f"    frame {d['frame']} ({d['image']}): {d['error']}")

    return {
        "scale_factor": chosen,
        "scale_factor_mean": mean_f,
        "scale_factor_median": median_f,
        "scale_factor_std": std_f,
        "per_frame_factors": factors,
        "details": details,
        "n_frames_used": len(factors),
    }


def apply_3d_scale_to_npz(npz_path, scale_factor, output_path=None):
    """Apply a scale factor to all 3D points in a point_cloud.npz file.

    Args:
        npz_path: Path to input .npz from AMB3R/VGGT.
        scale_factor: Multiplier to apply to all 3D coordinates.
        output_path: If provided, save scaled .npz to this path.
            If None, overwrites the input.

    Returns:
        Dict of scaled arrays.
    """
    npz = np.load(npz_path, allow_pickle=True)
    scaled = {}
    point_keys = ["points", "points_per_frame", "depth_per_frame"]
    pose_keys = ["poses", "extrinsic"]
    for k in npz.files:
        arr = npz[k]
        if k in point_keys:
            scaled[k] = arr * scale_factor
        elif k in pose_keys:
            # Translations live in last column for 4x4 / 3x4 poses
            arr = arr.copy().astype(np.float32)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                # (T, 3, 4) extrinsic: scale translation column
                if arr.shape[-1] == 4:
                    arr[..., :3, 3] *= scale_factor
                else:
                    pass
            scaled[k] = arr
        else:
            scaled[k] = arr

    out = output_path or npz_path
    np.savez_compressed(out, **scaled)
    return scaled


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply manual scale to 3D reconstruction")
    parser.add_argument("--npz", required=True, help="point_cloud.npz from AMB3R/VGGT")
    parser.add_argument("--scale_calibration", required=True,
                        help="scale_calibration.json from scale_picker.py")
    parser.add_argument("--image_order", required=True,
                        help="Comma-separated list of image filenames in AMB3R frame order")
    parser.add_argument("--output", help="Output path (default: overwrite input)")
    args = parser.parse_args()

    image_order = [s.strip() for s in args.image_order.split(",")]
    result = compute_3d_scale_factor(args.npz, args.scale_calibration, image_order)
    if result is None:
        print("ERROR: Could not compute 3D scale factor")
        exit(1)
    print(f"Mean factor: {result['scale_factor']:.4f} (median {result['scale_factor_median']:.4f}, "
          f"std {result['scale_factor_std']:.4f}, n={result['n_frames_used']} frames)")
    apply_3d_scale_to_npz(args.npz, result["scale_factor"], args.output)
    print(f"Applied scale factor {result['scale_factor']:.4f} to {args.npz}")
