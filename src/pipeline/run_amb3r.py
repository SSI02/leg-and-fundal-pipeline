"""
Worker script: Run AMB3R for metric-scale 3D reconstruction.
Executed inside the 'amb3r' conda environment.

Usage:
    conda activate amb3r
    python src/pipeline/run_amb3r.py --image_dir <path> --output_dir <path> [--max_images 4]

Outputs (saved to output_dir):
    - point_cloud.ply       : Colored metric-scale point cloud
    - point_cloud.npz       : Raw arrays (points, colors, confidence, poses)
    - point_cloud_front.ply : Frontend-only point cloud (for comparison)
"""

import os
import sys
import json
import argparse
import numpy as np

# Add AMB3R repo to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
AMB3R_DIR = os.path.join(PROJECT_DIR, "repos", "amb3r")
sys.path.insert(0, AMB3R_DIR)

import torch
import open3d as o3d
from torch.utils.data import DataLoader


def run_reconstruction(image_dir, output_dir, max_images=4, conf_threshold=0.0):
    """Run AMB3R reconstruction on images in image_dir."""

    from amb3r.model import AMB3R
    from amb3r.datasets import Demo

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    ckpt_path = os.path.join(AMB3R_DIR, "checkpoints", "amb3r.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"AMB3R checkpoint not found at {ckpt_path}. "
            "Download from: https://drive.google.com/file/d/14x0WW2rUE_he2hUEouP6ywSRnlJDeLel"
        )

    print(f"Loading AMB3R model from {ckpt_path}...")
    model = AMB3R()
    model.load_weights(ckpt_path)
    model.cuda()
    model.eval()

    # Load images
    print(f"Loading images from {image_dir} (max: {max_images})...")
    data = Demo(
        ROOT=image_dir,
        resolution=(518, 392),
        num_seq=1,
        full_video=True,
        kf_every=1,
        disable_crop=False,
        max_images=max_images,
    )
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)
    batch = next(iter(dataloader))
    _, views_all = batch

    for key in views_all.keys():
        views_all[key] = views_all[key].cuda()

    # Run inference
    print("Running AMB3R inference (frontend + backend)...")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            res = model(views_all)

    # Extract backend (refined) results
    pts_backend = res[-1]["world_points"].cpu().numpy().reshape(-1, 3)
    conf_backend = res[-1]["world_points_conf"].cpu().numpy().reshape(-1)
    conf_sig_backend = (conf_backend - 1) / conf_backend
    color_backend = (
        res[-1]["images"].permute(0, 1, 3, 4, 2).reshape(-1, 3).cpu().numpy()
    )
    poses_backend = res[-1]["pose"].cpu().numpy()

    # Extract frontend-only results (for comparison)
    pts_frontend = res[0]["world_points"].cpu().numpy().reshape(-1, 3)
    conf_frontend = res[0]["world_points_conf"].cpu().numpy().reshape(-1)
    conf_sig_frontend = (conf_frontend - 1) / conf_frontend
    color_frontend = (
        res[0]["images"].permute(0, 1, 3, 4, 2).reshape(-1, 3).cpu().numpy()
    )
    poses_frontend = res[0]["pose"].cpu().numpy()

    # Extract per-frame data (for segmentation masking later)
    pts_per_frame = res[-1]["world_points"][0].cpu().numpy()  # (T, H, W, 3)
    conf_per_frame = res[-1]["world_points_conf"][0].cpu().numpy()  # (T, H, W, 1)
    images_per_frame = (
        res[-1]["images"][0].permute(0, 2, 3, 1).cpu().numpy()
    )  # (T, H, W, 3)

    # Also extract metric depth if available
    depth_metric = None
    if "depth_metric" in res[-1]:
        depth_metric = res[-1]["depth_metric"][0].cpu().numpy()  # (T, H, W, 1)

    # Filter by confidence
    mask = conf_sig_backend > conf_threshold
    pts_filtered = pts_backend[mask]
    color_filtered = color_backend[mask]
    conf_filtered = conf_sig_backend[mask]

    # Save backend point cloud as PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_filtered)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(color_filtered, 0, 1))
    ply_path = os.path.join(output_dir, "point_cloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved backend point cloud: {ply_path} ({len(pts_filtered)} points)")

    # Save frontend point cloud for comparison
    mask_front = conf_sig_frontend > conf_threshold
    pcd_front = o3d.geometry.PointCloud()
    pcd_front.points = o3d.utility.Vector3dVector(pts_frontend[mask_front])
    pcd_front.colors = o3d.utility.Vector3dVector(
        np.clip(color_frontend[mask_front], 0, 1)
    )
    ply_front_path = os.path.join(output_dir, "point_cloud_front.ply")
    o3d.io.write_point_cloud(ply_front_path, pcd_front)
    print(f"Saved frontend point cloud: {ply_front_path} ({mask_front.sum()} points)")

    # Save raw data as NPZ for downstream processing
    npz_path = os.path.join(output_dir, "point_cloud.npz")
    save_dict = {
        "points": pts_backend,
        "colors": color_backend,
        "confidence": conf_sig_backend,
        "poses": poses_backend,
        "points_per_frame": pts_per_frame,
        "conf_per_frame": conf_per_frame,
        "images_per_frame": images_per_frame,
    }
    if depth_metric is not None:
        save_dict["depth_metric"] = depth_metric
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved raw data: {npz_path}")

    # Save AMB3R-resolution images for pose re-detection.
    # These are the exact images AMB3R saw, so pose keypoints on these
    # will map 1:1 to the point map coordinates.
    # IMPORTANT: use the ORIGINAL input filenames (not sequential
    # frame_000…NNN.jpg). Sequential names cause a name/content mismatch
    # when the recon subset is non-contiguous — the anterior frame the
    # user picks no longer points to the same image the pipeline reads.
    amb3r_imgs_dir = os.path.join(output_dir, "amb3r_images")
    os.makedirs(amb3r_imgs_dir, exist_ok=True)
    # Clear stale leftovers
    for old in os.listdir(amb3r_imgs_dir):
        op = os.path.join(amb3r_imgs_dir, old)
        if os.path.isfile(op):
            os.unlink(op)
    from PIL import Image as PILImage
    import glob
    # Build the input file list in deterministic order so amb3r_images
    # filenames mirror the input.
    input_files = sorted(
        f for f in glob.glob(os.path.join(image_dir, "*"))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if len(input_files) < pts_per_frame.shape[0]:
        print(f"WARNING: AMB3R loaded {pts_per_frame.shape[0]} frames but "
              f"input_dir has {len(input_files)} — filenames may be misaligned.")

    for t in range(pts_per_frame.shape[0]):
        img_arr = images_per_frame[t]
        if img_arr.min() < 0:
            img_arr = (img_arr + 1) / 2
        img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
        if t < len(input_files):
            out_name = os.path.basename(input_files[t])
        else:
            out_name = f"frame_{t:03d}.jpg"
        PILImage.fromarray(img_arr).save(
            os.path.join(amb3r_imgs_dir, out_name)
        )
    print(f"Saved {pts_per_frame.shape[0]} AMB3R-resolution images to "
          f"{amb3r_imgs_dir} (using original filenames)")

    # Save metadata — include image_files_in_order so downstream stages
    # know the T-axis ↔ filename mapping (must match what we just wrote
    # to amb3r_images/).
    image_files_in_order = [
        os.path.basename(input_files[t]) if t < len(input_files)
        else f"frame_{t:03d}.jpg"
        for t in range(pts_per_frame.shape[0])
    ]
    meta = {
        "num_images": int(pts_per_frame.shape[0]),
        "resolution_hw": [int(pts_per_frame.shape[1]), int(pts_per_frame.shape[2])],
        "num_points_backend": int(len(pts_filtered)),
        "num_points_frontend": int(mask_front.sum()),
        "conf_threshold": conf_threshold,
        "has_metric_depth": depth_metric is not None,
        "amb3r_images_dir": amb3r_imgs_dir,
        "image_files_in_order": image_files_in_order,
    }
    meta_path = os.path.join(output_dir, "reconstruction_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    print("AMB3R reconstruction complete.")
    return ply_path, npz_path


def main():
    parser = argparse.ArgumentParser(description="Run AMB3R 3D reconstruction")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=4,
        help="Maximum number of images to use (default: 4)",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for point filtering (default: 0.0)",
    )
    args = parser.parse_args()

    run_reconstruction(
        args.image_dir,
        args.output_dir,
        max_images=args.max_images,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
