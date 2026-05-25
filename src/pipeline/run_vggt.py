"""
Worker script: Run VGGT for 3D reconstruction (arbitrary-scale).
Executed inside the 'vv_vggt' conda environment.

VGGT (Visual Geometry Grounded Transformer, CVPR 2025 Best Paper) produces
dense 3D point maps, depth maps, and camera poses from multi-view images.
Unlike AMB3R, VGGT outputs are in ARBITRARY scale (not metric). Use an
ArUco marker or known reference to calibrate.

Usage:
    conda activate vv_vggt
    python src/pipeline/run_vggt.py \
        --image_dir data/input/patient_001 \
        --output_dir data/output/patient_001/reconstruction \
        [--max_images 4]

Outputs (saved to output_dir):
    - point_cloud.ply       : Colored point cloud (backend-equivalent)
    - point_cloud.npz       : Raw arrays (points_per_frame, conf, images, poses)
    - amb3r_images/          : VGGT-resolution images (for pose re-detection)
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VGGT_DIR = os.path.join(PROJECT_DIR, "repos", "vggt")
sys.path.insert(0, VGGT_DIR)

import torch


def run_reconstruction(image_dir, output_dir, max_images=4, conf_threshold=0.0,
                       mode="pad", per_frame=False):
    """Run VGGT reconstruction on images in image_dir.

    Args:
        image_dir: Directory containing input images.
        output_dir: Directory to save outputs.
        max_images: Maximum number of images to use.
        conf_threshold: Confidence threshold for point filtering.
        mode: Image preprocessing mode:
            - "pad" (default): Largest dimension scaled to 518 px, smaller
              dimension padded with white to make 518x518. PRESERVES ALL PIXELS.
              Required for accurate mask alignment downstream.
            - "crop": Width set to 518 px, height center-cropped if larger.
              Drops pixels for portrait images.
        per_frame: If True, run VGGT one frame at a time (much slower, uses
            ~3x less memory at peak). Loses multi-view fusion so cross-frame
            consistency is reduced. Use only when batch mode runs OOM.
    """
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from PIL import Image as PILImage

    os.makedirs(output_dir, exist_ok=True)

    # Collect images
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    image_files = sorted(set(image_files))

    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    # Limit number of images
    if len(image_files) > max_images:
        print(f"Using {max_images} of {len(image_files)} images")
        # Sample evenly
        indices = np.linspace(0, len(image_files) - 1, max_images, dtype=int)
        image_files = [image_files[i] for i in indices]

    print(f"Using {len(image_files)} images for VGGT reconstruction (mode='{mode}')")

    # Compute the preprocessing transform per image so we can match it on
    # downstream masks (segmentation must align with the point map).
    # The transform: image at original (W, H) → (1) scale to fit 518 box,
    # (2) pad with offset (left, top) to 518×518.
    preprocess_transforms = []
    for img_path in image_files:
        with PILImage.open(img_path) as im:
            W, H = im.size
        if mode == "pad":
            # Largest dim → 518; pad smaller dim
            target = 518
            if W > H:
                new_W = target
                new_H = int(round(H * target / W))
            else:
                new_H = target
                new_W = int(round(W * target / H))
            # VGGT.load_fn pads to square=518 with white background.
            # Padding is centered (top + bottom, or left + right).
            pad_top = (target - new_H) // 2
            pad_left = (target - new_W) // 2
            preprocess_transforms.append({
                "filename": os.path.basename(img_path),
                "orig_size": [int(W), int(H)],
                "scaled_size": [int(new_W), int(new_H)],
                "padded_size": [int(target), int(target)],
                "pad_left": int(pad_left),
                "pad_top": int(pad_top),
                "scale": float(new_W / W),  # = new_H / H
                "mode": "pad",
            })
        else:
            # mode == "crop": width = 518, height is scaled then center-cropped if > 518
            target = 518
            new_W = target
            new_H = int(round(H * target / W))
            # VGGT.load_fn requires divisible by 14
            if new_H % 14 != 0:
                new_H = (new_H + 13) // 14 * 14
            if new_H > target:
                start_y = (new_H - target) // 2
                preprocess_transforms.append({
                    "filename": os.path.basename(img_path),
                    "orig_size": [int(W), int(H)],
                    "scaled_size": [int(new_W), int(new_H)],
                    "cropped_size": [int(target), int(target)],
                    "crop_top": int(start_y),
                    "crop_left": 0,
                    "scale": float(new_W / W),
                    "mode": "crop",
                })
            else:
                preprocess_transforms.append({
                    "filename": os.path.basename(img_path),
                    "orig_size": [int(W), int(H)],
                    "scaled_size": [int(new_W), int(new_H)],
                    "cropped_size": [int(new_W), int(new_H)],
                    "crop_top": 0,
                    "crop_left": 0,
                    "scale": float(new_W / W),
                    "mode": "crop",
                })

    # Determine dtype based on GPU capability
    device = "cuda"
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # Load model in fp32 (some LayerNorm layers require fp32 weights).
    # Inference dtype is handled by torch.cuda.amp.autocast below — that
    # auto-converts ops to bfloat16 where safe and keeps fp32 where needed.
    print("Loading VGGT model from HuggingFace (fp32 weights, autocast inference)...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # Load and preprocess images
    print(f"Preprocessing images (mode='{mode}')...")
    images = load_and_preprocess_images(image_files, mode=mode).to(device)
    # images shape: (S, 3, H, W) where H, W are VGGT resolution

    S, C, H_vggt, W_vggt = images.shape
    print(f"VGGT input: {S} frames, {H_vggt}x{W_vggt}")

    # Add batch dimension
    images = images.unsqueeze(0)  # (1, S, 3, H, W)

    # Run inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"  GPU memory before VGGT inference: "
              f"{(total_b - free_b)/1e9:.1f}/{total_b/1e9:.1f} GB used")

    if per_frame:
        # Sequential per-frame mode: process each image individually.
        # Trade-off: ~3x less peak memory, but no multi-view fusion → cross-frame
        # depth consistency suffers. Outputs are concatenated frame-by-frame.
        print(f"Running VGGT inference (PER-FRAME mode, {S} frames)...")
        wp_list, wpc_list, d_list, dc_list, pe_list, ii_list = ([] for _ in range(6))
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                for fi in range(S):
                    print(f"  Frame {fi+1}/{S}")
                    one = images[:, fi:fi+1]  # (1, 1, 3, H, W)
                    pred = model(one)
                    wp_list.append(pred["world_points"][0, 0].cpu().numpy())
                    wpc_list.append(pred["world_points_conf"][0, 0].cpu().numpy())
                    d_list.append(pred["depth"][0, 0].cpu().numpy())
                    dc_list.append(pred["depth_conf"][0, 0].cpu().numpy())
                    pe_list.append(pred["pose_enc"][0, 0].cpu())
                    ii_list.append(pred["images"][0, 0].permute(1, 2, 0).cpu().numpy())
                    del pred
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        world_points = np.stack(wp_list, axis=0)
        world_points_conf = np.stack(wpc_list, axis=0)
        depth_maps = np.stack(d_list, axis=0)
        depth_conf = np.stack(dc_list, axis=0)
        pose_enc = torch.stack(pe_list, dim=0)
        input_images = np.stack(ii_list, axis=0)
    else:
        # Batched multi-view mode (default, gives best multi-view fusion)
        print(f"Running VGGT inference (BATCHED, {S} frames)...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        world_points = predictions["world_points"][0].cpu().numpy()
        world_points_conf = predictions["world_points_conf"][0].cpu().numpy()
        depth_maps = predictions["depth"][0].cpu().numpy()
        depth_conf = predictions["depth_conf"][0].cpu().numpy()
        pose_enc = predictions["pose_enc"][0].cpu()
        input_images = predictions["images"][0].permute(0, 2, 3, 1).cpu().numpy()
        del predictions

    # Free the model and GPU tensors — we're done with the GPU
    del model
    del images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Decode camera poses
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        pose_enc.unsqueeze(0), (H_vggt, W_vggt)
    )
    extrinsic = extrinsic[0].numpy()  # (S, 3, 4)
    intrinsic = intrinsic[0].numpy()  # (S, 3, 3)

    T = world_points.shape[0]
    H_out = world_points.shape[1]
    W_out = world_points.shape[2]
    print(f"VGGT output: {T} frames, point maps {H_out}x{W_out}")

    # Point map stats
    all_pts = world_points.reshape(-1, 3)
    all_conf = world_points_conf.reshape(-1)
    valid = np.linalg.norm(all_pts, axis=-1) > 0.01
    print(f"Total points: {len(all_pts)}, valid: {valid.sum()}")
    if valid.sum() > 0:
        vp = all_pts[valid]
        print(f"Point ranges: X[{vp[:,0].min():.3f}, {vp[:,0].max():.3f}] "
              f"Y[{vp[:,1].min():.3f}, {vp[:,1].max():.3f}] "
              f"Z[{vp[:,2].min():.3f}, {vp[:,2].max():.3f}]")

    # Flatten for PLY export
    pts_flat = world_points.reshape(-1, 3)
    conf_flat = world_points_conf.reshape(-1)
    colors_flat = input_images.reshape(-1, 3)

    # Filter by confidence
    if conf_threshold > 0:
        mask = conf_flat > conf_threshold
    else:
        mask = np.linalg.norm(pts_flat, axis=-1) > 0.01
    pts_filtered = pts_flat[mask]
    colors_filtered = colors_flat[mask]
    conf_filtered = conf_flat[mask]

    # Save PLY
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_filtered)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_filtered, 0, 1))
    ply_path = os.path.join(output_dir, "point_cloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved point cloud: {ply_path} ({len(pts_filtered)} points)")

    # Also save frontend-equivalent (same as main for VGGT, no backend)
    ply_front_path = os.path.join(output_dir, "point_cloud_front.ply")
    o3d.io.write_point_cloud(ply_front_path, pcd)

    # Save NPZ (compatible format with AMB3R for downstream pipeline)
    # Reshape conf to (T, H, W, 1) to match AMB3R format
    conf_per_frame = world_points_conf[:, :, :, np.newaxis]  # (T, H, W, 1)

    npz_path = os.path.join(output_dir, "point_cloud.npz")
    np.savez_compressed(
        npz_path,
        points=pts_flat,
        colors=colors_flat,
        confidence=conf_flat,
        poses=extrinsic,
        points_per_frame=world_points,       # (T, H, W, 3)
        conf_per_frame=conf_per_frame,       # (T, H, W, 1)
        images_per_frame=input_images,       # (T, H, W, 3)
        depth_per_frame=depth_maps,          # (T, H, W, 1)
        intrinsic=intrinsic,                 # (T, 3, 3)
        extrinsic=extrinsic,                 # (T, 3, 4)
    )
    print(f"Saved raw data: {npz_path}")

    # Save VGGT-resolution images using the ORIGINAL input filenames so the
    # downstream stages (pose, picker, anterior_frame lookup) all agree on
    # which frame is which. Previously these were saved with sequential
    # frame_000…frame_NNN.jpg names regardless of input, which caused a
    # disastrous mismatch when the recon-selected subset was non-contiguous
    # (image_files_in_order said [frame_000, frame_001, frame_003, ...]
    # but disk only had frame_000.jpg…frame_019.jpg sequentially — so a
    # user picking 'frame_010.jpg' got a different image than expected).
    imgs_dir = os.path.join(output_dir, "amb3r_images")  # Keep same name for compatibility
    os.makedirs(imgs_dir, exist_ok=True)
    # Clear any stale files first so leftovers from a prior run with
    # different frame counts can't pollute the directory.
    for old in os.listdir(imgs_dir):
        old_p = os.path.join(imgs_dir, old)
        if os.path.isfile(old_p):
            os.unlink(old_p)
    from PIL import Image as PILImage

    for t in range(T):
        img_arr = input_images[t]
        # Convert from [0,1] to [0,255]
        if img_arr.max() <= 1.0:
            img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
        else:
            img_arr = img_arr.clip(0, 255).astype(np.uint8)
        # Match the same name the orchestrator and picker use — derived from
        # the t-th original input file.
        orig_name = os.path.basename(image_files[t])
        PILImage.fromarray(img_arr).save(os.path.join(imgs_dir, orig_name))
    print(f"Saved {T} VGGT-resolution images to {imgs_dir} "
          f"(using original filenames)")

    # Save metadata
    meta = {
        "model": "VGGT-1B",
        "scale": "arbitrary (NOT metric, use ArUco or reference for calibration)",
        "num_images": int(T),
        "resolution_hw": [int(H_out), int(W_out)],
        "num_points": int(len(pts_filtered)),
        "conf_threshold": conf_threshold,
        "has_metric_depth": False,
        "amb3r_images_dir": imgs_dir,
        "preprocess_mode": mode,
        "preprocess_transforms": preprocess_transforms,
        "image_files_in_order": [os.path.basename(p) for p in image_files],
    }
    meta_path = os.path.join(output_dir, "reconstruction_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    print("VGGT reconstruction complete.")
    print("NOTE: VGGT outputs are in ARBITRARY scale. Use ArUco calibration for metric measurements.")
    return ply_path, npz_path


def main():
    parser = argparse.ArgumentParser(description="Run VGGT 3D reconstruction")
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save outputs"
    )
    parser.add_argument(
        "--max_images", type=int, default=4,
        help="Maximum number of images to use (default: 4)",
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.0,
        help="Confidence threshold for point filtering (default: 0.0)",
    )
    parser.add_argument(
        "--mode", type=str, default="pad", choices=["pad", "crop"],
        help="Preprocessing: 'pad' (default, preserves all pixels by padding "
             "to 518x518) or 'crop' (center-crop to 518 wide).",
    )
    parser.add_argument(
        "--per_frame", action="store_true",
        help="Process frames one at a time (~3x less peak GPU memory but "
             "loses multi-view fusion). Use only when batched mode runs OOM.",
    )
    args = parser.parse_args()

    run_reconstruction(
        args.image_dir, args.output_dir,
        max_images=args.max_images,
        conf_threshold=args.conf_threshold,
        mode=args.mode,
        per_frame=args.per_frame,
    )


if __name__ == "__main__":
    main()
