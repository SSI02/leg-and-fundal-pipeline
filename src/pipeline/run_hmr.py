"""
Worker script: Run HMR2.0 (4DHumans) for SMPL body model fitting.
Executed inside the 'hmr_env' conda environment.

Usage:
    conda activate hmr_env
    python src/pipeline/run_hmr.py --image_dir <path> --output_dir <path>

Outputs (saved to output_dir):
    - hmr_results.npz : SMPL parameters, 3D joints, vertices per person
    - meshes/         : OBJ mesh files per person (optional)
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
HMR_DIR = os.path.join(PROJECT_DIR, "repos", "4D-Humans")
sys.path.insert(0, HMR_DIR)


def run_hmr(image_dir, output_dir):
    """Run HMR2.0 on all images and save SMPL parameters + joints."""
    import torch
    from pathlib import Path

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

    print(f"Found {len(image_files)} images")

    # Load HMR2.0 model
    from hmr2.models import load_hmr2, download_models, DEFAULT_CHECKPOINT
    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
    from hmr2.utils.renderer import Renderer, cam_crop_to_full

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading HMR2.0 model...")
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()

    # Load detector
    from detectron2.config import LazyConfig
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor

    detectron2_cfg = model_zoo.get_config(
        "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
    )
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
    detector = DefaultPredictor(detectron2_cfg)

    all_joints = []
    all_vertices = []
    all_betas = []
    all_cam_t = []
    all_image_names = []

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        print(f"Processing: {img_name}")

        import cv2
        img_cv2 = cv2.imread(str(img_path))

        # Detect persons
        det_out = detector(img_cv2)
        instances = det_out["instances"]
        # Filter to person class (class 0 in COCO)
        valid = instances.pred_classes == 0
        instances = instances[valid]
        bboxes = instances.pred_boxes.tensor.cpu().numpy()

        if len(bboxes) == 0:
            print(f"  No persons detected in {img_name}")
            continue

        # Create dataset for HMR
        dataset = ViTDetDataset(model_cfg, img_cv2, bboxes)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=len(bboxes), shuffle=False, num_workers=0
        )

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            # Extract results
            pred_cam = out["pred_cam"]  # (N, 3)
            betas = out["pred_smpl_params"]["betas"]  # (N, 10)
            joints_3d = out["pred_keypoints_3d"]  # (N, 44, 3)
            vertices = out["pred_vertices"]  # (N, 6890, 3)

            # Convert weak-perspective camera to full camera translation
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max(dim=1)[0]
            cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal)

            n_persons = len(betas)
            for i in range(n_persons):
                all_joints.append(joints_3d[i].cpu().numpy())
                all_vertices.append(vertices[i].cpu().numpy())
                all_betas.append(betas[i].cpu().numpy())
                all_cam_t.append(cam_t[i].cpu().numpy())
                all_image_names.append(img_name)

            print(f"  Detected {n_persons} person(s)")

    if not all_joints:
        print("WARNING: No persons detected in any image")
        np.savez(
            os.path.join(output_dir, "hmr_results.npz"),
            joints_3d=np.array([]),
            vertices=np.array([]),
            betas=np.array([]),
            cam_t=np.array([]),
            image_names=np.array([]),
        )
        return

    # Save results
    npz_path = os.path.join(output_dir, "hmr_results.npz")
    np.savez(
        npz_path,
        joints_3d=np.stack(all_joints),  # (N_total, 44, 3)
        vertices=np.stack(all_vertices),  # (N_total, 6890, 3)
        betas=np.stack(all_betas),  # (N_total, 10)
        cam_t=np.stack(all_cam_t),  # (N_total, 3)
        image_names=np.array(all_image_names),
    )
    print(f"\nSaved HMR results: {npz_path}")
    print(f"Total persons: {len(all_joints)}")

    # Save metadata
    meta = {
        "num_persons": len(all_joints),
        "num_images": len(image_files),
        "joint_format": "openpose_25 + 19_extra (44 total)",
        "joint_indices": {
            "right_hip": 9, "right_knee": 10, "right_ankle": 11,
            "left_hip": 12, "left_knee": 13, "left_ankle": 14,
            "nose": 0, "neck": 1,
        },
    }
    with open(os.path.join(output_dir, "hmr_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run HMR2.0 SMPL body model fitting")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    run_hmr(args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
