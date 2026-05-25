"""
Worker script: Run HMR2.0 (4D-Humans) for SMPL body model estimation.
Executed inside the 'hmr_env' conda environment (or 4D-humans).

Estimates SMPL body parameters (pose, shape, translation) and 3D joints
directly from 2D images. This is SOTA for single-image human mesh recovery,
providing anatomically-constrained 3D joint positions that are more robust
than lifting 2D keypoints to 3D via point map lookup.

Usage:
    conda activate hmr_env
    python src/pipeline/run_hmr2.py \
        --image_dir data/input/patient_001 \
        --output_dir data/output/patient_001/hmr2

Outputs (saved to output_dir):
    - hmr2_results.npz : Raw arrays (joints_3d, vertices, betas, pred_cam, boxes)
    - hmr2_results.json: Per-image structured results with 3D joint positions
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
HMR2_DIR = os.path.join(PROJECT_DIR, "repos", "4D-Humans")
sys.path.insert(0, HMR2_DIR)


# SMPL joint names in OpenPose 25 order (first 25 of HMR2's 44 joints)
OPENPOSE_JOINT_NAMES = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "mid_hip",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]

# Indices for leg joints in OpenPose order
LEG_JOINT_INDICES = {
    "left_hip": 12,
    "left_knee": 13,
    "left_ankle": 14,
    "right_hip": 9,
    "right_knee": 10,
    "right_ankle": 11,
}


def run_hmr2(
    image_dir,
    output_dir,
    checkpoint=None,
    detector_type="vitdet",
    batch_size=8,
):
    """Run HMR2.0 on all images in image_dir.

    Args:
        image_dir: Directory containing input images.
        output_dir: Directory to save outputs.
        checkpoint: Path to HMR2 checkpoint (None for default).
        detector_type: 'vitdet' (accurate) or 'regnety' (fast).
        batch_size: Batch size for HMR2 inference.
    """
    import torch
    import cv2
    from pathlib import Path

    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils.renderer import cam_crop_to_full

    os.makedirs(output_dir, exist_ok=True)

    # Download and load model
    download_models(CACHE_DIR_4DHUMANS)
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    model, model_cfg = load_hmr2(checkpoint)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy

    if detector_type == "vitdet":
        from detectron2.config import LazyConfig
        import hmr2 as hmr2_module

        cfg_path = Path(hmr2_module.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg

        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

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
    print(f"Detector: {detector_type}")

    all_results = {}
    all_joints_3d = []
    all_vertices = []
    all_betas = []
    all_pred_cam = []

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        print(f"\nProcessing: {img_name}")

        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            print(f"  Failed to read image, skipping")
            continue

        img_height, img_width = img_cv2.shape[:2]

        # Detect humans
        det_out = detector(img_cv2)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        if len(boxes) == 0:
            print(f"  No persons detected")
            all_results[img_name] = {
                "image_path": img_path,
                "width": img_width,
                "height": img_height,
                "num_persons": 0,
                "persons": [],
            }
            continue

        print(f"  Detected {len(boxes)} persons")

        # Run HMR2.0
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        img_persons = []
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            pred_cam = out["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH
                / model_cfg.MODEL.IMAGE_SIZE
                * img_size.max()
            )
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
                .detach().cpu().numpy()
            )

            # Extract results per person
            joints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # (B, 44, 3)
            vertices = out["pred_vertices"].detach().cpu().numpy()  # (B, 6890, 3)
            betas = out["pred_smpl_params"]["betas"].detach().cpu().numpy()  # (B, 10)
            body_pose = out["pred_smpl_params"]["body_pose"].detach().cpu().numpy()  # (B, 23, 3, 3)
            global_orient = out["pred_smpl_params"]["global_orient"].detach().cpu().numpy()  # (B, 1, 3, 3)

            for i in range(len(joints_3d)):
                person_idx = len(img_persons)
                j3d = joints_3d[i]  # (44, 3)
                cam_t = pred_cam_t_full[i]  # (3,)

                # Extract leg joints
                leg_joints = {}
                for name, idx in LEG_JOINT_INDICES.items():
                    leg_joints[name] = {
                        "x": float(j3d[idx, 0]),
                        "y": float(j3d[idx, 1]),
                        "z": float(j3d[idx, 2]),
                        "joint_index": idx,
                    }

                # Compute basic leg metrics from SMPL joints
                l_hip = j3d[LEG_JOINT_INDICES["left_hip"]]
                l_knee = j3d[LEG_JOINT_INDICES["left_knee"]]
                l_ankle = j3d[LEG_JOINT_INDICES["left_ankle"]]
                r_hip = j3d[LEG_JOINT_INDICES["right_hip"]]
                r_knee = j3d[LEG_JOINT_INDICES["right_knee"]]
                r_ankle = j3d[LEG_JOINT_INDICES["right_ankle"]]

                l_femur = float(np.linalg.norm(l_knee - l_hip))
                l_tibia = float(np.linalg.norm(l_ankle - l_knee))
                r_femur = float(np.linalg.norm(r_knee - r_hip))
                r_tibia = float(np.linalg.norm(r_ankle - r_knee))

                person_data = {
                    "person_index": person_idx,
                    "detection_box": boxes[person_idx].tolist() if person_idx < len(boxes) else None,
                    "camera_translation": cam_t.tolist(),
                    "leg_joints_3d": leg_joints,
                    "body_shape_betas": betas[i].tolist(),
                    "leg_metrics": {
                        "left_femur_length": l_femur,
                        "left_tibia_length": l_tibia,
                        "left_total": l_femur + l_tibia,
                        "right_femur_length": r_femur,
                        "right_tibia_length": r_tibia,
                        "right_total": r_femur + r_tibia,
                    },
                }
                img_persons.append(person_data)

                all_joints_3d.append(j3d)
                all_vertices.append(vertices[i])
                all_betas.append(betas[i])
                all_pred_cam.append(cam_t)

        all_results[img_name] = {
            "image_path": img_path,
            "width": img_width,
            "height": img_height,
            "num_persons": len(img_persons),
            "persons": img_persons,
        }
        print(f"  HMR2 processed {len(img_persons)} persons")

    # Save structured JSON results
    json_path = os.path.join(output_dir, "hmr2_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON results: {json_path}")

    # Save raw numpy arrays for downstream 3D measurement code
    if all_joints_3d:
        npz_path = os.path.join(output_dir, "hmr2_results.npz")
        np.savez_compressed(
            npz_path,
            joints_3d=np.array(all_joints_3d),
            vertices=np.array(all_vertices),
            betas=np.array(all_betas),
            pred_cam=np.array(all_pred_cam),
        )
        print(f"Saved NPZ results: {npz_path}")

    total = sum(r["num_persons"] for r in all_results.values())
    print(f"Total persons processed: {total} across {len(image_files)} images")
    return json_path


def main():
    parser = argparse.ArgumentParser(description="Run HMR2.0 SMPL estimation")
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save outputs"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to HMR2 checkpoint (default: auto-download)",
    )
    parser.add_argument(
        "--detector", type=str, default="vitdet", choices=["vitdet", "regnety"],
        help="Person detector (default: vitdet)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size (default: 8)",
    )
    args = parser.parse_args()

    run_hmr2(
        args.image_dir,
        args.output_dir,
        checkpoint=args.checkpoint,
        detector_type=args.detector,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
