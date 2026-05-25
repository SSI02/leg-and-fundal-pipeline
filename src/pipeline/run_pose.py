"""
Worker script: Run pose estimation using MMPose (RTMPose / ViTPose++).
Executed inside the 'pose_env' conda environment.

Usage:
    conda activate pose_env
    python src/pipeline/run_pose.py --image_dir <path> --output_dir <path> [--model human]

Outputs (saved to output_dir):
    - pose_results.json : Per-image keypoint detections with coordinates and scores
    - vis/              : Visualized pose overlay images (optional)

Keypoint indices (COCO 17):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Keypoint indices relevant for leg deformity assessment
LEG_KEYPOINTS = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def run_pose_estimation(image_dir, output_dir, model_name="human", save_vis=True):
    """Run pose estimation on all images in image_dir."""

    from mmpose.apis import MMPoseInferencer

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "vis")
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    # Collect image files
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    image_files = sorted(set(image_files))

    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Found {len(image_files)} images in {image_dir}")
    print(f"Using model: {model_name}")

    # Initialize inferencer
    # We specify the detector config explicitly because the editable mmpose
    # install doesn't set up .mim paths correctly for the bundled RTMDet configs.
    MMPOSE_DIR = os.path.join(PROJECT_DIR, "repos", "mmpose")
    det_config = os.path.join(
        MMPOSE_DIR, "demo", "mmdetection_cfg", "rtmdet_m_640-8xb32_coco-person.py"
    )
    det_weights = (
        "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
        "rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
    )

    inferencer = MMPoseInferencer(
        model_name,
        det_model=det_config,
        det_weights=det_weights,
    )

    all_results = {}

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        print(f"Processing: {img_name}")

        # Run inference
        result_generator = inferencer(img_path, show=False)
        result = next(result_generator)

        # Extract per-person predictions
        persons = []
        for person_idx, person in enumerate(result["predictions"][0]):
            keypoints = person["keypoints"]  # list of [x, y]
            scores = person["keypoint_scores"]  # list of float

            # Build structured keypoint data
            kp_data = {}
            for i, (kp, score) in enumerate(zip(keypoints, scores)):
                kp_data[COCO_KEYPOINT_NAMES[i]] = {
                    "x": float(kp[0]),
                    "y": float(kp[1]),
                    "score": float(score),
                    "index": i,
                }

            # Extract leg-specific keypoints
            leg_kps = {}
            for name, idx in LEG_KEYPOINTS.items():
                leg_kps[name] = kp_data[name]

            # Compute bounding box from keypoints
            kp_array = np.array(keypoints)
            score_array = np.array(scores)
            valid = score_array > 0.3
            if valid.any():
                x_min = float(kp_array[valid, 0].min())
                y_min = float(kp_array[valid, 1].min())
                x_max = float(kp_array[valid, 0].max())
                y_max = float(kp_array[valid, 1].max())
                bbox = [x_min, y_min, x_max, y_max]
            else:
                bbox = None

            persons.append(
                {
                    "person_index": person_idx,
                    "keypoints": kp_data,
                    "leg_keypoints": leg_kps,
                    "bbox": bbox,
                    "mean_score": float(np.mean(scores)),
                    "num_keypoints": len(keypoints),
                }
            )

        all_results[img_name] = {
            "image_path": img_path,
            "num_persons": len(persons),
            "persons": persons,
        }

        # Save visualization
        if save_vis:
            vis_result_generator = inferencer(
                img_path, show=False, vis_out_dir=vis_dir
            )
            _ = next(vis_result_generator)

    # Save results as JSON
    output_path = os.path.join(output_dir, "pose_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved pose results: {output_path}")

    # Print summary
    total_persons = sum(r["num_persons"] for r in all_results.values())
    print(f"Total persons detected: {total_persons} across {len(image_files)} images")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run pose estimation with MMPose")
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
        "--model",
        type=str,
        default="human",
        choices=["human", "vitpose", "vitpose-s", "vitpose-l", "vitpose-h", "wholebody"],
        help="Pose estimation model (default: human = RTMPose-m)",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="Skip saving visualization images",
    )
    args = parser.parse_args()

    run_pose_estimation(
        args.image_dir,
        args.output_dir,
        model_name=args.model,
        save_vis=not args.no_vis,
    )


if __name__ == "__main__":
    main()
