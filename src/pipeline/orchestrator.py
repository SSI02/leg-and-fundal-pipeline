"""
Main orchestrator for the Leg Deformity Detection Pipeline (Phase 1).

Coordinates the execution of different pipeline stages across separate
conda environments:
    - vv_sam3 env  → SAM3 person segmentation (text-prompted, SOTA)
    - amb3r env    → 3D reconstruction (AMB3R backend)
    - vv_vggt env  → 3D reconstruction (VGGT, alternative)
    - pose_env     → 2D pose estimation (RTMPose / ViTPose++)
    - leg_pipeline → calibration, measurements, post-processing, visualization

Pipeline stages:
    1. ArUco calibration (optional)
    2. SAM3 person segmentation → per-image person masks
    3. 2D Pose estimation (RTMPose/ViTPose)
    4. 2D Clinical measurements (leg-only)
    5. 3D Reconstruction (AMB3R or VGGT)
    6. Point cloud post-processing (with SAM3 mask filtering)
    7. 2D Pose on reconstruction-resolution images
    8. 3D Clinical measurements (angles & ratios, leg-only)
    9. Debug visualizations

Usage:
    conda activate leg_pipeline
    python src/pipeline/orchestrator.py \
        --image_dir data/input/patient_001 \
        --output_dir data/output/patient_001 \
        [--marker_size_cm 10.0] \
        [--pose_model human] \
        [--max_images 4] \
        [--skip_3d] \
        [--skip_sam3]

The orchestrator calls worker scripts via subprocess with the appropriate
conda environment activated.
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_DIR = os.path.join(PROJECT_DIR, "src")


def get_conda_prefix():
    """Get the conda installation prefix."""
    # Try CONDA_PREFIX first (set when a conda env is active)
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        # CONDA_PREFIX points to the active env; base is two levels up if in an env
        # or the value itself if base is active
        base_candidate = os.path.dirname(os.path.dirname(conda_prefix))
        if os.path.isdir(os.path.join(base_candidate, "envs")):
            return base_candidate
        # Maybe we're in the base env directly
        if os.path.isdir(os.path.join(conda_prefix, "envs")):
            return conda_prefix

    # Try common conda locations
    for candidate in [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
        "/opt/conda",
    ]:
        if os.path.isdir(os.path.join(candidate, "envs")):
            return candidate

    # Fallback: try calling conda binary with full path
    for conda_bin in [
        os.path.expanduser("~/miniconda3/bin/conda"),
        os.path.expanduser("~/anaconda3/bin/conda"),
        "conda",
    ]:
        try:
            result = subprocess.run(
                [conda_bin, "info", "--base"], capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            continue

    raise RuntimeError("Could not find conda installation. Set CONDA_PREFIX or ensure conda is on PATH.")


def run_in_env(env_name, script_path, args_list, description=""):
    """Run a Python script inside a specific conda environment.

    Args:
        env_name: Name of the conda environment.
        script_path: Path to the Python script.
        args_list: List of command-line arguments.
        description: Human-readable description of the step.

    Returns:
        subprocess.CompletedProcess
    """
    conda_base = get_conda_prefix()
    python_path = os.path.join(conda_base, "envs", env_name, "bin", "python")

    if not os.path.exists(python_path):
        raise FileNotFoundError(
            f"Python not found for env '{env_name}' at {python_path}. "
            f"Please run: bash scripts/setup_{env_name}.sh"
        )

    cmd = [python_path, script_path] + args_list

    print(f"\n{'='*60}")
    print(f"STAGE: {description}")
    print(f"Env: {env_name}")
    print(f"Cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(
            f"Stage '{description}' failed with return code {result.returncode}.\n"
            f"stderr: {result.stderr[-1000:]}"
        )

    print(result.stdout[-500:] if result.stdout else "(no stdout)")
    print(f"Completed in {elapsed:.1f}s")
    return result


def env_available(env_name):
    """Check if a conda environment exists."""
    conda_base = get_conda_prefix()
    python_path = os.path.join(conda_base, "envs", env_name, "bin", "python")
    return os.path.exists(python_path)


def run_pipeline(
    image_dir,
    output_dir,
    marker_size_cm=None,
    scale_calibration=None,
    pose_model="human",
    recon_model="amb3r",
    max_images=4,
    skip_3d=False,
    skip_sam3=False,
    no_outlier_removal=False,
    conf_threshold=0.0,
):
    """Run the full leg deformity detection pipeline.

    Args:
        image_dir: Directory containing input images of standing patient.
        output_dir: Directory to save all outputs.
        marker_size_cm: Real-world ArUco marker side length in cm (None to skip).
        scale_calibration: Path to scale_calibration.json from scale_picker.py.
            If provided, this takes precedence over ArUco for per-image scale.
        pose_model: Pose model name ('human', 'vitpose', 'vitpose-l', etc.).
        recon_model: 3D reconstruction model ('amb3r' or 'vggt').
        max_images: Max images for reconstruction.
        skip_3d: Skip 3D reconstruction (only run 2D pose + measurements).
        skip_sam3: Skip SAM3 person segmentation.
        no_outlier_removal: Skip statistical outlier removal.
        conf_threshold: Confidence threshold for point cloud filtering.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure src/ is on path for local imports
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    results = {"image_dir": image_dir, "output_dir": output_dir, "stages": {}}
    start_total = time.time()

    # ─── Stage 1: Scale Calibration (manual or ArUco) ────────────────
    # Manual scale calibration (from scale_picker.py) takes precedence over
    # ArUco. Per-image scales handle different camera distances correctly.
    scale_factor = None
    per_image_scale = {}

    if scale_calibration and os.path.exists(scale_calibration):
        print("\n" + "=" * 60)
        print("STAGE 1: Manual Scale Calibration (per-image)")
        print("=" * 60)
        from calibration.manual_scale import load_manual_scale, load_manual_scale_full

        per_image_scale = load_manual_scale(scale_calibration)
        full_calib = load_manual_scale_full(scale_calibration)

        if per_image_scale:
            scales = list(per_image_scale.values())
            scale_factor = float(np.mean(scales)) if scales else None
            print(f"Loaded {len(per_image_scale)} per-image scales:")
            for img, s in per_image_scale.items():
                obj = full_calib.get(img, {}).get("object_description", "?")
                d_cm = full_calib.get(img, {}).get("real_distance_cm", "?")
                print(f"  {img}: {s:.5f} cm/px ({d_cm}cm = {obj})")
            print(f"Mean: {scale_factor:.5f} cm/px (fallback for uncalibrated images)")

            cal_path = os.path.join(output_dir, "calibration.json")
            with open(cal_path, "w") as f:
                json.dump({
                    "source": "manual_scale_picker",
                    "scale_calibration_path": scale_calibration,
                    "scale_cm_per_pixel": scale_factor,
                    "per_image_scale": per_image_scale,
                    "details": full_calib,
                }, f, indent=2)
            results["stages"]["calibration"] = {
                "source": "manual",
                "scale_cm_per_pixel": scale_factor,
                "per_image_scale": per_image_scale,
                "output": cal_path,
            }
        else:
            print("WARNING: scale_calibration.json is empty or invalid.")
            results["stages"]["calibration"] = {"skipped": True}

    elif marker_size_cm is not None:
        print("\n" + "=" * 60)
        print("STAGE 1: ArUco Calibration")
        print("=" * 60)

        from calibration.aruco import compute_scale_factor

        cal_result = compute_scale_factor(image_dir, marker_size_cm)
        scale_factor = cal_result.get("scale_cm_per_pixel")
        per_image_scale = cal_result.get("per_image_scale", {}) or {}

        cal_path = os.path.join(output_dir, "calibration.json")
        with open(cal_path, "w") as f:
            json.dump(cal_result, f, indent=2)

        if scale_factor:
            print(f"Global avg scale: {scale_factor:.6f} cm/pixel (fallback)")
            print(f"Detections: {cal_result['num_detections']}")
            print(f"Per-image scales:")
            for img, s in per_image_scale.items():
                print(f"  {img}: {s:.6f} cm/pixel")
        else:
            print("WARNING: No ArUco markers detected. Proceeding without metric scale.")

        results["stages"]["calibration"] = {
            "source": "aruco",
            "scale_cm_per_pixel": scale_factor,
            "per_image_scale": per_image_scale,
            "output": cal_path,
        }
    else:
        print("\nSkipping calibration (no --scale_calibration or --marker_size_cm).")
        print("  2D measurements will be reported in pixels.")
        results["stages"]["calibration"] = {"skipped": True}

    # ─── Stage 1b: SAM3 Person Segmentation ──────────────────────────
    segmentation_dir = os.path.join(output_dir, "segmentation")
    sam3_script = os.path.join(SRC_DIR, "pipeline", "run_sam3.py")

    if not skip_sam3 and env_available("vv_sam3"):
        try:
            run_in_env(
                "vv_sam3",
                sam3_script,
                [
                    "--image_dir", image_dir,
                    "--output_dir", segmentation_dir,
                    "--prompt", "person",
                    "--confidence", "0.3",
                ],
                description="SAM3 Person Segmentation",
            )
            results["stages"]["segmentation"] = {
                "output_dir": segmentation_dir,
                "model": "SAM3",
            }
        except Exception as e:
            print(f"WARNING: SAM3 segmentation failed: {e}")
            print("Continuing without person segmentation (point cloud will include background)")
            results["stages"]["segmentation"] = {"skipped": True, "reason": str(e)}
            segmentation_dir = None
    else:
        if skip_sam3:
            print("\nSkipping SAM3 segmentation (--skip_sam3 flag set)")
        else:
            print("\nSkipping SAM3 segmentation (vv_sam3 env not found)")
            print("  To enable: set up the vv_sam3 conda environment")
        results["stages"]["segmentation"] = {"skipped": True}
        segmentation_dir = None

    # ─── Stage 2: 2D Pose Estimation ─────────────────────────────────
    pose_output_dir = os.path.join(output_dir, "pose")
    pose_script = os.path.join(SRC_DIR, "pipeline", "run_pose.py")

    run_in_env(
        "pose_env",
        pose_script,
        [
            "--image_dir", image_dir,
            "--output_dir", pose_output_dir,
            "--model", pose_model,
        ],
        description="2D Pose Estimation",
    )

    pose_results_path = os.path.join(pose_output_dir, "pose_results.json")
    results["stages"]["pose"] = {"output": pose_results_path, "model": pose_model}

    # ─── Stage 3: Clinical Measurements (from 2D pose) ───────────────
    print("\n" + "=" * 60)
    print("STAGE 3: Clinical Measurements (2D)")
    print("=" * 60)

    from measurements.clinical import process_pose_results

    measurements_path = os.path.join(output_dir, "clinical_measurements.json")
    process_pose_results(pose_results_path, measurements_path,
                         scale_factor=scale_factor,
                         per_image_scale=per_image_scale)

    results["stages"]["measurements_2d"] = {
        "output": measurements_path,
        "units": "cm" if scale_factor else "pixels",
    }

    # Print summary
    with open(measurements_path, "r") as f:
        measurements = json.load(f)
    _print_measurement_summary(measurements)

    # ─── Stage 4: 3D Reconstruction (AMB3R or VGGT) ─────────────────
    if not skip_3d:
        recon_output_dir = os.path.join(output_dir, "reconstruction")

        if recon_model == "vggt":
            recon_script = os.path.join(SRC_DIR, "pipeline", "run_vggt.py")
            recon_env = "vv_vggt"
            recon_desc = "3D Reconstruction (VGGT)"
            is_metric = False
        else:
            recon_script = os.path.join(SRC_DIR, "pipeline", "run_amb3r.py")
            recon_env = "amb3r"
            recon_desc = "3D Reconstruction (AMB3R)"
            is_metric = True

        run_in_env(
            recon_env,
            recon_script,
            [
                "--image_dir", image_dir,
                "--output_dir", recon_output_dir,
                "--max_images", str(max_images),
                "--conf_threshold", str(conf_threshold),
            ],
            description=recon_desc,
        )

        results["stages"]["reconstruction"] = {
            "output_dir": recon_output_dir,
            "point_cloud": os.path.join(recon_output_dir, "point_cloud.ply"),
            "model": recon_model,
            "is_metric": is_metric,
        }

        if not is_metric:
            print("NOTE: VGGT outputs are in arbitrary scale. 3D measurements will be in arbitrary units.")
            print("      Use --scale_calibration with scale_picker.py for metric calibration.")

        # ─── Stage 4b: Apply manual 3D scale (if calibration provided) ───
        # If user provided scale_calibration.json, compute the true 3D scale
        # by looking up the clicked points in the per-frame point map.
        # This is the most reliable way to get true metric units.
        if scale_calibration and os.path.exists(scale_calibration):
            print("\n" + "=" * 60)
            print("STAGE 4b: Apply Manual 3D Scale Calibration")
            print("=" * 60)
            from calibration.manual_scale import (
                compute_3d_scale_factor, apply_3d_scale_to_npz,
            )

            # Determine image order used by the reconstruction:
            # AMB3R/VGGT processes images sorted alphabetically.
            import glob as _glob
            recon_images = sorted({
                os.path.basename(p)
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
                for p in _glob.glob(os.path.join(image_dir, ext)) +
                          _glob.glob(os.path.join(image_dir, ext.upper()))
            })
            recon_images = recon_images[:max_images]

            npz_path = os.path.join(recon_output_dir, "point_cloud.npz")
            scale_3d = compute_3d_scale_factor(
                npz_path, scale_calibration, recon_images,
            )

            if scale_3d is not None:
                factor = scale_3d["scale_factor"]
                print(f"3D scale factor: {factor:.4f} (median {scale_3d['scale_factor_median']:.4f}, "
                      f"std {scale_3d['scale_factor_std']:.4f}, n={scale_3d['n_frames_used']} frames)")
                for d in scale_3d.get("details", []):
                    if "error" in d:
                        print(f"  frame {d['frame']} ({d['image']}): {d['error']}")
                    else:
                        print(f"  frame {d['frame']} ({d['image']}): "
                              f"3D distance {d['amb3r_3d_distance_m']*100:.2f}cm vs "
                              f"real {d['real_distance_cm']:.2f}cm → factor {d['scale_factor_to_apply']:.4f}")

                # Apply to NPZ in-place so downstream stages see calibrated points
                apply_3d_scale_to_npz(npz_path, factor)
                # Also re-export PLY at correct scale
                try:
                    import open3d as o3d
                    pcd = o3d.io.read_point_cloud(
                        os.path.join(recon_output_dir, "point_cloud.ply"))
                    pts = np.asarray(pcd.points) * factor
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    o3d.io.write_point_cloud(
                        os.path.join(recon_output_dir, "point_cloud.ply"), pcd)
                except Exception as e:
                    print(f"  (PLY rescale skipped: {e})")

                results["stages"]["scale_3d"] = {
                    "factor": factor,
                    "details": scale_3d.get("details"),
                    "n_frames_used": scale_3d["n_frames_used"],
                }
            else:
                print("WARNING: Could not compute 3D scale from clicked points.")
                results["stages"]["scale_3d"] = {"skipped": True}

        # ─── Stage 5: Point Cloud Post-Processing (with SAM3 mask filtering) ─
        print("\n" + "=" * 60)
        print("STAGE 5: Point Cloud Post-Processing")
        if segmentation_dir:
            print("  (with SAM3 person segmentation mask filtering)")
        print("=" * 60)

        from measurements.postprocess import full_postprocess

        postprocess_dir = os.path.join(output_dir, "postprocessed")
        amb3r_npz = os.path.join(recon_output_dir, "point_cloud.npz")
        amb3r_imgs_dir = os.path.join(recon_output_dir, "amb3r_images")

        pp_outputs = full_postprocess(
            os.path.join(recon_output_dir, "point_cloud.ply"),
            postprocess_dir,
            segmentation_dir=segmentation_dir,
            amb3r_npz_path=amb3r_npz if segmentation_dir else None,
            amb3r_images_dir=amb3r_imgs_dir if segmentation_dir else None,
            skip_outlier_removal=no_outlier_removal,
        )

        results["stages"]["postprocess"] = pp_outputs

        # ─── Stage 5b: Re-run pose on AMB3R-resolution images ────────
        # AMB3R crops/resizes images internally, so pose keypoints from
        # original images DON'T map to the point map. We re-detect pose
        # on AMB3R's stored images so coordinates match 1:1.
        amb3r_imgs_dir = os.path.join(recon_output_dir, "amb3r_images")
        amb3r_pose_dir = os.path.join(output_dir, "pose_amb3r")

        if os.path.isdir(amb3r_imgs_dir):
            run_in_env(
                "pose_env",
                pose_script,
                [
                    "--image_dir", amb3r_imgs_dir,
                    "--output_dir", amb3r_pose_dir,
                    "--model", pose_model,
                ],
                description="2D Pose on AMB3R-resolution images (for 3D projection)",
            )
            amb3r_pose_path = os.path.join(amb3r_pose_dir, "pose_results.json")
        else:
            print("WARNING: AMB3R images not saved, falling back to original pose")
            amb3r_pose_path = pose_results_path

        # ─── Stage 6: 3D Clinical Measurements (AMB3R metric) ────────
        print("\n" + "=" * 60)
        print("STAGE 6: 3D Clinical Measurements (metric, from AMB3R point maps)")
        print("=" * 60)

        from measurements.clinical_3d import measure_from_pointmap

        amb3r_npz = os.path.join(recon_output_dir, "point_cloud.npz")
        measurements_3d_path = os.path.join(output_dir, "clinical_measurements_3d.json")

        if os.path.exists(amb3r_npz) and os.path.exists(amb3r_pose_path):
            measure_from_pointmap(
                amb3r_pose_path, amb3r_npz, measurements_3d_path
            )
            results["stages"]["measurements_3d"] = {
                "output": measurements_3d_path,
                "units": "meters",
                "method": "direct_lookup_from_amb3r_pointmap",
                "pose_source": "amb3r_resolution_images",
            }

            with open(measurements_3d_path, "r") as f:
                meas_3d = json.load(f)
            _print_3d_measurement_summary(meas_3d)
        else:
            print("WARNING: AMB3R NPZ or pose results missing, skipping 3D measurements")
            results["stages"]["measurements_3d"] = {"skipped": True}

    else:
        print("\nSkipping 3D reconstruction (--skip_3d flag set)")
        results["stages"]["reconstruction"] = {"skipped": True}
        results["stages"]["postprocess"] = {"skipped": True}
        results["stages"]["measurements_3d"] = {"skipped": True}

    # ─── Debug Visualizations ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING DEBUG VISUALIZATIONS")
    print("=" * 60)

    from visualization.debug_viz import run_all_debug
    debug_dir = run_all_debug(output_dir, image_dir=image_dir)
    results["stages"]["debug"] = {"output_dir": debug_dir}

    # ─── Save final results ──────────────────────────────────────────
    elapsed_total = time.time() - start_total
    results["total_time_seconds"] = round(elapsed_total, 1)

    results_path = os.path.join(output_dir, "pipeline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE ({elapsed_total:.1f}s)")
    print(f"Results: {results_path}")
    print(f"Debug:   {debug_dir}")
    print(f"{'='*60}")

    return results


def _print_measurement_summary(measurements):
    """Print a human-readable summary of clinical measurements."""
    for img_name, persons in measurements.items():
        print(f"\n--- {img_name} ---")
        for person_data in persons:
            if "error" in person_data:
                print(f"  Person {person_data['person_index']}: {person_data['error']}")
                continue

            a = person_data["assessment"]
            units = a.get("units", "pixels")

            print(f"  Person {person_data['person_index']}:")

            for side in ["left_leg", "right_leg"]:
                leg = a.get(side)
                if leg:
                    hka = leg.get("hka_angle", 0)
                    dev = leg.get("hka_deviation", 0)
                    cls = leg.get("classification", "?")
                    sev = leg.get("severity", "?")
                    length = leg.get("total_leg_length", 0)
                    print(
                        f"    {side}: HKA={hka:.1f}° (dev={dev:+.1f}°) "
                        f"| {cls} ({sev}) | length={length:.1f}{units}"
                    )

            knee_gap = a.get("intercondylar_distance", 0)
            ankle_gap = a.get("intermalleolar_distance", 0)
            print(f"    Knee gap: {knee_gap:.1f}{units} | Ankle gap: {ankle_gap:.1f}{units}")

            if a.get("flags"):
                print(f"    FLAGS: {'; '.join(a['flags'])}")

            overall = a.get("overall_classification", "Unknown")
            print(f"    Overall: {overall}")


def _print_3d_measurement_summary(measurements):
    """Print 3D measurement summary (angles and ratios only — scale-invariant)."""
    print("\n--- 3D MEASUREMENTS (angles & ratios — scale-invariant) ---")
    print("    NOTE: Absolute distances are in AMB3R raw units (NOT metric)")
    for img_name, persons in measurements.items():
        print(f"\n  {img_name}:")
        for person_data in persons:
            if "error" in person_data:
                print(f"    Person {person_data['person_index']}: {person_data['error']}")
                continue

            a = person_data["assessment"]
            print(f"    Person {person_data['person_index']} [{a.get('method', '?')}]:")

            for side in ["left_leg", "right_leg"]:
                leg = a.get(side)
                if leg:
                    hka = leg.get("hka_angle_3d", 0)
                    dev = leg.get("hka_deviation_3d", 0)
                    cls = leg.get("classification", "?")
                    sev = leg.get("severity", "?")
                    print(
                        f"      {side}: HKA={hka:.1f}° (dev={dev:+.1f}°) "
                        f"| {cls} ({sev})"
                    )

            # Ratios
            ratios = a.get("ratios", {})
            if ratios:
                l_ft = ratios.get("left_femur_tibia_ratio", 0)
                r_ft = ratios.get("right_femur_tibia_ratio", 0)
                sym = ratios.get("leg_symmetry_ratio", 0)
                lld_pct = ratios.get("leg_length_difference_pct", 0)
                print(f"      Femur/tibia ratio: L={l_ft:.2f} R={r_ft:.2f} (normal ~1.1)")
                print(f"      Leg symmetry (L/R): {sym:.3f} (1.0 = equal)")
                if lld_pct > 2.0:
                    print(f"      Limb length asymmetry: {lld_pct:.1f}%")

            if a.get("flags"):
                for flag in a["flags"]:
                    print(f"      FLAG: {flag}")

            overall = a.get("overall_classification", "Unknown")
            print(f"      Overall: {overall}")


def main():
    parser = argparse.ArgumentParser(
        description="Leg Deformity Detection Pipeline (Phase 1)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images of standing patient",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save all pipeline outputs",
    )
    parser.add_argument(
        "--marker_size_cm",
        type=float,
        default=None,
        help="ArUco marker real-world side length in cm (omit to skip calibration)",
    )
    parser.add_argument(
        "--scale_calibration",
        type=str,
        default=None,
        help="Path to scale_calibration.json from scale_picker.py "
             "(takes precedence over ArUco; per-image scale)",
    )
    parser.add_argument(
        "--pose_model",
        type=str,
        default="human",
        choices=["human", "vitpose", "vitpose-s", "vitpose-l", "vitpose-h", "wholebody"],
        help="Pose estimation model (default: human = RTMPose-m)",
    )
    parser.add_argument(
        "--recon_model",
        type=str,
        default="amb3r",
        choices=["amb3r", "vggt"],
        help="3D reconstruction model: amb3r (metric) or vggt (arbitrary scale) (default: amb3r)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=4,
        help="Max images for 3D reconstruction (default: 4)",
    )
    parser.add_argument(
        "--skip_3d",
        action="store_true",
        help="Skip 3D reconstruction, only run 2D pose + measurements",
    )
    parser.add_argument(
        "--skip_sam3",
        action="store_true",
        help="Skip SAM3 person segmentation",
    )
    parser.add_argument(
        "--no_outlier_removal",
        action="store_true",
        help="Skip statistical outlier removal in point cloud post-processing",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for point cloud filtering",
    )
    args = parser.parse_args()

    run_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        marker_size_cm=args.marker_size_cm,
        scale_calibration=args.scale_calibration,
        pose_model=args.pose_model,
        recon_model=args.recon_model,
        max_images=args.max_images,
        skip_3d=args.skip_3d,
        skip_sam3=args.skip_sam3,
        no_outlier_removal=args.no_outlier_removal,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
