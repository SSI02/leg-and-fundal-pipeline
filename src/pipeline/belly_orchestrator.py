"""
Main orchestrator for the Belly Analysis Pipeline.

Estimates belly volume and belly-button location for pregnant patients.

Pipeline stages:
    1. Frame extraction (if input is video)
    2. Manual scale calibration (from scale_picker.py — required for true volume)
    3. SAM3 belly segmentation (text-prompted)
    4. 3D reconstruction (AMB3R or VGGT)
    5. Apply 3D scale calibration (so points are in true meters)
    6. 2D pose estimation on AMB3R-resolution images (for feet location)
    7. Belly point cloud → mesh → volume → belly button → feet distance

Conda envs used:
    - leg_pipeline: orchestration, calibration, belly mesh/volume
    - vv_sam3: belly segmentation
    - amb3r or vv_vggt: 3D reconstruction
    - pose_env: 2D pose

Usage:
    conda activate leg_pipeline
    python src/pipeline/belly_orchestrator.py \
        --image_dir data/input/belly_001 \
        --output_dir data/output/belly_001 \
        --scale_calibration data/input/belly_001/scale_calibration.json
"""

import os
import sys
import json
import argparse
import subprocess
import time
import glob
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def get_conda_prefix():
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        base_candidate = os.path.dirname(os.path.dirname(conda_prefix))
        if os.path.isdir(os.path.join(base_candidate, "envs")):
            return base_candidate
        if os.path.isdir(os.path.join(conda_prefix, "envs")):
            return conda_prefix
    for candidate in [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
        "/opt/conda",
    ]:
        if os.path.isdir(os.path.join(candidate, "envs")):
            return candidate
    raise RuntimeError("Could not find conda installation.")


def env_python(env_name):
    base = get_conda_prefix()
    p = os.path.join(base, "envs", env_name, "bin", "python")
    return p if os.path.exists(p) else None


def run_in_env(env_name, script_path, args_list, description=""):
    py = env_python(env_name)
    if py is None:
        raise FileNotFoundError(f"Env '{env_name}' not found. Set it up first.")
    cmd = [py, script_path] + args_list
    print(f"\n{'='*60}\nSTAGE: {description}\nEnv: {env_name}\nCmd: {' '.join(cmd)}\n{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(
            f"Stage '{description}' failed (rc={result.returncode}).\n"
            f"stderr (last 1k): {result.stderr[-1000:]}"
        )
    if result.stdout:
        print(result.stdout[-1500:])
    print(f"Completed in {elapsed:.1f}s")


# Per-subject defaults: SAM3 prompt cascade, pose-stage decision, and whether
# to use seed-point geometric prompts.
SUBJECT_PRESETS = {
    "pregnant": {
        "sam_prompt": "person",
        "sam_fallback_prompts": ("belly", "stomach", "abdomen"),
        "run_pose": True,           # need ankle keypoints for distance-to-ground
        "use_seed_points": True,    # narrow SAM3 from 'person' (whole body) to the belly
    },
    "balloon": {
        "sam_prompt": "balloon",
        "sam_fallback_prompts": ("ball", "round object", "sphere"),
        "run_pose": False,          # no person → no feet → no distance computation
        "use_seed_points": False,   # 'balloon' text prompt is already specific;
                                    # adding a box prompt can pin SAM3 to a small
                                    # region around the click and miss parts of
                                    # the balloon → worse mask quality.
    },
    "balloon_held": {
        # Person holds a balloon at belly level. Segment the BALLOON for the
        # 3D mesh + volume + apex localization, but ALSO run pose detection
        # on the original images to find the person's ankles/feet so we can
        # report the (balloon apex → ground/feet) distance.
        "sam_prompt": "balloon",
        "sam_fallback_prompts": ("ball", "round object", "sphere"),
        "run_pose": True,
        "use_seed_points": False,
    },
}


def run_belly_pipeline(
    image_dir=None,
    video=None,
    n_frames=30,  # frames to extract from video for SAM3 (must match picker so seed clicks align)
    recon_max_frames=20,  # cap for VGGT/AMB3R reconstruction (uniform-sampled subset)
    output_dir=None,
    scale_calibration=None,
    recon_model="vggt",  # default for belly: preserves all pixels via pad mode
    subject="pregnant",  # 'pregnant' or 'balloon' — sets prompt + pose defaults
    sam_prompt=None,  # if None, derived from subject
    sam_fallback_prompts=None,  # if None, derived from subject
    sam_confidence=0.25,
    seed_points=None,
    use_seed_points=None,  # True/False/None (None = use subject preset default)
    low_memory=False,  # default OFF: batched VGGT for proper multi-view fusion
                        # (per-frame mode loses fusion → overlapping clouds)
    skip_3d=False,
    no_outlier_removal=False,
    poisson_depth=8,
    conf_pct_keep=75,
):
    """Run the full belly analysis pipeline.

    Args:
        image_dir: Directory of input images (mutually exclusive with `video`).
        video: Path to video file. Frames will be extracted to <video>_frames/.
        n_frames: Number of frames to extract from video.
        output_dir: Output directory.
        scale_calibration: Path to scale_calibration.json (required for metric).
        recon_model: 'amb3r' or 'vggt'.
        sam_prompt: SAM3 text prompt for belly.
        sam_confidence: SAM3 detection confidence threshold.
    """
    assert image_dir or video, "Must specify image_dir or video"
    os.makedirs(output_dir, exist_ok=True)

    # Resolve subject preset → default prompts + pose behavior
    if subject not in SUBJECT_PRESETS:
        raise ValueError(f"Unknown subject '{subject}'. "
                         f"Choices: {list(SUBJECT_PRESETS)}")
    preset = SUBJECT_PRESETS[subject]
    if sam_prompt is None:
        sam_prompt = preset["sam_prompt"]
    if sam_fallback_prompts is None:
        sam_fallback_prompts = preset["sam_fallback_prompts"]
    run_pose_stage = preset["run_pose"]
    if use_seed_points is None:
        use_seed_points = preset.get("use_seed_points", True)
    print(f"\n[subject={subject}] SAM prompt='{sam_prompt}', "
          f"fallbacks={list(sam_fallback_prompts)}, "
          f"pose stage {'ON (computes belly→ground distance)' if run_pose_stage else 'SKIPPED (no person)'}, "
          f"seed-prompt {'ON' if use_seed_points else 'OFF (text prompt alone)'}")

    results = {"output_dir": output_dir, "stages": {}, "subject": subject}
    start_total = time.time()

    # ─── Stage 0: Frame extraction (if video) ────────────────────────
    if video:
        print("\n" + "="*60 + "\nSTAGE 0: Frame extraction\n" + "="*60)
        from calibration.extract_frames import extract_frames
        frames_dir = os.path.splitext(video)[0] + "_frames"

        # CRITICAL: if the picker has already extracted frames here, reuse
        # them so any clicks in scale_calibration.json / belly_seed.json stay
        # aligned with the actual files. Re-extracting with a different
        # n_frames would overwrite frame_NNN.jpg with different video content,
        # silently invalidating every saved click.
        manifest_path = os.path.join(frames_dir, "_video_frame_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                existing = json.load(f)
            existing_n = len(existing.get("saved", []))
            if existing_n == n_frames:
                print(f"Reusing existing {existing_n} frames in {frames_dir} "
                      f"(manifest matches n_frames={n_frames})")
            else:
                print(f"\nERROR: {frames_dir} has {existing_n} frames from "
                      f"a previous extraction, but you asked for n_frames={n_frames}.")
                print(f"  Re-extracting would silently invalidate any clicks made "
                      f"in scale_calibration.json or belly_seed.json against the "
                      f"existing {existing_n}-frame layout.")
                print(f"  Either:")
                print(f"    (a) re-run the picker with --n_frames {n_frames} to refresh clicks, OR")
                print(f"    (b) re-run this pipeline with --n_frames {existing_n} to match existing frames.")
                sys.exit(1)
        else:
            extract_frames(video, frames_dir, n_frames=n_frames)
        image_dir = frames_dir
        results["stages"]["frame_extraction"] = {
            "frames_dir": frames_dir, "n_frames": n_frames,
            "reused_existing": os.path.exists(manifest_path),
        }

    # ─── Stage 1: SAM3 Belly Segmentation ────────────────────────────
    seg_dir = os.path.join(output_dir, "segmentation")
    sam3_script = os.path.join(SRC_DIR, "pipeline", "run_sam3.py")

    if env_python("vv_sam3"):
        sam3_args = [
            "--image_dir", image_dir,
            "--output_dir", seg_dir,
            "--prompt", sam_prompt,
            "--confidence", str(sam_confidence),
        ]
        if sam_fallback_prompts:
            sam3_args += ["--fallback_prompts", ",".join(sam_fallback_prompts)]
        # Only pass seed points to SAM3 if the subject preset enables them.
        # For 'balloon', the text prompt is already specific enough; adding a
        # geometric box prompt around a single click can constrain SAM3 to a
        # small region and miss parts of the object.
        if use_seed_points and seed_points and os.path.exists(seed_points):
            sam3_args += ["--seed_points", seed_points]
            print(f"Using seed points from {seed_points} (subject={subject})")
        elif seed_points and os.path.exists(seed_points) and not use_seed_points:
            print(f"NOTE: SEED_POINTS='{seed_points}' provided but ignored "
                  f"(subject={subject} → text-prompt-only segmentation gives "
                  f"better masks for this object class).")
        run_in_env(
            "vv_sam3", sam3_script, sam3_args,
            description=f"SAM3 Segmentation (prompt: '{sam_prompt}', "
                        f"fallbacks: {list(sam_fallback_prompts)})",
        )
        results["stages"]["segmentation"] = {
            "output_dir": seg_dir,
            "prompt": sam_prompt,
            "fallback_prompts": list(sam_fallback_prompts),
        }
    else:
        print("\nERROR: vv_sam3 env not found. Belly segmentation is required for this pipeline.")
        sys.exit(1)

    # ─── Stage 1b: Select frames for reconstruction ──────────────────
    # Intersection of "frames where SAM3 found something" and "input frames",
    # uniformly down-sampled to at most `recon_max_frames`. This avoids
    # wasting GPU on frames with no segmentation, and caps VGGT memory.
    seg_json_path = os.path.join(seg_dir, "segmentation.json")
    selected_basenames = []
    if os.path.exists(seg_json_path):
        with open(seg_json_path) as f:
            seg_data = json.load(f)
        # Frames with valid segmentation (status == "ok" OR num_detections > 0)
        valid_frames = sorted(
            n for n, r in seg_data.items()
            if (r.get("status") == "ok") or (r.get("num_detections", 0) > 0)
        )
        n_valid = len(valid_frames)
        if n_valid == 0:
            print(f"\nERROR: SAM3 found nothing in any of {len(seg_data)} frames.")
            sys.exit(1)
        # Uniform sub-sample if more than recon_max_frames
        if n_valid > recon_max_frames:
            idxs = np.linspace(0, n_valid - 1, recon_max_frames, dtype=int)
            selected_basenames = [valid_frames[i] for i in idxs]
            print(f"\n[frame selection] {n_valid} frames had valid segmentation; "
                  f"uniformly sampled {len(selected_basenames)} of them for "
                  f"reconstruction (cap={recon_max_frames}):")
        else:
            selected_basenames = valid_frames
            print(f"\n[frame selection] All {n_valid} valid-segmentation frames "
                  f"used for reconstruction (≤ cap={recon_max_frames}):")
        for n in selected_basenames:
            print(f"   • {n}")

    # Symlink (or copy) the selected frames to recon_frames_dir so VGGT/AMB3R
    # only sees those. This keeps the rest of the pipeline simple and ensures
    # recon_meta.image_files_in_order matches our intended subset.
    recon_frames_dir = os.path.join(output_dir, "recon_frames")
    os.makedirs(recon_frames_dir, exist_ok=True)
    # Clean any stale entries
    for old in os.listdir(recon_frames_dir):
        old_p = os.path.join(recon_frames_dir, old)
        if os.path.islink(old_p) or os.path.isfile(old_p):
            os.unlink(old_p)
    if selected_basenames:
        for bn in selected_basenames:
            src = os.path.abspath(os.path.join(image_dir, bn))
            dst = os.path.join(recon_frames_dir, bn)
            if os.path.exists(src):
                try:
                    os.symlink(src, dst)
                except OSError:
                    import shutil
                    shutil.copy(src, dst)
        recon_input_dir = recon_frames_dir
        recon_n = len(selected_basenames)
    else:
        # Fall back to original directory if SAM3 results unavailable
        recon_input_dir = image_dir
        recon_n = recon_max_frames
    results["stages"]["frame_selection"] = {
        "selected_basenames": selected_basenames,
        "n_selected": len(selected_basenames),
        "recon_input_dir": recon_input_dir,
    }

    # ─── Stage 2: 3D Reconstruction ──────────────────────────────────
    if skip_3d:
        print("\nSkipping 3D reconstruction (--skip_3d). Cannot continue without point cloud.")
        return results

    recon_dir = os.path.join(output_dir, "reconstruction")
    if recon_model == "vggt":
        recon_script = os.path.join(SRC_DIR, "pipeline", "run_vggt.py")
        recon_env = "vv_vggt"
        recon_desc = ("3D Reconstruction (VGGT, " +
                      ("PER-FRAME low-memory" if low_memory else "BATCHED multi-view")
                      + ")")
        is_metric = False
    else:
        recon_script = os.path.join(SRC_DIR, "pipeline", "run_amb3r.py")
        recon_env = "amb3r"
        recon_desc = "3D Reconstruction (AMB3R)"
        is_metric = True

    recon_args = [
        "--image_dir", recon_input_dir,
        "--output_dir", recon_dir,
        "--max_images", str(recon_n),
        "--conf_threshold", "0.0",
    ]
    if recon_model == "vggt" and low_memory:
        recon_args.append("--per_frame")

    run_in_env(recon_env, recon_script, recon_args, description=recon_desc)
    results["stages"]["reconstruction"] = {
        "output_dir": recon_dir, "model": recon_model, "is_metric": is_metric,
    }

    npz_path = os.path.join(recon_dir, "point_cloud.npz")

    # ─── Stage 3: Apply manual 3D scale calibration ───────────────────
    if scale_calibration and os.path.exists(scale_calibration):
        print("\n" + "="*60 + "\nSTAGE 3: Applying Manual 3D Scale\n" + "="*60)
        from calibration.manual_scale import (
            compute_3d_scale_factor, apply_3d_scale_to_npz,
        )

        # Use recon_meta to get the EXACT frame ordering and per-frame
        # preprocess transforms (pad/crop) used by VGGT/AMB3R.
        # Without this, click coordinates would be misaligned with the
        # reconstruction's point map (especially for pad-mode reconstructions
        # of portrait/landscape inputs with letterboxing).
        recon_meta_path = os.path.join(recon_dir, "reconstruction_meta.json")
        scale_3d = compute_3d_scale_factor(
            npz_path, scale_calibration,
            recon_meta_path=recon_meta_path,
        )
        if scale_3d is not None:
            factor = scale_3d["scale_factor"]
            print(f"3D scale factor (chosen=MEDIAN): {factor:.4f}")
            print(f"  for reference: mean={scale_3d.get('scale_factor_mean', 0):.4f}, "
                  f"std={scale_3d['scale_factor_std']:.4f}, "
                  f"n={scale_3d['n_frames_used']} frames")
            apply_3d_scale_to_npz(npz_path, factor)
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(os.path.join(recon_dir, "point_cloud.ply"))
                pts = np.asarray(pcd.points) * factor
                pcd.points = o3d.utility.Vector3dVector(pts)
                o3d.io.write_point_cloud(os.path.join(recon_dir, "point_cloud.ply"), pcd)
            except Exception as e:
                print(f"  (PLY rescale skipped: {e})")
            results["stages"]["scale_3d"] = {"factor": factor, "details": scale_3d.get("details")}
        else:
            print("WARNING: 3D scale could not be computed; volume estimates will be in raw units")
            results["stages"]["scale_3d"] = {"skipped": True}
    else:
        print("\nNOTE: No scale_calibration provided. Volume estimates will be in raw units.")
        print("      Run scale_picker.py first, then pass --scale_calibration.")
        results["stages"]["scale_3d"] = {"skipped": True}

    # ─── Stage 4: Pose on AMB3R-resolution images (for feet) ─────────
    # Skipped for non-person subjects (e.g., balloon test) since there's
    # no person to detect feet for, and distance-to-ground isn't meaningful.
    pose_amb3r_dir = os.path.join(output_dir, "pose_amb3r")
    amb3r_imgs_dir = os.path.join(recon_dir, "amb3r_images")
    pose_results_path = None

    if not run_pose_stage:
        print(f"\nSkipping pose stage (subject={subject}, no person to detect)")
        results["stages"]["pose"] = {"skipped": True,
                                       "reason": f"subject={subject}"}
    elif os.path.isdir(amb3r_imgs_dir) and env_python("pose_env"):
        try:
            run_in_env(
                "pose_env",
                os.path.join(SRC_DIR, "pipeline", "run_pose.py"),
                [
                    "--image_dir", amb3r_imgs_dir,
                    "--output_dir", pose_amb3r_dir,
                    "--model", "human",
                    "--no_vis",
                ],
                description="2D Pose on Reconstruction Images (for feet location)",
            )
            pose_results_path = os.path.join(pose_amb3r_dir, "pose_results.json")
            results["stages"]["pose"] = {"output": pose_results_path}
        except Exception as e:
            print(f"WARNING: pose detection failed: {e}")
            results["stages"]["pose"] = {"skipped": True, "reason": str(e)}
    else:
        results["stages"]["pose"] = {"skipped": True}

    # ─── Stage 5: Belly Mesh + Volume + Button + Distance ────────────
    print("\n" + "="*60 + "\nSTAGE 5: Belly Analysis\n" + "="*60)
    from measurements.belly import run_belly_pipeline as run_belly

    belly_dir = os.path.join(output_dir, "belly")
    recon_meta_path = os.path.join(recon_dir, "reconstruction_meta.json")
    try:
        belly_results = run_belly(
            amb3r_npz_path=npz_path,
            segmentation_dir=seg_dir,
            output_dir=belly_dir,
            pose_results_path=pose_results_path,
            conf_pct_keep=conf_pct_keep,
            poisson_depth=poisson_depth,
            recon_meta_path=recon_meta_path if os.path.exists(recon_meta_path) else None,
        )
        results["stages"]["belly"] = {
            "output_dir": belly_dir,
            "volume_cm3": belly_results.get("volume", {}).get("bulge_volume_cm3"),
            "belly_button_3d": belly_results.get("belly_button", {}).get("position_3d"),
            "distance_to_midfeet_cm": belly_results.get("distances", {}).get("distance_belly_to_midfeet_cm"),
        }
    except Exception as e:
        print(f"ERROR: belly analysis failed: {e}")
        import traceback
        traceback.print_exc()
        results["stages"]["belly"] = {"error": str(e)}

    # ─── Stage 6: Debug Visualizations ─────────────────────────────
    print("\n" + "="*60 + "\nSTAGE 6: Generating Debug Visualizations\n" + "="*60)
    try:
        from visualization.debug_viz import run_belly_debug
        run_belly_debug(output_dir, image_dir=image_dir)
        results["stages"]["debug"] = {"output_dir": os.path.join(output_dir, "debug")}
    except Exception as e:
        print(f"WARNING: debug viz failed: {e}")
        import traceback
        traceback.print_exc()
        results["stages"]["debug"] = {"error": str(e)}

    # ─── Save final results ──────────────────────────────────────────
    elapsed = time.time() - start_total
    results["total_time_seconds"] = round(elapsed, 1)
    out_json = os.path.join(output_dir, "pipeline_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BELLY PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"Summary: {out_json}")
    print(f"{'='*60}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Belly Analysis Pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image_dir", help="Directory of input images")
    src.add_argument("--video", help="Path to video file (frames extracted automatically)")
    parser.add_argument("--n_frames", type=int, default=30,
                        help="Frames to extract from video (default 30). MUST match "
                             "the picker's setting so seed clicks align.")
    parser.add_argument("--recon_max_frames", type=int, default=20,
                        help="Cap on frames used for 3D reconstruction (default 20). "
                             "Pipeline takes the intersection of (extracted frames) "
                             "and (frames where SAM3 found something), then uniformly "
                             "samples this many. Lower = less VRAM but worse fusion.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scale_calibration", default=None,
                        help="Path to scale_calibration.json (required for metric volume)")
    parser.add_argument("--recon_model", default="vggt", choices=["amb3r", "vggt"],
                        help="3D reconstruction (default: vggt, which uses pad "
                             "mode to preserve all image pixels — required for "
                             "accurate mask alignment with portrait video)")
    parser.add_argument("--subject", default="pregnant",
                        choices=list(SUBJECT_PRESETS.keys()),
                        help="What's in the video. 'pregnant': run pose for "
                             "feet location and belly→ground distance. "
                             "'balloon': skip pose (no person), use balloon-"
                             "specific SAM3 prompts. Default: pregnant.")
    parser.add_argument("--sam_prompt", default=None,
                        help="SAM3 primary text prompt (default: 'person', which "
                             "works for clothed patients; the protrusion-based "
                             "belly volume algorithm works on the front-torso surface)")
    parser.add_argument("--sam_fallback_prompts", default=None,
                        help="Comma-separated fallback prompts if primary fails. "
                             "If omitted, derived from --subject.")
    parser.add_argument("--sam_confidence", type=float, default=0.25)
    parser.add_argument("--seed_points", type=str, default=None,
                        help="Path to belly_seed.json (from scale_picker --mode seed). "
                             "Only used if seed prompting is enabled for this subject "
                             "(see --use_seed_points / subject preset).")
    parser.add_argument("--use_seed_points", dest="use_seed_points",
                        action="store_true", default=None,
                        help="Force-enable SAM3 seed-point box prompts (override subject preset)")
    parser.add_argument("--no_seed_points", dest="use_seed_points",
                        action="store_false",
                        help="Force-disable SAM3 seed-point box prompts")
    parser.add_argument("--low_memory", dest="low_memory", action="store_true",
                        default=False,
                        help="VGGT per-frame mode: ~3x less peak VRAM but the "
                             "per-frame point clouds are in INDEPENDENT coordinate "
                             "systems → they overlap incorrectly when merged. "
                             "Only use as a fallback when batched mode OOMs; "
                             "prefer reducing --n_frames first.")
    parser.add_argument("--no_low_memory", dest="low_memory", action="store_false",
                        help="(default) Run VGGT in batched multi-view mode for "
                             "proper cross-frame fusion.")
    parser.add_argument("--poisson_depth", type=int, default=8)
    parser.add_argument("--conf_pct_keep", type=int, default=75,
                        help="Top X%% of point confidences to keep (default 75)")
    parser.add_argument("--skip_3d", action="store_true")
    args = parser.parse_args()

    run_belly_pipeline(
        image_dir=args.image_dir,
        video=args.video,
        n_frames=args.n_frames,
        recon_max_frames=args.recon_max_frames,
        output_dir=args.output_dir,
        scale_calibration=args.scale_calibration,
        recon_model=args.recon_model,
        subject=args.subject,
        sam_prompt=args.sam_prompt,  # None → derived from subject
        sam_fallback_prompts=(
            tuple(s.strip() for s in args.sam_fallback_prompts.split(",") if s.strip())
            if args.sam_fallback_prompts else None  # None → derived from subject
        ),
        seed_points=args.seed_points,
        use_seed_points=args.use_seed_points,  # None unless --use_seed_points / --no_seed_points
        low_memory=args.low_memory,
        sam_confidence=args.sam_confidence,
        poisson_depth=args.poisson_depth,
        conf_pct_keep=args.conf_pct_keep,
        skip_3d=args.skip_3d,
    )


if __name__ == "__main__":
    main()
