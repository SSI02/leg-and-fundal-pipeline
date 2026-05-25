"""
Leg Deformity Pipeline — clean orchestrator (mirrors the belly pipeline).

Pipeline stages:
    0. Frame extraction (if input is a video; reuses picker's manifest)
    1. SAM3 person segmentation (text-prompted, optional seed-point clicks)
    2. Frame selection — intersect with valid SAM3 frames, uniformly subsample
    3. 3D reconstruction (VGGT default, pad mode) on selected frames
    4. Apply manual 3D scale calibration (from scale_picker --mode scale)
    5. 2D pose estimation on reconstruction-resolution images
    6. Per-frame leg measurements + robust aggregation with margin-based
       classification (HKA, MAD, lengths, ratios, LLD)
    7. Debug visualizations

Required conda envs:
    - leg_pipeline    : orchestration, measurements, visualisation
    - vv_sam3         : SAM3 person segmentation
    - vv_vggt (or amb3r) : 3D reconstruction
    - pose_env        : 2D pose (RTMPose / ViTPose++)

Usage:
    conda activate leg_pipeline
    python src/pipeline/leg_orchestrator.py \\
        --video data/input/patient.mp4 \\
        --output_dir data/output/patient \\
        --scale_calibration data/input/patient_scale.json
"""

import os
import sys
import json
import argparse
import subprocess
import time
import glob

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ════════════════════════════════════════════════════════════════════
#  Conda env helpers (same convention as belly_orchestrator.py)
# ════════════════════════════════════════════════════════════════════

def _get_conda_prefix():
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        base_candidate = os.path.dirname(os.path.dirname(conda_prefix))
        if os.path.isdir(os.path.join(base_candidate, "envs")):
            return base_candidate
        if os.path.isdir(os.path.join(conda_prefix, "envs")):
            return conda_prefix
    for candidate in [os.path.expanduser("~/miniconda3"),
                       os.path.expanduser("~/anaconda3"),
                       os.path.expanduser("~/miniforge3"), "/opt/conda"]:
        if os.path.isdir(os.path.join(candidate, "envs")):
            return candidate
    raise RuntimeError("Could not find conda installation.")


def _env_python(env_name):
    base = _get_conda_prefix()
    p = os.path.join(base, "envs", env_name, "bin", "python")
    return p if os.path.exists(p) else None


def _run_in_env(env_name, script_path, args_list, description=""):
    py = _env_python(env_name)
    if py is None:
        raise FileNotFoundError(f"Env '{env_name}' not found.")
    cmd = [py, script_path] + args_list
    print(f"\n{'='*60}\nSTAGE: {description}\nEnv: {env_name}\n"
          f"Cmd: {' '.join(cmd)}\n{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(
            f"Stage '{description}' failed (rc={result.returncode}). "
            f"stderr tail:\n{result.stderr[-1500:]}"
        )
    if result.stdout:
        print(result.stdout[-2000:])
    print(f"Completed in {elapsed:.1f}s")


# ════════════════════════════════════════════════════════════════════
#  Subject presets — controls SAM3 prompts and behavior per body type
# ════════════════════════════════════════════════════════════════════

SUBJECT_PRESETS = {
    "standing": {
        # Standing patient, lower body in frame (or full body). SAM3 segments
        # the full person; the leg measurement code then uses pose keypoints
        # to pick out the leg landmarks specifically.
        "sam_prompt": "person",
        "sam_fallback_prompts": ("body", "human", "patient"),
        "use_seed_points": False,   # text prompt 'person' segments the whole
                                    # patient reliably; no need to seed.
    },
}


def _build_person_pointcloud(points_per_frame, _images_per_frame_unused,
                              segmentation_dir, recon_meta, image_order):
    """Apply per-frame SAM3 masks to the point map and stack the surviving
    points into a single (N, 3) cloud.

    This mirrors the belly pipeline's mask-aligned cloud build but keeps
    only the geometry — colours aren't needed for slab fitting. Mask
    alignment via the recon's preprocess transforms is essential because
    VGGT pad-mode reshapes the canvas; a naive resize bleeds points across
    the body boundary.
    """
    from PIL import Image as PILImage
    try:
        from measurements.belly import _transform_mask_to_recon_space
    except ImportError:
        _transform_mask_to_recon_space = None

    seg_path = os.path.join(segmentation_dir, "segmentation.json")
    if not os.path.exists(seg_path):
        return None
    with open(seg_path) as f:
        seg = json.load(f)

    T, H, W = points_per_frame.shape[:3]
    transforms_by_name = {
        t["filename"]: t for t in recon_meta.get("preprocess_transforms", [])
    }

    person_chunks = []
    for t, name in enumerate(image_order[:T]):
        if name not in seg:
            continue
        mask_path = seg[name].get("combined_mask_path")
        if not mask_path or not os.path.exists(mask_path):
            continue
        orig_mask = np.array(PILImage.open(mask_path).convert("L"))
        tf = transforms_by_name.get(name)
        if tf is not None and _transform_mask_to_recon_space is not None:
            mask = _transform_mask_to_recon_space(orig_mask, tf, H, W)
        else:
            mask = (np.array(PILImage.fromarray(orig_mask).resize(
                    (W, H), PILImage.NEAREST)) > 127)
        pts = points_per_frame[t][mask]
        # Drop invalid (near-origin) points from VGGT
        pts = pts[np.linalg.norm(pts, axis=-1) > 0.01]
        if len(pts) > 0:
            person_chunks.append(pts)

    if not person_chunks:
        return None
    return np.concatenate(person_chunks, axis=0)


def _build_headline(assessment, left_vol, right_vol, metric_calibrated):
    """Compose a single human-readable line that captures the bottom line."""
    parts = []
    parts.append(assessment.overall_assessment or "Assessment unavailable")
    for side, leg, vol in [("L", assessment.left, left_vol),
                            ("R", assessment.right, right_vol)]:
        if leg.hka_deviation_deg_median is None:
            continue
        bits = [f"{side}: {leg.classification}"]
        if leg.severity and leg.severity != "none":
            bits[-1] += f"/{leg.severity}"
        bits.append(f"dev {leg.hka_deviation_deg_median:+.1f}°")
        if metric_calibrated and vol is not None and vol.volume_cm3 is not None:
            bits.append(f"vol {vol.volume_cm3:.0f} cm³")
        parts.append(" ".join(bits))
    if metric_calibrated and assessment.intercondylar_distance_cm is not None:
        parts.append(
            f"knee gap {assessment.intercondylar_distance_cm:.1f}cm / "
            f"ankle gap {assessment.intermalleolar_distance_cm:.1f}cm"
        )
    return "  ·  ".join(parts)


# ════════════════════════════════════════════════════════════════════
#  Pipeline driver
# ════════════════════════════════════════════════════════════════════

def run_leg_pipeline(
    image_dir=None,
    video=None,
    n_frames=30,
    recon_max_frames=20,
    output_dir=None,
    scale_calibration=None,
    seed_points=None,
    recon_model="vggt",          # 'vggt' (default, pad mode) or 'amb3r'
    subject="standing",
    sam_prompt=None,             # None → from subject preset
    sam_fallback_prompts=None,   # None → from subject preset
    sam_confidence=0.25,
    use_seed_points=None,        # None → from subject preset
    pose_model="human",          # RTMPose-m default
    low_memory=False,            # per-frame VGGT (slower, less VRAM, worse fusion)
    skip_3d=False,
    anterior_frame=None,         # filename OR integer index of the chosen
                                 # anterior frame. When set, single-frame 2D
                                 # HKA is the PRIMARY classification (recon
                                 # is still used for volume).
):
    """Run the full leg pipeline end-to-end. See the module docstring."""
    assert image_dir or video, "Need image_dir or video"
    os.makedirs(output_dir, exist_ok=True)

    # Resolve subject preset
    if subject not in SUBJECT_PRESETS:
        raise ValueError(f"Unknown subject '{subject}'. "
                          f"Choices: {list(SUBJECT_PRESETS)}")
    preset = SUBJECT_PRESETS[subject]
    if sam_prompt is None:
        sam_prompt = preset["sam_prompt"]
    if sam_fallback_prompts is None:
        sam_fallback_prompts = preset["sam_fallback_prompts"]
    if use_seed_points is None:
        use_seed_points = preset.get("use_seed_points", False)

    print(f"\n[subject={subject}] SAM prompt='{sam_prompt}', "
          f"fallbacks={list(sam_fallback_prompts)}, "
          f"seed-prompt {'ON' if use_seed_points else 'OFF'}, "
          f"recon={recon_model}, low_memory={low_memory}")

    results = {
        "output_dir": output_dir,
        "subject": subject,
        "stages": {},
    }
    start_total = time.time()

    # ─── Stage 0: Frame extraction (if video) ──────────────────────────
    if video:
        print("\n" + "=" * 60 + "\nSTAGE 0: Frame extraction\n" + "=" * 60)
        from calibration.extract_frames import extract_frames
        frames_dir = os.path.splitext(video)[0] + "_frames"

        manifest_path = os.path.join(frames_dir, "_video_frame_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                existing = json.load(f)
            existing_n = len(existing.get("saved", []))
            if existing_n == n_frames:
                print(f"Reusing existing {existing_n} frames in {frames_dir} "
                      f"(manifest matches n_frames={n_frames})")
            else:
                print(f"\nERROR: {frames_dir} already has {existing_n} frames; "
                      f"you asked for n_frames={n_frames}. Re-extracting would "
                      f"invalidate any clicks saved against the existing layout.")
                print(f"  Either re-run the picker with --n_frames {n_frames},")
                print(f"  or re-run this pipeline with --n_frames {existing_n}.")
                sys.exit(1)
        else:
            extract_frames(video, frames_dir, n_frames=n_frames)
        image_dir = frames_dir
        results["stages"]["frame_extraction"] = {
            "frames_dir": frames_dir, "n_frames": n_frames,
            "reused_existing": os.path.exists(manifest_path),
        }

    # ─── Stage 1: SAM3 person segmentation ────────────────────────────
    seg_dir = os.path.join(output_dir, "segmentation")
    sam3_script = os.path.join(SRC_DIR, "pipeline", "run_sam3.py")

    if _env_python("vv_sam3") is None:
        print("\nERROR: vv_sam3 env not found. Person segmentation is required.")
        sys.exit(1)

    sam3_args = [
        "--image_dir", image_dir,
        "--output_dir", seg_dir,
        "--prompt", sam_prompt,
        "--confidence", str(sam_confidence),
    ]
    if sam_fallback_prompts:
        sam3_args += ["--fallback_prompts", ",".join(sam_fallback_prompts)]
    if use_seed_points and seed_points and os.path.exists(seed_points):
        sam3_args += ["--seed_points", seed_points]
        print(f"Using seed points from {seed_points}")
    elif seed_points and not use_seed_points:
        print(f"NOTE: SEED_POINTS provided but ignored "
              f"(subject={subject} → text-prompt-only is preferred).")
    _run_in_env("vv_sam3", sam3_script, sam3_args,
                  description=f"SAM3 Person Segmentation (prompt='{sam_prompt}')")
    results["stages"]["segmentation"] = {
        "output_dir": seg_dir, "prompt": sam_prompt,
        "fallback_prompts": list(sam_fallback_prompts),
    }

    # ─── Stage 2: Frame selection ──────────────────────────────────────
    seg_json_path = os.path.join(seg_dir, "segmentation.json")
    selected_basenames = []
    if os.path.exists(seg_json_path):
        with open(seg_json_path) as f:
            seg_data = json.load(f)
        valid_frames = sorted(
            n for n, r in seg_data.items()
            if (r.get("status") == "ok") or (r.get("num_detections", 0) > 0)
        )
        n_valid = len(valid_frames)
        if n_valid == 0:
            print("\nERROR: SAM3 found no person in any frame.")
            sys.exit(1)
        if n_valid > recon_max_frames:
            idxs = np.linspace(0, n_valid - 1, recon_max_frames, dtype=int)
            selected_basenames = [valid_frames[i] for i in idxs]
            print(f"\n[frame selection] {n_valid} valid frames; "
                  f"uniformly sampled {len(selected_basenames)} for reconstruction:")
        else:
            selected_basenames = valid_frames
            print(f"\n[frame selection] all {n_valid} valid frames used "
                  f"(≤ recon_max_frames={recon_max_frames}):")
        for n in selected_basenames:
            print(f"   • {n}")

    recon_frames_dir = os.path.join(output_dir, "recon_frames")
    os.makedirs(recon_frames_dir, exist_ok=True)
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
        recon_input_dir = image_dir
        recon_n = recon_max_frames

    results["stages"]["frame_selection"] = {
        "selected_basenames": selected_basenames,
        "n_selected": len(selected_basenames),
        "recon_input_dir": recon_input_dir,
    }

    # ─── Stage 3: 3D reconstruction ────────────────────────────────────
    if skip_3d:
        print("\nSkipping 3D reconstruction (--skip_3d).")
        return results

    recon_dir = os.path.join(output_dir, "reconstruction")
    if recon_model == "vggt":
        recon_script = os.path.join(SRC_DIR, "pipeline", "run_vggt.py")
        recon_env = "vv_vggt"
        recon_desc = ("3D Reconstruction (VGGT, " +
                      ("PER-FRAME low-memory" if low_memory else "BATCHED multi-view")
                      + ")")
        is_metric = False  # VGGT is arbitrary scale until manual calibration
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
    _run_in_env(recon_env, recon_script, recon_args, description=recon_desc)
    results["stages"]["reconstruction"] = {
        "output_dir": recon_dir, "model": recon_model, "is_metric": is_metric,
    }
    npz_path = os.path.join(recon_dir, "point_cloud.npz")
    recon_meta_path = os.path.join(recon_dir, "reconstruction_meta.json")

    # ─── Stage 4: Apply manual 3D scale calibration ────────────────────
    metric_calibrated = False
    if scale_calibration and os.path.exists(scale_calibration):
        print("\n" + "=" * 60 + "\nSTAGE 4: Manual 3D Scale\n" + "=" * 60)
        from calibration.manual_scale import (
            compute_3d_scale_factor, apply_3d_scale_to_npz,
        )
        scale_3d = compute_3d_scale_factor(
            npz_path, scale_calibration, recon_meta_path=recon_meta_path,
        )
        if scale_3d is not None and scale_3d.get("scale_factor"):
            factor = scale_3d["scale_factor"]
            print(f"3D scale factor (MEDIAN): {factor:.4f}")
            print(f"  for reference: mean={scale_3d.get('scale_factor_mean', 0):.4f}, "
                  f"std={scale_3d['scale_factor_std']:.4f}, "
                  f"n={scale_3d['n_frames_used']} frames")
            apply_3d_scale_to_npz(npz_path, factor)
            metric_calibrated = True
            results["stages"]["scale_3d"] = {
                "factor": factor,
                "n_frames_used": scale_3d.get("n_frames_used"),
                "scale_factor_std": scale_3d.get("scale_factor_std"),
            }
        else:
            print("WARNING: 3D scale could not be computed; lengths will be in raw units")
            results["stages"]["scale_3d"] = {"skipped": True}
    else:
        print("\nNo scale_calibration.json provided — distances will be in "
              "VGGT's arbitrary units. Run scale_picker.py to calibrate.")
        results["stages"]["scale_3d"] = {"skipped": True}

    # ─── Stage 5: 2D pose on reconstruction-resolution images ──────────
    amb3r_imgs_dir = os.path.join(recon_dir, "amb3r_images")
    pose_out_dir = os.path.join(output_dir, "pose")

    if not os.path.isdir(amb3r_imgs_dir):
        print("\nERROR: reconstruction images not saved; cannot run pose.")
        sys.exit(1)
    if _env_python("pose_env") is None:
        print("\nERROR: pose_env not found.")
        sys.exit(1)

    pose_script = os.path.join(SRC_DIR, "pipeline", "run_pose.py")
    _run_in_env("pose_env", pose_script, [
        "--image_dir", amb3r_imgs_dir,
        "--output_dir", pose_out_dir,
        "--model", pose_model,
    ], description=f"2D Pose Estimation (model='{pose_model}')")
    pose_results_path = os.path.join(pose_out_dir, "pose_results.json")
    results["stages"]["pose"] = {"output": pose_results_path, "model": pose_model}

    # ─── Stage 6: Per-frame measurements + aggregation + classification ─
    print("\n" + "=" * 60 + "\nSTAGE 6: Clinical Measurements + Classification\n"
          + "=" * 60)
    from measurements.leg_metrics import measure_from_pose_and_pointmap
    from dataclasses import asdict

    npz = np.load(npz_path, allow_pickle=True)
    points_per_frame = npz["points_per_frame"]   # (T, H, W, 3)

    with open(recon_meta_path) as f:
        recon_meta = json.load(f)
    image_order = recon_meta.get("image_files_in_order", []) or selected_basenames

    with open(pose_results_path) as f:
        pose_results = json.load(f)

    # ─── Stage 6 INTEGRITY CHECK ───────────────────────────────────────
    # Verify that pose_results, image_order, and the actual files on disk
    # in amb3r_images/ all reference the SAME filenames. A mismatch means
    # a stale run leaked through (recon used different N_FRAMES than
    # picker, or sequential-vs-original filenames bug).
    actual_files = set(
        os.path.basename(p) for p in glob.glob(
            os.path.join(amb3r_imgs_dir, "*.jpg"))
    )
    pose_keys = set(pose_results.keys())
    order_set = set(image_order)
    if not (actual_files == pose_keys == order_set):
        print("\n" + "!" * 60)
        print(" ⚠  FRAME-NAME INCONSISTENCY DETECTED — assessment may be WRONG")
        print("!" * 60)
        only_disk = sorted(actual_files - pose_keys - order_set)
        only_pose = sorted(pose_keys - actual_files - order_set)
        only_order = sorted(order_set - actual_files - pose_keys)
        if only_disk:  print(f"  files on disk but not in pose/meta:    {only_disk[:5]}")
        if only_pose:  print(f"  in pose_results but missing on disk:   {only_pose[:5]}")
        if only_order: print(f"  in recon_meta but missing on disk:     {only_order[:5]}")
        print("  → This usually means a prior run had different "
              "N_FRAMES/RECON_MAX_FRAMES. Recommend:")
        print(f"     rm -rf {os.path.dirname(amb3r_imgs_dir)}")
        print(f"     rm -rf {os.path.dirname(pose_results_path)}")
        print(f"     then re-run the pipeline from scratch.")
        print("!" * 60 + "\n")

    # ─── Stage 6a: Single-frame 2D HKA on chosen anterior frame ────────
    # When --anterior_frame is provided, this is the PRIMARY classification
    # — clinically equivalent to a standing-radiograph HKA reading, much
    # more reliable than multi-frame 3D fusion on noisy captures.

    # Sanitize: treat empty string and literal "null" as missing — both can
    # leak in from a `jq -r` on a missing field or unset env var.
    if isinstance(anterior_frame, str) and anterior_frame.strip().lower() in (
            "", "null", "none"):
        anterior_frame = None

    anterior_assessment = None
    print("\n" + "─" * 60)
    if anterior_frame is not None:
        print(f"  CLASSIFICATION MODE: single anterior frame "
              f"(--anterior_frame = {anterior_frame!r})")
    else:
        print(f"  CLASSIFICATION MODE: multi-frame 3D aggregate "
              f"(LEGACY — no --anterior_frame given)")
        print(f"  ⚠ For best results, run:")
        print(f"  ⚠     bash scripts/run_anterior_picker.sh <video> "
              f"<patient>_anterior.json")
        print(f"  ⚠ Then re-run this pipeline with:")
        print(f"  ⚠     ANTERIOR_FRAME=$(jq -r .anterior_frame "
              f"<patient>_anterior.json)")
    print("─" * 60)

    if anterior_frame is not None:
        from measurements.leg_metrics import measure_anterior_frame_2d
        # Resolve frame_name from either an integer index or filename
        if isinstance(anterior_frame, str) and anterior_frame.isdigit():
            anterior_frame = int(anterior_frame)
        if isinstance(anterior_frame, int):
            if 0 <= anterior_frame < len(image_order):
                af_name = image_order[anterior_frame]
                af_idx = anterior_frame
            else:
                print(f"  ❌ [anterior-frame] index {anterior_frame} out of "
                      f"range (0..{len(image_order)-1}); falling back to "
                      f"multi-frame.")
                af_name = None; af_idx = None
        else:
            af_name = anterior_frame
            af_idx = (image_order.index(af_name) if af_name in image_order
                      else None)
        # If the chosen frame isn't in pose_results (e.g., the picker showed
        # the user all extracted frames but recon processed a subset),
        # snap to the NEAREST frame the pipeline actually processed.
        if af_name and af_name not in pose_results:
            import re
            m = re.search(r"(\d+)", af_name or "")
            if m and pose_results:
                target = int(m.group(1))
                def _frame_idx(n):
                    mm = re.search(r"(\d+)", n)
                    return int(mm.group(1)) if mm else 10**9
                near = min(pose_results.keys(),
                            key=lambda n: abs(_frame_idx(n) - target))
                print(f"  ↻ [anterior-frame] '{af_name}' not in pose_results; "
                      f"snapping to nearest available frame: '{near}'")
                af_name = near
                af_idx = (image_order.index(af_name)
                          if af_name in image_order else None)

        if af_name and af_name in pose_results:
            print(f"  ✓ Using '{af_name}' as the anterior-view classification "
                  f"source")
            anterior_assessment = measure_anterior_frame_2d(
                pose_results[af_name], af_name, af_idx,
            )
            if anterior_assessment.view_warning:
                print(f"  ⚠ {anterior_assessment.view_warning}")
            for side, dev, cls, sev in [
                ("LEFT",  anterior_assessment.left_hka_deviation_deg,
                          anterior_assessment.left_classification,
                          anterior_assessment.left_severity),
                ("RIGHT", anterior_assessment.right_hka_deviation_deg,
                          anterior_assessment.right_classification,
                          anterior_assessment.right_severity),
            ]:
                if dev is not None:
                    print(f"    {side}: dev={dev:+.2f}° → {cls} ({sev})")
        elif af_name:
            print(f"  ❌ [anterior-frame] '{af_name}' not in pose_results "
                  f"(available frames: {list(pose_results.keys())[:5]}...); "
                  f"falling back to multi-frame.")
        else:
            print(f"  ❌ Could not resolve anterior frame; falling back to "
                  f"multi-frame.")

    assessment, left_frames, right_frames = measure_from_pose_and_pointmap(
        pose_results, points_per_frame, image_order,
    )

    # ─── Lower-leg volume (slab-wise ellipse fit) ──────────────────────
    # Skip if no scale calibration — volumes would be in arbitrary units³
    # and meaningless to a clinician.
    left_vol = right_vol = None
    if metric_calibrated:
        from measurements.leg_metrics import compute_bilateral_lower_leg_volumes
        seg_dir = os.path.join(output_dir, "segmentation")
        try:
            person_pts = _build_person_pointcloud(
                points_per_frame, npz["images_per_frame"]
                                  if "images_per_frame" in npz else None,
                seg_dir, recon_meta, image_order,
            )
        except Exception as e:
            print(f"  [volume] could not build person cloud: {e}")
            person_pts = None
        if person_pts is not None and len(person_pts) > 0:
            print(f"\n  Computing lower-leg volumes from {len(person_pts):,} "
                  f"person-cloud points")
            left_vol, right_vol = compute_bilateral_lower_leg_volumes(
                person_pts, assessment, left_frames, right_frames,
                metric_calibrated=True,
            )

    leg_out = os.path.join(output_dir, "leg_assessment.json")

    # ─── Build the easy-to-read summary block ─────────────────────────
    def _fmt(v, suffix="", fmt="{:.2f}"):
        return None if v is None else (fmt.format(v) + suffix)

    # The JSON has TWO shapes depending on whether anterior_frame was set:
    #
    #   anterior mode (PREFERRED): classification fields come from the chosen
    #     anterior frame's 2D HKA. Multi-frame 3D is suppressed entirely —
    #     only the per-frame 3D keypoints survive (needed by the volume
    #     estimator for median knee/ankle positions). This keeps the JSON
    #     focused on the trustworthy single-frame call.
    #
    #   multi-frame mode (FALLBACK when no anterior frame is chosen): the
    #     full per-leg LegAggregate + bootstrap classification is emitted,
    #     same as before.

    if anterior_assessment is not None:
        # Side blocks built straight from the 2D anterior assessment.
        af = anterior_assessment
        def _af_side_block(prefix2):
            return {
                "classification":      getattr(af, f"{prefix2}_classification"),
                "severity":            getattr(af, f"{prefix2}_severity"),
                "hka_angle_deg":       getattr(af, f"{prefix2}_hka_deg"),
                "hka_deviation_deg":   getattr(af, f"{prefix2}_hka_deviation_deg"),
                "mad_px":              getattr(af, f"{prefix2}_mad_px"),
                "class_probabilities": {
                    k: round(v, 3)
                    for k, v in (getattr(af, f"{prefix2}_class_probabilities") or {}).items()
                },
                "note":                getattr(af, f"{prefix2}_note"),
                "hip_xy_px":           getattr(af, f"{prefix2}_hip_xy"),
                "knee_xy_px":          getattr(af, f"{prefix2}_knee_xy"),
                "ankle_xy_px":         getattr(af, f"{prefix2}_ankle_xy"),
            }
        left_block = _af_side_block("left")
        right_block = _af_side_block("right")

        summary_left = {
            "classification": af.left_classification,
            "severity": af.left_severity,
            "hka_deviation_deg": _fmt(af.left_hka_deviation_deg, "°"),
        }
        summary_right = {
            "classification": af.right_classification,
            "severity": af.right_severity,
            "hka_deviation_deg": _fmt(af.right_hka_deviation_deg, "°"),
        }
        if metric_calibrated and left_vol is not None and left_vol.volume_cm3:
            summary_left["lower_leg_volume_cm3"] = _fmt(left_vol.volume_cm3, " cm³", "{:.0f}")
        if metric_calibrated and right_vol is not None and right_vol.volume_cm3:
            summary_right["lower_leg_volume_cm3"] = _fmt(right_vol.volume_cm3, " cm³", "{:.0f}")

        summary = {
            "primary_method": "single_anterior_frame_2d",
            "frame_used": af.frame_name,
            "view_quality_label": af.view_quality_label,
            "view_warning": af.view_warning,
            "stance_symmetry_warning": af.stance_symmetry_warning,
            "leg_length_asymmetry_pct": af.leg_length_asymmetry_pct,
            "overall_assessment": af.overall_assessment,
            "left": summary_left,
            "right": summary_right,
        }

        headline_parts = [af.overall_assessment or ""]
        for s, leg, vol in [("L", left_block, left_vol),
                              ("R", right_block, right_vol)]:
            d = leg.get("hka_deviation_deg")
            if d is None: continue
            bit = f"{s}: {leg['classification']}"
            if leg["severity"] and leg["severity"] != "none":
                bit += f"/{leg['severity']}"
            bit += f"  dev {d:+.1f}°"
            if vol is not None and vol.volume_cm3 is not None:
                bit += f"  vol {vol.volume_cm3:.0f} cm³"
            headline_parts.append(bit)
        summary["headline"] = "  ·  ".join(headline_parts)

        out_data = {
            "subject": subject,
            "metric_calibrated": metric_calibrated,
            "primary_method": "single_anterior_frame_2d",
            "anterior_frame_assessment": asdict(af),
            "n_frames_total": min(len(image_order), points_per_frame.shape[0]),
            "left": left_block,
            "right": right_block,
            "overall_assessment": af.overall_assessment,
            "lower_leg_volume_left":
                asdict(left_vol) if left_vol is not None else None,
            "lower_leg_volume_right":
                asdict(right_vol) if right_vol is not None else None,
            # per-frame 3D keypoints retained for the volume slab viz;
            # multi-frame aggregate/classification fields are intentionally
            # NOT emitted in anterior mode.
            "per_frame_left":  [asdict(f) for f in left_frames],
            "per_frame_right": [asdict(f) for f in right_frames],
            "notes": [
                ("Primary classification: single-frame 2D HKA on the "
                 "chosen anterior frame (clinically equivalent to a "
                 "standing radiograph)."),
                ("Soft classification: each class gets a probability from a "
                 "Gaussian (σ = 2° measurement noise) over the bands "
                 "≤5° = normal; 5–7° = borderline; 7–10° = mild; "
                 "10–15° = moderate; > 15° = severe. Boundaries are NOT "
                 "hard cuts."),
                ("Lower-leg volume uses the slab-wise ellipse-fit method "
                 "ONLY. Requires --scale_calibration."),
                ("Multi-frame 3D classification is SUPPRESSED in anterior "
                 "mode — the per-frame 3D keypoints retained here are "
                 "used only for the volume slab fit."),
            ],
            "summary": summary,
        }
    else:
        # ── Fallback: multi-frame 3D aggregate (legacy path) ──────────
        def _leg_summary(leg, vol):
            s = {
                "classification": leg.classification,
                "severity": leg.severity,
                "hka_deviation_deg": _fmt(leg.hka_deviation_deg_median, "°"),
                "reliability": leg.reliability_label,
            }
            if metric_calibrated and leg.tibia_length_cm_median is not None:
                s["tibia_length_cm"] = _fmt(leg.tibia_length_cm_median,
                                             " cm", "{:.1f}")
            if vol is not None and vol.volume_cm3 is not None:
                s["lower_leg_volume_cm3"] = _fmt(vol.volume_cm3, " cm³", "{:.0f}")
            return s

        summary = {
            "primary_method": "multi_frame_3d_aggregate",
            "overall_assessment": assessment.overall_assessment,
            "view_quality_label": assessment.view_label,
            "left": _leg_summary(assessment.left, left_vol),
            "right": _leg_summary(assessment.right, right_vol),
        }
        if metric_calibrated:
            summary["knee_gap_cm"] = _fmt(
                assessment.intercondylar_distance_cm, " cm", "{:.2f}")
            summary["ankle_gap_cm"] = _fmt(
                assessment.intermalleolar_distance_cm, " cm", "{:.2f}")
            summary["leg_length_discrepancy"] = (
                None if assessment.leg_length_difference_cm is None
                else f"{assessment.leg_length_difference_cm:.2f} cm "
                     f"({assessment.leg_length_difference_pct:.1f}%) "
                     f"— {assessment.leg_length_classification} "
                     f"({assessment.leg_length_discrepancy_side} shorter)"
            )
            summary["stance_classification"] = (
                f"{assessment.genu_alignment_classification} "
                f"({assessment.genu_alignment_severity})"
                if assessment.genu_alignment_classification else None
            )
        summary["flags_count"] = len(assessment.flags or [])
        summary["headline"] = _build_headline(assessment, left_vol, right_vol,
                                                metric_calibrated)
        summary["recommendation"] = (
            "Multi-frame results are often unreliable on noisy captures. "
            "For trustworthy classification, run the anterior picker and "
            "re-run with ANTERIOR_FRAME=<filename>."
        )

        out_data = {
            "subject": subject,
            "metric_calibrated": metric_calibrated,
            "primary_method": "multi_frame_3d_aggregate",
            "n_frames_total": min(len(image_order), points_per_frame.shape[0]),
            "n_frames_used_left": assessment.left.n_frames_used,
            "n_frames_used_right": assessment.right.n_frames_used,
            "left": asdict(assessment.left),
            "right": asdict(assessment.right),
            "lower_leg_volume_left":
                asdict(left_vol) if left_vol is not None else None,
            "lower_leg_volume_right":
                asdict(right_vol) if right_vol is not None else None,
            "intercondylar_distance_cm": assessment.intercondylar_distance_cm,
            "intermalleolar_distance_cm": assessment.intermalleolar_distance_cm,
            "leg_length_difference_cm": assessment.leg_length_difference_cm,
            "leg_length_difference_pct": assessment.leg_length_difference_pct,
            "leg_length_discrepancy_side": assessment.leg_length_discrepancy_side,
            "leg_length_classification": assessment.leg_length_classification,
            "leg_length_note": assessment.leg_length_note,
            "genu_alignment_classification":
                assessment.genu_alignment_classification,
            "genu_alignment_severity": assessment.genu_alignment_severity,
            "genu_alignment_note": assessment.genu_alignment_note,
            "view_quality": assessment.view_quality,
            "view_label": assessment.view_label,
            "view_warning": assessment.view_warning,
            "view_separation_ratios": assessment.view_separation_ratios,
            "overall_assessment": assessment.overall_assessment,
            "flags": assessment.flags,
            "per_frame_left": [asdict(f) for f in left_frames],
            "per_frame_right": [asdict(f) for f in right_frames],
            "notes": [
                ("⚠ Multi-frame 3D aggregate mode — classifications from "
                 "this mode are often noisy on real captures. For best "
                 "results, run scripts/run_anterior_picker.sh, then re-run "
                 "with ANTERIOR_FRAME=<frame>."),
                ("Lower-leg volume uses the slab-wise ellipse-fit method "
                 "ONLY. Requires --scale_calibration."),
            ],
            "summary": summary,
        }
    with open(leg_out, "w") as f:
        json.dump(out_data, f, indent=2, default=float)
    print(f"\nSaved: {leg_out}")
    results["stages"]["leg_measurements"] = {"output": leg_out}

    # ─── Print summary ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    if anterior_assessment is not None:
        af = anterior_assessment
        print(f"OVERALL: {af.overall_assessment}")
        print(f"Method:  single-frame 2D HKA on {af.frame_name}")
        print(f"View:    {af.view_quality_label}  "
              f"(hip-sep {af.hip_sep_ratio:.2f}, "
              f"asym {af.leg_length_asymmetry_pct or 0:.0f}%)")
        if af.view_warning:
            print(f"  ⚠ {af.view_warning}")
        if af.stance_symmetry_warning:
            print(f"  ⚠ {af.stance_symmetry_warning}")
        print("=" * 70)
        for label, dev, cls, sev, probs, note, vol in [
            ("LEFT",  af.left_hka_deviation_deg, af.left_classification,
                      af.left_severity, af.left_class_probabilities or {},
                      af.left_note, left_vol),
            ("RIGHT", af.right_hka_deviation_deg, af.right_classification,
                      af.right_severity, af.right_class_probabilities or {},
                      af.right_note, right_vol),
        ]:
            if dev is None:
                print(f"  {label}: no measurement")
                continue
            print(f"\n  {label}: dev={dev:+.2f}°  →  {cls} ({sev})")
            print(f"    {note}")
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            print(f"    top probabilities: "
                  + "  ".join(f"{k} {v*100:.0f}%" for k, v in top))
            if vol is not None and vol.volume_cm3 is not None:
                print(f"    lower-leg volume: {vol.volume_cm3:.0f} cm³")
    else:
        print(f"OVERALL: {assessment.overall_assessment}")
        print("Method:  multi-frame 3D aggregate (LEGACY — consider using "
              "--anterior_frame for better results)")
        print("=" * 70)
        for side, leg in [("LEFT", assessment.left), ("RIGHT", assessment.right)]:
            if leg.hka_angle_deg_median is None:
                print(f"  {side}: not enough valid frames "
                      f"({leg.n_frames_used}/{leg.n_frames_total})")
                continue
            print(f"\n  {side} leg ({leg.n_frames_used}/{leg.n_frames_total} frames used):")
            print(f"    HKA dev: {leg.hka_deviation_deg_median:+.2f}° "
                  f"(IQR {leg.hka_deviation_deg_iqr:.2f}°) "
                  f"→ {leg.classification} ({leg.severity})")
            if metric_calibrated:
                print(f"    Femur {leg.femur_length_cm_median:.1f} cm   "
                      f"Tibia {leg.tibia_length_cm_median:.1f} cm")
        if assessment.flags:
            print("\n  Flags:")
            for f in assessment.flags:
                print(f"    • {f}")

    # ─── Stage 7: Debug visualisations ────────────────────────────────
    print("\n" + "=" * 60 + "\nSTAGE 7: Debug Visualisations\n" + "=" * 60)
    try:
        from visualization.debug_viz import run_leg_debug
        run_leg_debug(output_dir, image_dir=image_dir)
        results["stages"]["debug"] = {"output_dir": os.path.join(output_dir, "debug")}
    except Exception as e:
        print(f"WARNING: leg debug viz failed: {e}")
        import traceback; traceback.print_exc()
        results["stages"]["debug"] = {"error": str(e)}

    # ─── Save pipeline_results.json ───────────────────────────────────
    elapsed = time.time() - start_total
    results["total_time_seconds"] = round(elapsed, 1)
    out_json = os.path.join(output_dir, "pipeline_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n{'='*60}\nLEG PIPELINE COMPLETE ({elapsed:.1f}s)\n{'='*60}")
    print(f"  Results: {out_json}")
    print(f"  Assessment: {leg_out}")
    return results


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Leg Deformity Pipeline (clean)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image_dir", help="Directory of input images")
    src.add_argument("--video", help="Path to video file")
    parser.add_argument("--n_frames", type=int, default=30,
                        help="Frames to extract from video (default 30). MUST match "
                             "the scale-picker's setting.")
    parser.add_argument("--recon_max_frames", type=int, default=20,
                        help="Cap on frames used for 3D reconstruction (default 20).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scale_calibration", default=None,
                        help="Path to scale_calibration.json from scale_picker.py. "
                             "Required to get measurements in cm.")
    parser.add_argument("--seed_points", default=None,
                        help="Path to seed-point JSON (optional)")
    parser.add_argument("--recon_model", default="vggt",
                        choices=["amb3r", "vggt"],
                        help="Default: vggt (pad mode preserves all pixels).")
    parser.add_argument("--subject", default="standing",
                        choices=list(SUBJECT_PRESETS.keys()))
    parser.add_argument("--sam_prompt", default=None)
    parser.add_argument("--sam_fallback_prompts", default=None,
                        help="Comma-separated. If omitted, derived from subject preset.")
    parser.add_argument("--sam_confidence", type=float, default=0.25)
    parser.add_argument("--use_seed_points", dest="use_seed_points",
                        action="store_true", default=None)
    parser.add_argument("--no_seed_points", dest="use_seed_points",
                        action="store_false")
    parser.add_argument("--pose_model", default="human",
                        choices=["human", "vitpose", "vitpose-s",
                                  "vitpose-l", "vitpose-h", "wholebody"])
    parser.add_argument("--low_memory", action="store_true", default=False)
    parser.add_argument("--skip_3d", action="store_true", default=False)
    parser.add_argument("--anterior_frame", default=None,
                          help="Frame filename or integer index — when set, "
                               "classification uses single-frame 2D HKA on "
                               "this frame (clinically preferred over multi-"
                               "frame 3D for unclear cases).")
    args = parser.parse_args()

    run_leg_pipeline(
        image_dir=args.image_dir,
        video=args.video,
        n_frames=args.n_frames,
        recon_max_frames=args.recon_max_frames,
        output_dir=args.output_dir,
        scale_calibration=args.scale_calibration,
        seed_points=args.seed_points,
        recon_model=args.recon_model,
        subject=args.subject,
        sam_prompt=args.sam_prompt,
        sam_fallback_prompts=(
            tuple(s.strip() for s in args.sam_fallback_prompts.split(",") if s.strip())
            if args.sam_fallback_prompts else None
        ),
        sam_confidence=args.sam_confidence,
        use_seed_points=args.use_seed_points,
        pose_model=args.pose_model,
        low_memory=args.low_memory,
        skip_3d=args.skip_3d,
        anterior_frame=args.anterior_frame,
    )


if __name__ == "__main__":
    main()
