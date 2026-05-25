"""Verify the new ANTERIOR-MODE pipeline (single-frame 2D classification +
ellipse volume) using CACHED pose / reconstruction / segmentation.

Writes the new-shape leg_assessment.json and regenerates the pruned
visualisation set.

Usage:
    python scripts/verify_anterior_classification.py <patient_id> <frame_idx>
Example:
    python scripts/verify_anterior_classification.py patient_005 7
"""
import json
import os
import sys
from dataclasses import asdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))

from measurements.leg_metrics import (
    measure_anterior_frame_2d, measure_from_pose_and_pointmap,
    compute_bilateral_lower_leg_volumes,
)
from leg_orchestrator import _build_person_pointcloud
from visualization.debug_viz import run_leg_debug

PATIENT = sys.argv[1] if len(sys.argv) > 1 else "patient_005"
FRAME_IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 7

base = os.path.join(ROOT, "data", "output", PATIENT)
npz_path  = os.path.join(base, "reconstruction", "point_cloud.npz")
meta_path = os.path.join(base, "reconstruction", "reconstruction_meta.json")
pose_path = os.path.join(base, "pose", "pose_results.json")
seg_dir   = os.path.join(base, "segmentation")
recon_imgs = os.path.join(base, "reconstruction", "amb3r_images")
leg_dir   = os.path.join(base, "debug", "leg")
leg_out   = os.path.join(base, "leg_assessment.json")

print(f"=== {PATIENT} : anterior_frame_idx = {FRAME_IDX} ===")

npz = np.load(npz_path, allow_pickle=True)
points_per_frame = npz["points_per_frame"]
with open(meta_path) as f:
    meta = json.load(f)
image_order = meta.get("image_files_in_order", [])
with open(pose_path) as f:
    pose_results = json.load(f)
af_name = image_order[FRAME_IDX]

# 1. Single-frame 2D anterior assessment (PRIMARY)
af = measure_anterior_frame_2d(pose_results[af_name], af_name, FRAME_IDX)
print(f"Anterior frame: {af_name}   view={af.view_quality_label}   "
      f"hip-sep={af.hip_sep_ratio:.3f}   asym={af.leg_length_asymmetry_pct:.1f}%")
for s, dev, cls, sev in [
    ("L", af.left_hka_deviation_deg, af.left_classification, af.left_severity),
    ("R", af.right_hka_deviation_deg, af.right_classification, af.right_severity),
]:
    print(f"  {s}: dev={dev:+.2f}° → {cls}/{sev}")
print(f"  Overall: {af.overall_assessment}")

# 2. Multi-frame measurement — runs ONLY to extract per-frame 3D keypoints
#    that the volume estimator needs. Aggregates are discarded.
assessment, left_frames, right_frames = measure_from_pose_and_pointmap(
    pose_results, points_per_frame, image_order,
)

# 3. Volume (ellipse-only)
with open(leg_out) as f:
    existing = json.load(f)
metric_calibrated = bool(existing.get("metric_calibrated", False))
left_vol = right_vol = None
if metric_calibrated:
    person_pts = _build_person_pointcloud(
        points_per_frame, None, seg_dir, meta, image_order,
    )
    if person_pts is not None and len(person_pts) > 0:
        print(f"person cloud: {len(person_pts):,} points")
        left_vol, right_vol = compute_bilateral_lower_leg_volumes(
            person_pts, assessment, left_frames, right_frames,
            metric_calibrated=True,
        )

if left_vol:
    print(f"  L vol: {left_vol.volume_cm3:.0f} cm³  "
          f"(tibia kp={left_vol.tibia_length_cm:.1f}, "
          f"cloud={left_vol.tibia_length_cloud_cm or 0:.1f} cm)")
if right_vol:
    print(f"  R vol: {right_vol.volume_cm3:.0f} cm³  "
          f"(tibia kp={right_vol.tibia_length_cm:.1f}, "
          f"cloud={right_vol.tibia_length_cloud_cm or 0:.1f} cm)")


# 4. Build the new-shape JSON (anterior mode)
def _fmt(v, suffix="", fmt="{:.2f}"):
    return None if v is None else (fmt.format(v) + suffix)

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

# Headline
headline_parts = [af.overall_assessment or ""]
for s, leg_block, vol in [("L", _af_side_block("left"), left_vol),
                            ("R", _af_side_block("right"), right_vol)]:
    d = leg_block.get("hka_deviation_deg")
    if d is None: continue
    bit = f"{s}: {leg_block['classification']}"
    if leg_block["severity"] and leg_block["severity"] != "none":
        bit += f"/{leg_block['severity']}"
    bit += f"  dev {d:+.1f}°"
    if vol is not None and vol.volume_cm3 is not None:
        bit += f"  vol {vol.volume_cm3:.0f} cm³"
    headline_parts.append(bit)
summary["headline"] = "  ·  ".join(headline_parts)

out_data = {
    "subject": existing.get("subject", "standing"),
    "metric_calibrated": metric_calibrated,
    "primary_method": "single_anterior_frame_2d",
    "anterior_frame_assessment": asdict(af),
    "n_frames_total": min(len(image_order), points_per_frame.shape[0]),
    "left":  _af_side_block("left"),
    "right": _af_side_block("right"),
    "overall_assessment": af.overall_assessment,
    "lower_leg_volume_left":
        asdict(left_vol) if left_vol is not None else None,
    "lower_leg_volume_right":
        asdict(right_vol) if right_vol is not None else None,
    # per-frame 3D keypoints retained for the volume slab viz
    "per_frame_left":  [asdict(f) for f in left_frames],
    "per_frame_right": [asdict(f) for f in right_frames],
    "notes": [
        ("Primary classification: single-frame 2D HKA on the chosen "
         "anterior frame (clinically equivalent to a standing radiograph)."),
        ("Soft classification: each class gets a probability from a Gaussian "
         "(σ = 2° measurement noise) over the bands "
         "≤5° = normal; 5–7° = borderline; 7–10° = mild; "
         "10–15° = moderate; > 15° = severe. Boundaries are NOT hard cuts."),
        ("Lower-leg volume uses the slab-wise ellipse-fit method ONLY. "
         "Requires --scale_calibration."),
        ("Multi-frame 3D classification is SUPPRESSED in anterior mode — "
         "the per-frame 3D keypoints retained here are used only for the "
         "volume slab fit."),
    ],
    "summary": summary,
}

with open(leg_out, "w") as f:
    json.dump(out_data, f, indent=2, default=float)
print(f"\nWrote new-shape JSON: {leg_out}")

# 5. Regenerate visualisations (will auto-detect anterior mode + cleanup stale)
print()
print("─" * 60)
print("Regenerating visualisations...")
print("─" * 60)
input_frames_dir = os.path.join(ROOT, "data", "input", f"{PATIENT}_frames")
run_leg_debug(base, image_dir=input_frames_dir
              if os.path.isdir(input_frames_dir) else None)

# 6. List the resulting leg_dir contents so user can verify
print()
print("─" * 60)
print(f"Final contents of {leg_dir}:")
print("─" * 60)
for fname in sorted(os.listdir(leg_dir)):
    full = os.path.join(leg_dir, fname)
    if os.path.isdir(full):
        print(f"  📁 {fname}/  ({len(os.listdir(full))} files)")
    else:
        print(f"     {fname}")
