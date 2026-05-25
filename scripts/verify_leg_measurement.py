"""Re-run the measurement + volume stages on a patient using cached
pose + reconstruction + segmentation outputs. Verifies new flags and
the lower-leg volume estimator without re-running VGGT/SAM3/pose.
"""
import json
import os
import sys
from dataclasses import asdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from measurements.leg_metrics import (
    measure_from_pose_and_pointmap,
    compute_bilateral_lower_leg_volumes,
)
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))
from leg_orchestrator import _build_person_pointcloud, _build_headline

PATIENT = sys.argv[1] if len(sys.argv) > 1 else "patient_001"
OUT = os.path.join(ROOT, "data", "output", PATIENT)
npz_path = os.path.join(OUT, "reconstruction", "point_cloud.npz")
meta_path = os.path.join(OUT, "reconstruction", "reconstruction_meta.json")
pose_path = os.path.join(OUT, "pose", "pose_results.json")
seg_dir = os.path.join(OUT, "segmentation")

npz = np.load(npz_path, allow_pickle=True)
points_per_frame = npz["points_per_frame"]
with open(meta_path) as f:
    meta = json.load(f)
image_order = meta.get("image_files_in_order", [])
with open(pose_path) as f:
    pose_results = json.load(f)

print("=" * 70)
print(f"Verifying {PATIENT}  (cached pose + recon + seg)")
print(f"  Frames: T={points_per_frame.shape[0]}, image_order={len(image_order)}")
print("=" * 70)

assessment, left_frames, right_frames = measure_from_pose_and_pointmap(
    pose_results, points_per_frame, image_order,
)

# Existing JSON tells us metric_calibrated
with open(os.path.join(OUT, "leg_assessment.json")) as f:
    existing = json.load(f)
metric_calibrated = bool(existing.get("metric_calibrated", False))

left_vol = right_vol = None
if metric_calibrated:
    try:
        person_pts = _build_person_pointcloud(
            points_per_frame, None, seg_dir, meta, image_order,
        )
        if person_pts is not None and len(person_pts) > 0:
            print(f"  person cloud: {len(person_pts):,} raw points")
            left_vol, right_vol = compute_bilateral_lower_leg_volumes(
                person_pts, assessment, left_frames, right_frames,
                metric_calibrated=True,
            )
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  [WARN] could not compute volume: {e}")
else:
    print("  metric calibration unavailable — skipping volume")

a = assessment
print()
print("─" * 70)
print(f"View label:    {a.view_label}    quality={a.view_quality or 0:.2f}")
print(f"Frames used:   left={a.left.n_frames_used}, right={a.right.n_frames_used}")

def _side(name, agg, vol):
    print(f"\n[{name}]")
    if agg.hka_deviation_deg_median is None:
        print("  (no valid frames)")
        return
    print(f"  HKA dev:       {agg.hka_deviation_deg_median:+.2f}°  "
          f"IQR={agg.hka_deviation_deg_iqr:.2f}°")
    print(f"  Class:         {agg.classification} / {agg.severity}")
    print(f"  HKA reliab:    {agg.reliability_score:.2f} ({agg.reliability_label})")
    if vol is not None:
        print(f"  ── Lower-leg volume ─")
        print(f"  Tibia (kp):    {vol.tibia_length_cm or 0:.1f} cm")
        if vol.tibia_length_cloud_cm:
            print(f"  Tibia (cloud): {vol.tibia_length_cloud_cm:.1f} cm")
        print(f"  Points raw:    {vol.n_points_raw:,}")
        print(f"  Points used:   {vol.n_points_used:,}  "
              f"(after SOR: {vol.n_points_after_sor:,})")
        print(f"  Slabs:         {vol.n_slabs_with_data} / {vol.n_slabs_total}")
        if vol.volume_cm3 is not None:
            print(f"  Volume (slab): {vol.volume_cm3:.0f} cm³")
            if vol.volume_consensus_cm3 is not None:
                print(f"  Volume (cons): {vol.volume_consensus_cm3:.0f} cm³  "
                      f"range [{vol.volume_range_low_cm3:.0f}, "
                      f"{vol.volume_range_high_cm3:.0f}] cm³")
            print(f"  Vol reliab:    {vol.reliability_score:.2f} ({vol.reliability_label})")
            if vol.mean_circumference_cm:
                print(f"  Mean circ:     {vol.mean_circumference_cm:.1f} cm  "
                      f"(max {vol.max_circumference_cm:.1f} cm)")
        if vol.distortion_warning:
            print(f"  ⚠ {vol.distortion_warning}")
        if vol.note:
            print(f"  Note: {vol.note}")

_side("LEFT",  a.left, left_vol)
_side("RIGHT", a.right, right_vol)
print()
print("─" * 70)
print("Headline:")
print(" ", _build_headline(a, left_vol, right_vol, metric_calibrated))
print("─" * 70)
print("FLAGS:")
for fl in a.flags:
    print(f"  • {fl}")

# Persist the updated assessment + volume + summary
existing.update({
    "n_frames_total": min(len(image_order), points_per_frame.shape[0]),
    "n_frames_used_left": a.left.n_frames_used,
    "n_frames_used_right": a.right.n_frames_used,
    "left": asdict(a.left),
    "right": asdict(a.right),
    "lower_leg_volume_left":  asdict(left_vol)  if left_vol  is not None else None,
    "lower_leg_volume_right": asdict(right_vol) if right_vol is not None else None,
    "intercondylar_distance_cm": a.intercondylar_distance_cm,
    "intermalleolar_distance_cm": a.intermalleolar_distance_cm,
    "leg_length_difference_cm": a.leg_length_difference_cm,
    "leg_length_difference_pct": a.leg_length_difference_pct,
    "leg_length_discrepancy_side": a.leg_length_discrepancy_side,
    "leg_length_classification": a.leg_length_classification,
    "leg_length_note": a.leg_length_note,
    "genu_alignment_classification": a.genu_alignment_classification,
    "genu_alignment_severity": a.genu_alignment_severity,
    "genu_alignment_note": a.genu_alignment_note,
    "view_quality": a.view_quality,
    "view_label": a.view_label,
    "view_warning": a.view_warning,
    "view_separation_ratios": a.view_separation_ratios,
    "overall_assessment": a.overall_assessment,
    "flags": a.flags,
    "per_frame_left": [asdict(f) for f in left_frames],
    "per_frame_right": [asdict(f) for f in right_frames],
})

# Build summary block
def _fmt(v, suffix="", fmt="{:.2f}"):
    return None if v is None else (fmt.format(v) + suffix)

def _leg_summary(leg, vol):
    s = {
        "classification": leg.classification,
        "severity": leg.severity,
        "hka_deviation_deg": _fmt(leg.hka_deviation_deg_median, "°"),
        "reliability": leg.reliability_label,
    }
    if metric_calibrated and leg.tibia_length_cm_median is not None:
        s["tibia_length_cm"] = _fmt(leg.tibia_length_cm_median, " cm", "{:.1f}")
    if vol is not None and vol.volume_cm3 is not None:
        s["lower_leg_volume_cm3"] = _fmt(vol.volume_cm3, " cm³", "{:.0f}")
        if vol.volume_consensus_cm3 is not None:
            s["lower_leg_volume_consensus_cm3"] = _fmt(
                vol.volume_consensus_cm3, " cm³", "{:.0f}")
        if vol.volume_range_low_cm3 is not None and vol.volume_range_high_cm3 is not None:
            s["lower_leg_volume_range_cm3"] = (
                f"{vol.volume_range_low_cm3:.0f}-{vol.volume_range_high_cm3:.0f} cm³"
            )
        s["lower_leg_volume_reliability"] = vol.reliability_label
    return s

summary = {
    "overall_assessment": a.overall_assessment,
    "view_quality_label": a.view_label,
    "left":  _leg_summary(a.left,  left_vol),
    "right": _leg_summary(a.right, right_vol),
}
if metric_calibrated:
    summary["knee_gap_cm"]  = _fmt(a.intercondylar_distance_cm, " cm")
    summary["ankle_gap_cm"] = _fmt(a.intermalleolar_distance_cm, " cm")
    summary["leg_length_discrepancy"] = (
        None if a.leg_length_difference_cm is None
        else f"{a.leg_length_difference_cm:.2f} cm "
             f"({a.leg_length_difference_pct:.1f}%) — "
             f"{a.leg_length_classification} "
             f"({a.leg_length_discrepancy_side} shorter)"
    )
    summary["stance_classification"] = (
        f"{a.genu_alignment_classification} ({a.genu_alignment_severity})"
        if a.genu_alignment_classification else None
    )
summary["flags_count"] = len(a.flags or [])
summary["headline"] = _build_headline(a, left_vol, right_vol, metric_calibrated)
existing["summary"] = summary

out_path = os.path.join(OUT, "leg_assessment.json")
with open(out_path, "w") as f:
    json.dump(existing, f, indent=2, default=float)
print()
print(f"Updated → {out_path}")
