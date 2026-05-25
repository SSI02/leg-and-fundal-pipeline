"""
3D Clinical Measurements for Leg Deformity Assessment.

Maps 2D pose keypoints into AMB3R/VGGT per-pixel 3D point map to get
3D joint positions, then computes scale-invariant leg measurements:
  - HKA angles, deviation, varus/valgus classification
  - Femur/tibia ratio, leg symmetry ratio
  - Limb length asymmetry (percentage)

NOTE: AMB3R's metric scale head is unreliable (~40-67% of true).
All absolute distances are in raw reconstruction units and should NOT
be used for clinical decisions. Only angles and ratios are trustworthy.
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple


@dataclass
class Leg3D:
    """3D clinical measurements for one leg."""
    side: str
    hip_3d: Optional[List[float]] = None
    knee_3d: Optional[List[float]] = None
    ankle_3d: Optional[List[float]] = None

    # 3D angles (degrees)
    hka_angle_3d: Optional[float] = None
    hka_deviation_3d: Optional[float] = None

    # 3D lengths (metric — cm or meters depending on AMB3R output)
    femur_length_3d: Optional[float] = None
    tibia_length_3d: Optional[float] = None
    total_leg_length_3d: Optional[float] = None

    # Mechanical axis deviation in 3D (distance from knee to hip-ankle line)
    mad_3d: Optional[float] = None

    classification: Optional[str] = None
    severity: Optional[str] = None
    method: Optional[str] = None  # "direct_lookup" or "smpl_aligned"


@dataclass
class Assessment3D:
    """Full 3D leg deformity assessment."""
    left_leg: Optional[Leg3D] = None
    right_leg: Optional[Leg3D] = None

    intercondylar_distance_3d: Optional[float] = None
    intermalleolar_distance_3d: Optional[float] = None
    leg_length_difference_3d: Optional[float] = None
    leg_length_discrepancy_side: Optional[str] = None

    overall_classification: Optional[str] = None
    flags: list = field(default_factory=list)
    units: str = "amb3r_raw (NOT metric)"
    method: str = "direct_lookup"

    # Scale-invariant ratios
    ratios: Optional[dict] = None


# ─── Geometry utils ───────────────────────────────────────────────

def angle_3d(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3 in 3D. Returns degrees."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def distance_3d(p1, p2):
    """Euclidean distance in 3D."""
    return float(np.linalg.norm(np.array(p2) - np.array(p1)))


def point_to_line_distance_3d(point, line_start, line_end):
    """Perpendicular distance from a 3D point to a 3D line segment."""
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)
    ab = b - a
    ap = p - a
    cross = np.cross(ab, ap)
    return float(np.linalg.norm(cross) / (np.linalg.norm(ab) + 1e-8))


def classify_deformity(deviation_deg):
    """Classify based on HKA deviation. + = varus, - = valgus."""
    abs_dev = abs(deviation_deg)
    if abs_dev <= 3.0:
        return "normal", "none"
    classification = "varus" if deviation_deg > 0 else "valgus"
    if abs_dev <= 5.0:
        severity = "mild"
    elif abs_dev <= 10.0:
        severity = "moderate"
    else:
        severity = "severe"
    return classification, severity


def signed_deviation_3d(hip, knee, ankle, side):
    """Compute signed HKA deviation in 3D.

    Uses the coronal plane (frontal view) projection to determine
    varus vs valgus direction. In world coordinates, we project
    the mechanical axis and knee position onto the coronal plane.

    Convention: + = varus (lateral), - = valgus (medial)
    """
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)

    hka = angle_3d(hip, knee, ankle)
    abs_deviation = 180.0 - hka
    mad = point_to_line_distance_3d(knee, hip, ankle)

    # Determine sign: project onto coronal plane (X-Z or X-Y depending on orientation)
    # Use cross product of mechanical axis with knee offset
    mech_axis = ankle - hip
    to_knee = knee - hip
    cross = np.cross(mech_axis, to_knee)

    # The sign of the Y-component of the cross product indicates lateral/medial
    # This assumes Y is roughly vertical (up) and X is roughly left-right
    # For left leg: positive cross[1] → varus, negative → valgus
    # For right leg: negative cross[1] → varus, positive → valgus
    if side == "left":
        deviation = abs_deviation if cross[1] >= 0 else -abs_deviation
    else:
        deviation = abs_deviation if cross[1] < 0 else -abs_deviation

    return deviation, mad


# ─── Method 1: Direct Lookup from AMB3R point maps ────────────────

def lookup_3d_from_pointmap(
    keypoints_2d,
    points_per_frame,
    images_per_frame,
    frame_idx=0,
    search_radius=5,
):
    """Map 2D keypoints to 3D positions using AMB3R's per-pixel point map.

    Args:
        keypoints_2d: Dict of keypoint_name -> {"x": px, "y": py, "score": float}
        points_per_frame: (T, H, W, 3) AMB3R world_points per frame
        images_per_frame: (T, H, W, 3) images per frame (for resolution matching)
        frame_idx: Which frame to use for lookup
        search_radius: Pixel radius to average nearby 3D points (for robustness)

    Returns:
        Dict of keypoint_name -> [x, y, z] in metric 3D, or None if lookup failed.
    """
    pts_frame = points_per_frame[frame_idx]  # (H_amb, W_amb, 3)
    img_frame = images_per_frame[frame_idx]  # (H_img, W_img, 3)

    H_amb, W_amb = pts_frame.shape[:2]
    H_img, W_img = img_frame.shape[:2]

    results = {}
    for name, kp in keypoints_2d.items():
        if kp["score"] < 0.3:
            results[name] = None
            continue

        # Scale keypoint coordinates from original image space to AMB3R resolution
        # AMB3R uses 518x392 (W x H), but the pose was run on original resolution
        # We need to know the original image size to scale properly
        px = kp["x"]
        py = kp["y"]

        # Scale to AMB3R resolution
        # Note: AMB3R's Demo dataset resizes to (518, 392) — W=518, H=392
        # The pose keypoints are in original image coordinates
        # We need to scale: x_amb = x_orig * (W_amb / W_orig), y_amb = y_orig * (H_amb / H_orig)
        # H_img, W_img from AMB3R's stored images are already at AMB3R resolution
        # But the pose keypoints are at original resolution
        # We'll assume the caller provides the scale factors or original dimensions

        # For now, scale based on AMB3R's internal resolution
        # The keypoint coords need to be mapped to AMB3R's H_amb x W_amb grid
        # This will be handled by the caller passing scaled keypoints

        ix = int(round(px))
        iy = int(round(py))

        # Clamp to valid range
        ix = max(0, min(ix, W_amb - 1))
        iy = max(0, min(iy, H_amb - 1))

        # Average 3D points in a small neighborhood for robustness
        y_lo = max(0, iy - search_radius)
        y_hi = min(H_amb, iy + search_radius + 1)
        x_lo = max(0, ix - search_radius)
        x_hi = min(W_amb, ix + search_radius + 1)

        patch = pts_frame[y_lo:y_hi, x_lo:x_hi]  # (patch_h, patch_w, 3)

        # Filter out zero/invalid points
        valid = np.linalg.norm(patch, axis=-1) > 0.01
        if valid.sum() == 0:
            results[name] = None
            continue

        pt_3d = patch[valid].mean(axis=0)
        results[name] = pt_3d.tolist()

    return results


def measure_from_pointmap(
    pose_results_path,
    amb3r_npz_path,
    output_path,
):
    """Compute 3D clinical measurements by mapping 2D pose to AMB3R's 3D point map.

    IMPORTANT: The pose_results_path MUST be from running pose detection on
    AMB3R's stored images (amb3r_images/frame_XXX.jpg), NOT on the original
    images. This ensures 2D keypoint coordinates map directly to the point
    map pixels without any coordinate scaling.

    Args:
        pose_results_path: Path to pose_results.json (from AMB3R-resolution images)
        amb3r_npz_path: Path to point_cloud.npz from AMB3R
        output_path: Path to save 3D measurements JSON
    """
    with open(pose_results_path) as f:
        pose_results = json.load(f)

    amb3r_data = np.load(amb3r_npz_path, allow_pickle=True)
    pts_per_frame = amb3r_data["points_per_frame"]  # (T, H, W, 3)
    imgs_per_frame = amb3r_data["images_per_frame"]  # (T, H, W, 3)

    T, H_amb, W_amb, _ = pts_per_frame.shape
    print(f"  AMB3R point map: {T} frames, {H_amb}x{W_amb} (HxW)")

    all_assessments = {}
    image_names = sorted(pose_results.keys())

    for frame_idx, img_name in enumerate(image_names):
        if frame_idx >= T:
            break

        img_data = pose_results[img_name]
        frame_assessments = []

        for person in img_data["persons"]:
            lk = person["leg_keypoints"]

            # Check confidence
            min_score = 0.3
            has_legs = all(
                lk[n]["score"] > min_score
                for n in ["left_hip", "right_hip", "left_knee", "right_knee",
                          "left_ankle", "right_ankle"]
            )
            if not has_legs:
                frame_assessments.append({
                    "person_index": person["person_index"],
                    "error": "Insufficient leg keypoint confidence for 3D lookup",
                })
                continue

            # Keypoints are already in AMB3R resolution (no scaling needed)
            # because pose was run on AMB3R's stored images
            print(f"  Frame {frame_idx}, Person {person['person_index']}:")
            for name in ["left_hip", "right_hip", "left_knee", "right_knee",
                         "left_ankle", "right_ankle"]:
                kp = lk[name]
                print(f"    {name}: ({kp['x']:.1f}, {kp['y']:.1f}) score={kp['score']:.2f}")

            # Lookup 3D positions directly (no coordinate scaling)
            joints_3d = lookup_3d_from_pointmap(
                lk, pts_per_frame, imgs_per_frame, frame_idx
            )

            # Check we got all joints
            required = ["left_hip", "right_hip", "left_knee", "right_knee",
                        "left_ankle", "right_ankle"]
            if any(joints_3d.get(n) is None for n in required):
                missing = [n for n in required if joints_3d.get(n) is None]
                frame_assessments.append({
                    "person_index": person["person_index"],
                    "error": f"3D lookup failed for: {missing}",
                })
                continue

            # Print 3D positions for debug
            for name in required:
                p = joints_3d[name]
                if p:
                    print(f"    {name} -> 3D: ({p[0]*100:.1f}, {p[1]*100:.1f}, {p[2]*100:.1f}) cm")

            # Compute 3D assessment
            assessment = compute_3d_assessment(joints_3d, method="direct_lookup")
            frame_assessments.append({
                "person_index": person["person_index"],
                "assessment": asdict(assessment),
            })

        all_assessments[img_name] = frame_assessments

    with open(output_path, "w") as f:
        json.dump(all_assessments, f, indent=2)
    print(f"Saved 3D clinical measurements: {output_path}")
    return output_path


def compute_3d_assessment(joints_3d, method="direct_lookup"):
    """Compute full 3D assessment from joint positions.

    AMB3R's metric scale head is unreliable (measured 40-67% of true size).
    Therefore we ONLY compute scale-invariant measurements:
      - RELIABLE: HKA angles, deviation, classification, severity,
                  femur/tibia ratio, leg symmetry (L vs R ratio)
      - UNRELIABLE (kept as raw but marked): absolute lengths, gaps, height
        These use AMB3R's raw scale which is ~2x too small.

    Args:
        joints_3d: Dict of joint_name -> [x, y, z] (AMB3R raw units)
        method: "direct_lookup" or "smpl_aligned"
    """
    assessment = Assessment3D(method=method)
    assessment.units = "amb3r_raw (NOT metric — scale is unreliable, ~40-67% of true)"

    l_hip = joints_3d["left_hip"]
    r_hip = joints_3d["right_hip"]
    l_knee = joints_3d["left_knee"]
    r_knee = joints_3d["right_knee"]
    l_ankle = joints_3d["left_ankle"]
    r_ankle = joints_3d["right_ankle"]

    # ── RELIABLE: Angles (scale-invariant) ──────────────────────────
    # Left leg
    left = Leg3D(side="left", method=method)
    left.hip_3d = l_hip
    left.knee_3d = l_knee
    left.ankle_3d = l_ankle
    left.hka_angle_3d = angle_3d(l_hip, l_knee, l_ankle)
    left.hka_deviation_3d, left.mad_3d = signed_deviation_3d(l_hip, l_knee, l_ankle, "left")
    left.classification, left.severity = classify_deformity(left.hka_deviation_3d)

    # Raw lengths (kept for ratio computation, NOT for absolute measurement)
    left.femur_length_3d = distance_3d(l_hip, l_knee)
    left.tibia_length_3d = distance_3d(l_knee, l_ankle)
    left.total_leg_length_3d = left.femur_length_3d + left.tibia_length_3d
    assessment.left_leg = left

    # Right leg
    right = Leg3D(side="right", method=method)
    right.hip_3d = r_hip
    right.knee_3d = r_knee
    right.ankle_3d = r_ankle
    right.hka_angle_3d = angle_3d(r_hip, r_knee, r_ankle)
    right.hka_deviation_3d, right.mad_3d = signed_deviation_3d(r_hip, r_knee, r_ankle, "right")
    right.classification, right.severity = classify_deformity(right.hka_deviation_3d)

    right.femur_length_3d = distance_3d(r_hip, r_knee)
    right.tibia_length_3d = distance_3d(r_knee, r_ankle)
    right.total_leg_length_3d = right.femur_length_3d + right.tibia_length_3d
    assessment.right_leg = right

    # ── RELIABLE: Ratios (scale-invariant) ──────────────────────────
    # Femur/tibia ratio per leg
    left_ft_ratio = left.femur_length_3d / (left.tibia_length_3d + 1e-8)
    right_ft_ratio = right.femur_length_3d / (right.tibia_length_3d + 1e-8)

    # Leg symmetry ratio (left/right total length)
    leg_symmetry = left.total_leg_length_3d / (right.total_leg_length_3d + 1e-8)

    # Which leg is shorter (reliable — relative comparison)
    if left.total_leg_length_3d < right.total_leg_length_3d:
        assessment.leg_length_discrepancy_side = "left"
    elif right.total_leg_length_3d < left.total_leg_length_3d:
        assessment.leg_length_discrepancy_side = "right"
    else:
        assessment.leg_length_discrepancy_side = "equal"

    # Leg length difference as percentage (scale-invariant)
    avg_leg = (left.total_leg_length_3d + right.total_leg_length_3d) / 2
    assessment.leg_length_difference_3d = abs(left.total_leg_length_3d - right.total_leg_length_3d)
    leg_diff_pct = (assessment.leg_length_difference_3d / avg_leg * 100) if avg_leg > 0.01 else 0

    # ── UNRELIABLE: Absolute distances (AMB3R scale is wrong) ───────
    # Kept in output but clearly marked. Do NOT use for clinical decisions.
    assessment.intercondylar_distance_3d = distance_3d(l_knee, r_knee)
    assessment.intermalleolar_distance_3d = distance_3d(l_ankle, r_ankle)

    # ── Flags (only angle-based and ratio-based) ────────────────────
    flags = []
    for leg in [left, right]:
        if leg.hka_deviation_3d is not None and abs(leg.hka_deviation_3d) > 3.0:
            flags.append(
                f"{leg.side.capitalize()} leg: {leg.classification} "
                f"({leg.severity}, {abs(leg.hka_deviation_3d):.1f}° deviation)"
            )
    if leg_diff_pct > 2.0:
        flags.append(
            f"Limb length asymmetry: {leg_diff_pct:.1f}% "
            f"({assessment.leg_length_discrepancy_side} shorter)"
        )
    assessment.flags = flags

    # Overall classification
    l_cls = left.classification
    r_cls = right.classification
    if l_cls == "normal" and r_cls == "normal":
        assessment.overall_classification = "Normal alignment"
    elif l_cls == r_cls:
        assessment.overall_classification = f"Bilateral {l_cls}"
    elif l_cls == "normal":
        assessment.overall_classification = f"Right {r_cls}"
    elif r_cls == "normal":
        assessment.overall_classification = f"Left {l_cls}"
    else:
        assessment.overall_classification = f"Mixed: Left {l_cls}, Right {r_cls}"

    assessment.ratios = {
        "left_femur_tibia_ratio": round(left_ft_ratio, 3),
        "right_femur_tibia_ratio": round(right_ft_ratio, 3),
        "leg_symmetry_ratio": round(leg_symmetry, 3),
        "leg_length_difference_pct": round(leg_diff_pct, 1),
        "scale_warning": "Absolute lengths are in AMB3R raw units (~40-67% of true metric). Use angles and ratios only.",
    }

    return assessment


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="3D Clinical Measurements (leg-only)")
    parser.add_argument("--pose_results", required=True)
    parser.add_argument("--amb3r_npz", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    measure_from_pointmap(args.pose_results, args.amb3r_npz, args.output)
