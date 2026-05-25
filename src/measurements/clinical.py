"""
Clinical measurements for leg deformity assessment.

Computes HKA angle, mechanical axis deviation, intercondylar/intermalleolar
distances, leg lengths, and other measurements from 2D pose keypoints
and optionally from 3D point cloud data.

Keypoint convention (COCO 17):
    11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee
    15: left_ankle, 16: right_ankle
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class LegMeasurements:
    """Clinical measurements for one leg."""

    side: str  # "left" or "right"

    # Joint positions (2D pixel or 3D metric)
    hip: Optional[list] = None
    knee: Optional[list] = None
    ankle: Optional[list] = None

    # Angles (degrees)
    hka_angle: Optional[float] = None  # Hip-Knee-Ankle angle (180 = straight)
    hka_deviation: Optional[float] = None  # Deviation from 180 (+ = varus, - = valgus)

    # Lengths
    femur_length: Optional[float] = None  # Hip to knee distance
    tibia_length: Optional[float] = None  # Knee to ankle distance
    total_leg_length: Optional[float] = None  # Femur + tibia

    # Classification
    classification: Optional[str] = None  # "normal", "varus", "valgus"
    severity: Optional[str] = None  # "mild", "moderate", "severe"

    # Confidence
    joint_confidence: Optional[dict] = None


@dataclass
class FullAssessment:
    """Complete leg deformity assessment."""

    left_leg: Optional[LegMeasurements] = None
    right_leg: Optional[LegMeasurements] = None

    # Inter-leg measurements
    intercondylar_distance: Optional[float] = None  # Knee gap (pixel or metric)
    intermalleolar_distance: Optional[float] = None  # Ankle gap (pixel or metric)

    # Limb length discrepancy
    leg_length_difference: Optional[float] = None
    leg_length_discrepancy_side: Optional[str] = None  # Which leg is shorter

    # Recurvatum (requires lateral view)
    left_recurvatum_angle: Optional[float] = None
    right_recurvatum_angle: Optional[float] = None

    # Overall classification
    overall_classification: Optional[str] = None
    flags: list = field(default_factory=list)

    # Scale info
    units: str = "pixels"  # "pixels" or "cm"
    scale_factor: Optional[float] = None  # cm per pixel (if calibrated)


def compute_angle(p1, p2, p3):
    """Compute the angle at p2 formed by vectors p1->p2 and p3->p2.

    Returns angle in degrees. 180 = straight line.

    Args:
        p1, p2, p3: Points as (x, y) or (x, y, z) arrays/lists.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)

    v1 = p1 - p2  # Vector from knee to hip
    v2 = p3 - p2  # Vector from knee to ankle

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def compute_distance(p1, p2):
    """Euclidean distance between two points."""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    return float(np.linalg.norm(p2 - p1))


def signed_hka_deviation(hip, knee, ankle, side="left"):
    """Compute signed HKA deviation for varus/valgus classification.

    Positive = varus (knee lateral to mechanical axis, bow-legged)
    Negative = valgus (knee medial to mechanical axis, knock-kneed)

    Assumes standard AP (anterior-posterior) view with patient facing camera:
    - Patient's LEFT leg appears on the RIGHT side of the image
    - Patient's RIGHT leg appears on the LEFT side of the image

    Uses the 2D cross product to determine which side of the hip-ankle
    line the knee falls on. In image coordinates (y increases downward):
    - cross > 0: knee is to the LEFT of the hip→ankle vector
    - cross < 0: knee is to the RIGHT of the hip→ankle vector

    For LEFT leg:  knee LEFT of line (cross > 0) = medial = valgus (-)
                   knee RIGHT of line (cross < 0) = lateral = varus (+)
    For RIGHT leg: knee LEFT of line (cross > 0) = lateral = varus (+)
                   knee RIGHT of line (cross < 0) = medial = valgus (-)

    Args:
        hip, knee, ankle: 2D (x, y) coordinates.
        side: "left" or "right" leg.

    Returns:
        (deviation, mad): Signed deviation in degrees, mechanical axis deviation distance.
    """
    hip = np.array(hip[:2], dtype=float)
    knee = np.array(knee[:2], dtype=float)
    ankle = np.array(ankle[:2], dtype=float)

    # Cross product to determine which side of the line the knee is on
    mech_axis = ankle - hip
    to_knee = knee - hip
    cross = mech_axis[0] * to_knee[1] - mech_axis[1] * to_knee[0]

    # Distance from knee to mechanical axis line (MAD)
    line_len = np.linalg.norm(mech_axis) + 1e-8
    mad = abs(cross) / line_len

    # HKA angle (always in [0, 180] from acos)
    hka = compute_angle(hip, knee, ankle)
    abs_deviation = 180.0 - hka

    # Apply sign based on side and cross product direction
    if side == "left":
        # Left leg: cross > 0 (knee left of line) = medial = valgus (negative)
        deviation = abs_deviation if cross < 0 else -abs_deviation
    else:
        # Right leg: cross > 0 (knee left of line) = lateral = varus (positive)
        deviation = abs_deviation if cross > 0 else -abs_deviation

    return deviation, mad


def classify_deformity(deviation_deg):
    """Classify leg deformity based on HKA deviation.

    Args:
        deviation_deg: HKA deviation in degrees.
            Positive = varus, Negative = valgus.

    Returns:
        (classification, severity) tuple.
    """
    abs_dev = abs(deviation_deg)

    if abs_dev <= 3.0:
        return "normal", "none"

    if deviation_deg > 0:
        classification = "varus"
    else:
        classification = "valgus"

    if abs_dev <= 5.0:
        severity = "mild"
    elif abs_dev <= 10.0:
        severity = "moderate"
    else:
        severity = "severe"

    return classification, severity


def measure_single_leg(hip, knee, ankle, side, confidence=None):
    """Compute all measurements for a single leg.

    Args:
        hip, knee, ankle: 2D (x, y) or 3D (x, y, z) coordinates.
        side: "left" or "right".
        confidence: Optional dict with joint confidence scores.

    Returns:
        LegMeasurements dataclass.
    """
    measurements = LegMeasurements(side=side)
    measurements.hip = list(hip)
    measurements.knee = list(knee)
    measurements.ankle = list(ankle)
    measurements.joint_confidence = confidence

    # HKA angle
    measurements.hka_angle = compute_angle(hip, knee, ankle)
    deviation, _ = signed_hka_deviation(hip, knee, ankle, side=side)
    measurements.hka_deviation = deviation

    # Lengths
    measurements.femur_length = compute_distance(hip, knee)
    measurements.tibia_length = compute_distance(knee, ankle)
    measurements.total_leg_length = measurements.femur_length + measurements.tibia_length

    # Classification
    measurements.classification, measurements.severity = classify_deformity(deviation)

    return measurements


def measure_full_assessment(pose_data, image_name=None, scale_factor=None):
    """Compute full leg deformity assessment from pose detection results.

    Args:
        pose_data: Dict from pose_results.json for one image, one person.
            Expected keys: 'leg_keypoints' with 'left_hip', 'right_hip', etc.
        image_name: Optional image filename for context.
        scale_factor: Optional cm-per-pixel scale factor for metric conversion.

    Returns:
        FullAssessment dataclass.
    """
    assessment = FullAssessment()

    if scale_factor:
        assessment.units = "cm"
        assessment.scale_factor = scale_factor

    lk = pose_data["leg_keypoints"]

    # Extract 2D positions
    l_hip = [lk["left_hip"]["x"], lk["left_hip"]["y"]]
    r_hip = [lk["right_hip"]["x"], lk["right_hip"]["y"]]
    l_knee = [lk["left_knee"]["x"], lk["left_knee"]["y"]]
    r_knee = [lk["right_knee"]["x"], lk["right_knee"]["y"]]
    l_ankle = [lk["left_ankle"]["x"], lk["left_ankle"]["y"]]
    r_ankle = [lk["right_ankle"]["x"], lk["right_ankle"]["y"]]

    # Confidence scores
    l_conf = {
        "hip": lk["left_hip"]["score"],
        "knee": lk["left_knee"]["score"],
        "ankle": lk["left_ankle"]["score"],
    }
    r_conf = {
        "hip": lk["right_hip"]["score"],
        "knee": lk["right_knee"]["score"],
        "ankle": lk["right_ankle"]["score"],
    }

    # Single leg measurements
    assessment.left_leg = measure_single_leg(l_hip, l_knee, l_ankle, "left", l_conf)
    assessment.right_leg = measure_single_leg(r_hip, r_knee, r_ankle, "right", r_conf)

    # Inter-leg measurements
    assessment.intercondylar_distance = compute_distance(l_knee, r_knee)
    assessment.intermalleolar_distance = compute_distance(l_ankle, r_ankle)

    # Leg length discrepancy
    l_len = assessment.left_leg.total_leg_length
    r_len = assessment.right_leg.total_leg_length
    assessment.leg_length_difference = abs(l_len - r_len)
    if l_len < r_len:
        assessment.leg_length_discrepancy_side = "left"
    elif r_len < l_len:
        assessment.leg_length_discrepancy_side = "right"
    else:
        assessment.leg_length_discrepancy_side = "equal"

    # Apply metric scale if available
    if scale_factor:
        _apply_scale(assessment, scale_factor)

    # Generate flags
    assessment.flags = _generate_flags(assessment)

    # Overall classification
    assessment.overall_classification = _overall_classification(assessment)

    return assessment


def _apply_scale(assessment, scale_factor):
    """Convert pixel measurements to cm using scale factor."""
    for leg in [assessment.left_leg, assessment.right_leg]:
        if leg:
            if leg.femur_length is not None:
                leg.femur_length *= scale_factor
            if leg.tibia_length is not None:
                leg.tibia_length *= scale_factor
            if leg.total_leg_length is not None:
                leg.total_leg_length *= scale_factor

    if assessment.intercondylar_distance is not None:
        assessment.intercondylar_distance *= scale_factor
    if assessment.intermalleolar_distance is not None:
        assessment.intermalleolar_distance *= scale_factor
    if assessment.leg_length_difference is not None:
        assessment.leg_length_difference *= scale_factor


def _generate_flags(assessment):
    """Generate clinical flags based on measurements."""
    flags = []

    for leg in [assessment.left_leg, assessment.right_leg]:
        if leg and leg.hka_deviation is not None:
            abs_dev = abs(leg.hka_deviation)
            if abs_dev > 3.0:
                flags.append(
                    f"{leg.side.capitalize()} leg: {leg.classification} "
                    f"({leg.severity}, {abs_dev:.1f}° deviation)"
                )

    if assessment.leg_length_difference is not None:
        threshold = 1.0 if assessment.units == "cm" else 10.0  # 1cm or 10px
        if assessment.leg_length_difference > threshold:
            flags.append(
                f"Limb length discrepancy: {assessment.leg_length_difference:.1f}"
                f"{assessment.units} ({assessment.leg_length_discrepancy_side} shorter)"
            )

    return flags


def _overall_classification(assessment):
    """Determine overall classification."""
    l_class = assessment.left_leg.classification if assessment.left_leg else "unknown"
    r_class = assessment.right_leg.classification if assessment.right_leg else "unknown"

    if l_class == "normal" and r_class == "normal":
        return "Normal alignment"
    elif l_class == r_class:
        return f"Bilateral {l_class}"
    elif l_class == "normal":
        return f"Right {r_class}"
    elif r_class == "normal":
        return f"Left {l_class}"
    else:
        return f"Mixed: Left {l_class}, Right {r_class}"


def process_pose_results(pose_results_path, output_path, scale_factor=None,
                         per_image_scale=None):
    """Process pose results JSON and compute clinical measurements for all images.

    Args:
        pose_results_path: Path to pose_results.json from run_pose.py.
        output_path: Path to save clinical measurements JSON.
        scale_factor: Optional global cm-per-pixel scale factor (fallback).
        per_image_scale: Optional dict of image_name → cm-per-pixel scale.
            If provided, each image uses its own scale from ArUco detection.
            Images without a marker fall back to scale_factor (if set) or pixels.

    Returns:
        Path to output JSON.
    """
    with open(pose_results_path, "r") as f:
        pose_results = json.load(f)

    if per_image_scale is None:
        per_image_scale = {}

    all_assessments = {}

    for img_name, img_data in pose_results.items():
        # Use per-image scale if available, otherwise fall back to global
        img_scale = per_image_scale.get(img_name, scale_factor)

        img_assessments = []
        for person in img_data["persons"]:
            # Check if we have sufficient leg keypoints
            lk = person["leg_keypoints"]
            min_score = 0.3
            has_legs = all(
                lk[name]["score"] > min_score
                for name in [
                    "left_hip", "right_hip",
                    "left_knee", "right_knee",
                    "left_ankle", "right_ankle",
                ]
            )

            if not has_legs:
                img_assessments.append(
                    {
                        "person_index": person["person_index"],
                        "error": "Insufficient leg keypoint confidence",
                        "keypoint_scores": {
                            name: lk[name]["score"] for name in lk
                        },
                    }
                )
                continue

            assessment = measure_full_assessment(person, img_name, img_scale)
            img_assessments.append(
                {
                    "person_index": person["person_index"],
                    "assessment": asdict(assessment),
                }
            )

        all_assessments[img_name] = img_assessments

    with open(output_path, "w") as f:
        json.dump(all_assessments, f, indent=2)

    print(f"Saved clinical measurements: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute clinical measurements from pose results"
    )
    parser.add_argument(
        "--pose_results",
        type=str,
        required=True,
        help="Path to pose_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save clinical measurements JSON",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help="cm per pixel scale factor (from ArUco calibration)",
    )
    args = parser.parse_args()

    process_pose_results(args.pose_results, args.output, args.scale_factor)
