"""
Leg deformity metrics — clean module with robust classification.

Computes per-frame 3D measurements from VGGT/AMB3R point maps and pose
keypoints, aggregates them across frames via the median (robust to outliers),
and classifies the resulting values with explicit margin zones so noise
doesn't push a borderline case into the wrong category.

Metrics (per leg, per frame, then aggregated):
  - HKA angle (Hip-Knee-Ankle), 180° = mechanically straight
  - HKA deviation = 180° − HKA, signed (+ varus, − valgus)
  - Mechanical Axis Deviation (MAD), distance knee → hip-ankle line (mm)
  - Femur length (hip→knee), tibia length (knee→ankle), total (mm)
  - Femur / tibia ratio (scale-invariant)

Bilateral metrics:
  - Intercondylar distance (knee gap)
  - Intermalleolar distance (ankle gap)
  - Leg length discrepancy (absolute + percentage)

Classification with explicit margin zones:
              |dev|
    normal      [ 0,  5°]         ← physiologic alignment; absorbs ~2-3° of
                                     measurement noise so a true-normal patient
                                     isn't misclassified by pose/VGGT jitter
    borderline  ( 5,  7°]         ← transitional / ambiguous, flag for review
    mild        ( 7, 10°]
    moderate    (10, 15°]
    severe      (15°, ∞)

A measurement within 1° of a category boundary is also flagged "near_threshold".
A high-variance measurement (IQR > 3°) is flagged "low confidence".

Why 5° for normal? Anatomic literature (e.g. Cooke et al, Bellemans et al)
considers HKA 175°-185° (i.e. |dev| ≤ 5°) as physiologically normal — there
is natural variation around mechanically-straight alignment. The pipeline's
inherent measurement noise (per-frame IQR ~2-3°) means a 3° cutoff would
mark many anatomically-normal patients as borderline varus/valgus.

Per-frame measurements use pose confidence; frames where any joint score
< 0.30 are dropped from aggregation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
import math
import numpy as np


# ════════════════════════════════════════════════════════════════════
#  Classification thresholds — single source of truth
# ════════════════════════════════════════════════════════════════════

# All in degrees. |HKA deviation| from 180° (i.e. degrees off straight).
# Normal range is 5° wide on each side so noise in the per-frame measurements
# (typical IQR 2-3°) doesn't push a physiologically-normal patient into
# borderline varus/valgus. See module docstring for rationale.
NORMAL_MAX_DEG = 5.0
BORDERLINE_MAX_DEG = 7.0
MILD_MAX_DEG = 10.0
MODERATE_MAX_DEG = 15.0
# Anything above MODERATE_MAX_DEG is severe.

# A value within this many degrees of a threshold boundary is "near-threshold"
NEAR_THRESHOLD_MARGIN_DEG = 1.0

# IQR (inter-quartile range across frames) above which we don't trust the median
LOW_CONFIDENCE_IQR_DEG = 3.0

# Pose keypoint confidence cutoff for using a frame's measurement
MIN_KEYPOINT_SCORE = 0.30

# Implausibility check: |HKA dev| above this is biologically unusual and
# almost always indicates a measurement-level problem (loose clothing biasing
# pose, very oblique view, body rotation, etc.) rather than true anatomy.
# We still report the number but flag it for the user.
IMPLAUSIBLE_DEVIATION_DEG = 25.0

# LLD (leg length discrepancy) thresholds — as percentage of average leg
LLD_NORMAL_MAX_PCT = 1.5
LLD_MILD_MAX_PCT = 3.0
LLD_MODERATE_MAX_PCT = 5.0
# > LLD_MODERATE_MAX_PCT = severe

# Knee/ankle gap thresholds for genu varum / valgum classification.
# All in cm (only meaningful with metric calibration).
#
# Clinical reference (feet-together stance for varum, knees-together for valgum):
#   - normal: both gaps small
#   - genu varum (bow-legs): knees apart, ankles together → intercondylar large
#   - genu valgum (knock-knees): ankles apart, knees together → intermalleolar large
GAP_NORMAL_MAX_CM = 2.0       # ≤2cm = within normal range
GAP_MILD_MAX_CM = 5.0         # 2-5cm = mild
GAP_MODERATE_MAX_CM = 8.0     # 5-8cm = moderate; >8cm = severe


# ════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ════════════════════════════════════════════════════════════════════

def _angle_3d(p1, p2, p3) -> float:
    """Angle at p2 (degrees) formed by p1→p2 and p3→p2."""
    v1 = np.asarray(p1) - np.asarray(p2)
    v2 = np.asarray(p3) - np.asarray(p2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _distance_3d(p1, p2) -> float:
    return float(np.linalg.norm(np.asarray(p2) - np.asarray(p1)))


def _point_to_line_distance_3d(point, line_a, line_b) -> float:
    p = np.asarray(point)
    a = np.asarray(line_a)
    b = np.asarray(line_b)
    ab = b - a
    return float(np.linalg.norm(np.cross(ab, p - a)) / (np.linalg.norm(ab) + 1e-12))


def signed_hka_deviation_3d(hip, knee, ankle, side: str,
                              other_hip=None) -> Tuple[float, float]:
    """Compute signed HKA deviation + MAD in 3D.

    Sign convention:  + varus  (knee lateral to mechanical axis, bow-legged)
                      − valgus (knee medial  to mechanical axis, knock-kneed)

    Sign determination has TWO modes:

    1. ROBUST (when `other_hip` is provided — preferred):
       The lateral direction is defined by (this-side hip − other-side hip),
       which gives the patient's TRUE lateral direction regardless of how the
       camera is oriented. Then varus ⇔ knee deflection has a positive
       component along the lateral direction. Works for any viewing angle.

    2. IMAGE-PLANE FALLBACK (when `other_hip` is None):
       Assumes the patient is captured anteriorly (facing the camera, image
       Y points down). Uses the 2D cross product in the X-Y plane —
       equivalent to taking the Z-component of the 3D cross. This is what
       the OLD code SHOULD have done; it was reading `cross[1]` (Y component)
       which is ≈ 0 for legs in the image plane and produced noise-driven
       random signs.

    Args:
        hip, knee, ankle: 3D coordinates of this leg's joints.
        side: "left" or "right".
        other_hip: 3D coordinates of the OTHER leg's hip (recommended).
    """
    hip = np.asarray(hip, dtype=float)
    knee = np.asarray(knee, dtype=float)
    ankle = np.asarray(ankle, dtype=float)

    hka = _angle_3d(hip, knee, ankle)
    abs_dev = 180.0 - hka
    mad = _point_to_line_distance_3d(knee, hip, ankle)

    # ── Mode 1: robust hip-pair method ─────────────────────────────────
    if other_hip is not None:
        other_hip = np.asarray(other_hip, dtype=float)
        # "Lateral" for this leg = direction from the OTHER hip toward THIS hip,
        # i.e. the direction of outward-from-midline for this leg.
        lateral = hip - other_hip
        ln = np.linalg.norm(lateral)
        if ln < 1e-9:
            # Hips coincide (shouldn't happen) — fall through to image-plane mode
            pass
        else:
            lateral /= ln

            # Project knee onto mechanical axis line, get perpendicular deflection
            mech_axis = ankle - hip
            ma_norm = np.linalg.norm(mech_axis)
            if ma_norm < 1e-9:
                return float(abs_dev), float(mad)
            mech_unit = mech_axis / ma_norm
            to_knee = knee - hip
            along = float(to_knee @ mech_unit)
            knee_on_axis = hip + along * mech_unit
            deflection = knee - knee_on_axis   # perpendicular to mech_axis

            # Component of deflection along the patient's lateral direction:
            # positive = bowed outward (varus), negative = bowed inward (valgus).
            lateral_component = float(deflection @ lateral)

            # Sign convention is the SAME for both legs because `lateral` already
            # encodes the side-specific "outward" direction.
            deviation = abs_dev if lateral_component > 0 else -abs_dev
            return float(deviation), float(mad)

    # ── Mode 2: image-plane (X-Y) fallback ────────────────────────────
    # In VGGT/AMB3R world coords with anterior view: X = image right,
    # Y = image down. The leg lies (approximately) in this X-Y plane,
    # so we use the 2D cross product there.
    mech_2d = (ankle - hip)[:2]
    knee_2d = (knee - hip)[:2]
    cross_z = float(mech_2d[0] * knee_2d[1] - mech_2d[1] * knee_2d[0])

    # In image-Y-down coords:
    #   LEFT leg:  varus ⇔ knee at +X (patient's left side, image right) ⇔ cross_z < 0
    #   RIGHT leg: varus ⇔ knee at −X (patient's right side, image left) ⇔ cross_z > 0
    if side == "left":
        deviation = abs_dev if cross_z < 0 else -abs_dev
    else:
        deviation = abs_dev if cross_z > 0 else -abs_dev
    return float(deviation), float(mad)


# ════════════════════════════════════════════════════════════════════
#  Classification with margins
# ════════════════════════════════════════════════════════════════════

# Measurement-uncertainty stand-in: even for a clean anterior frame, pose
# keypoint placement has ~1.5–2° of intrinsic noise. We use a Gaussian
# centred on the measured deviation with σ = MEASUREMENT_SIGMA_DEG to
# compute SOFT class probabilities — no hard boundary, smooth transition.
MEASUREMENT_SIGMA_DEG = 2.0


def _gaussian_cdf(x: float, mu: float, sigma: float) -> float:
    """Standard-normal CDF — no scipy dependency."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def soft_class_probabilities(deviation_deg: float,
                                sigma_deg: float = MEASUREMENT_SIGMA_DEG,
                                ) -> dict:
    """Compute SOFT probabilities over the classification bands.

    Models the measurement as a Gaussian: N(measured, σ²). Each class
    band's probability is the Gaussian's CDF-difference over its interval.
    No hard cut-offs — a measurement near a boundary gets meaningful
    probability mass on both sides.

    Returns a dict like:
        {
          "normal":            0.55,
          "varus_borderline":  0.30,
          "varus_mild":        0.12,
          "varus_moderate":    0.03,
          ...
        }
    Probabilities sum to 1.
    """
    mu = float(deviation_deg)
    sigma = float(sigma_deg)
    # Buckets in signed-deviation space — varus on the +side, valgus on −side.
    bands = [
        (-1e9, -MODERATE_MAX_DEG,   "valgus_severe"),
        (-MODERATE_MAX_DEG, -MILD_MAX_DEG, "valgus_moderate"),
        (-MILD_MAX_DEG, -BORDERLINE_MAX_DEG, "valgus_mild"),
        (-BORDERLINE_MAX_DEG, -NORMAL_MAX_DEG, "valgus_borderline"),
        (-NORMAL_MAX_DEG, NORMAL_MAX_DEG, "normal"),
        (NORMAL_MAX_DEG, BORDERLINE_MAX_DEG, "varus_borderline"),
        (BORDERLINE_MAX_DEG, MILD_MAX_DEG, "varus_mild"),
        (MILD_MAX_DEG, MODERATE_MAX_DEG, "varus_moderate"),
        (MODERATE_MAX_DEG, 1e9, "varus_severe"),
    ]
    probs = {}
    for lo, hi, name in bands:
        p = _gaussian_cdf(hi, mu, sigma) - _gaussian_cdf(lo, mu, sigma)
        if p > 1e-4:
            probs[name] = float(p)
    # Renormalise (handles edge-tails truncation rounding)
    s = sum(probs.values()) or 1.0
    return {k: v / s for k, v in probs.items()}


def soft_classify_hka(deviation_deg: float,
                        sigma_deg: float = MEASUREMENT_SIGMA_DEG,
                        ) -> Tuple[str, str, dict, str]:
    """Soft classification that returns:
        (best_class, best_severity, full_probabilities, summary_note)

    `best_class` ∈ {normal, varus, valgus} is the most-likely macro class.
    `best_severity` is the most-likely severity within that macro class.
    """
    probs = soft_class_probabilities(deviation_deg, sigma_deg)
    macro = {"normal": 0.0, "varus": 0.0, "valgus": 0.0}
    for k, p in probs.items():
        if k == "normal":
            macro["normal"] += p
        elif k.startswith("varus_"):
            macro["varus"] += p
        elif k.startswith("valgus_"):
            macro["valgus"] += p
    best_class = max(macro, key=macro.get)

    # Severity = highest-prob bucket within the best macro class
    if best_class == "normal":
        best_sev = "none"
    else:
        sub = {k: p for k, p in probs.items() if k.startswith(best_class + "_")}
        sub_best = max(sub, key=sub.get) if sub else best_class + "_borderline"
        best_sev = sub_best.split("_", 1)[1]

    # Confidence note based on top-class probability
    p_best = macro[best_class]
    if p_best >= 0.80:
        note = (f"dev = {deviation_deg:+.2f}° → {best_class} "
                f"(P = {p_best * 100:.0f}%), confident.")
    elif p_best >= 0.55:
        # mention the runner-up
        runner_up = sorted(macro.items(), key=lambda kv: -kv[1])[1]
        note = (f"dev = {deviation_deg:+.2f}° → {best_class} "
                f"(P = {p_best * 100:.0f}%), but {runner_up[0]} also possible "
                f"(P = {runner_up[1] * 100:.0f}%).")
    else:
        # genuinely uncertain — report top 2-3 buckets
        top_sorted = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        spread = "; ".join(f"{k}: {v * 100:.0f}%" for k, v in top_sorted)
        note = (f"dev = {deviation_deg:+.2f}° — UNCERTAIN. Top buckets: "
                f"{spread}.")
    return best_class, best_sev, probs, note


# ════════════════════════════════════════════════════════════════════
#  Single-frame 2D HKA (clinically equivalent to standing radiograph)
# ════════════════════════════════════════════════════════════════════
#
# When the user picks a clean anterior frame, the 2D image-plane HKA is
# clinically valid — this is how standing-radiograph HKA is measured in
# orthopaedic practice. We compute it directly from the 2D pose keypoints
# (no 3D depth ambiguity), which is far more reliable than averaging noisy
# multi-frame 3D measurements.

def _angle_2d(p1, p2, p3) -> float:
    """Angle at p2 (degrees) formed by p1→p2 and p3→p2, in 2D."""
    v1 = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    v2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def signed_hka_deviation_2d(hip_xy, knee_xy, ankle_xy, side: str,
                              other_hip_xy=None) -> Tuple[float, float]:
    """Signed HKA deviation from 2D image-plane keypoints.

    Sign convention (matches the 3D version):
        + varus  (knee lateral to hip-ankle line)
        − valgus (knee medial  to hip-ankle line)

    Uses the OTHER hip's location to define the patient's lateral
    direction so the sign is view-independent within the image plane.
    Image Y is "down" (pixel convention).
    """
    hip = np.asarray(hip_xy, dtype=float)
    knee = np.asarray(knee_xy, dtype=float)
    ank = np.asarray(ankle_xy, dtype=float)

    hka = _angle_2d(hip, knee, ank)
    abs_dev = 180.0 - hka

    # Distance from knee to the hip-ankle line (mechanical-axis deflection)
    ab = ank - hip
    mad = abs(ab[0] * (knee[1] - hip[1]) - ab[1] * (knee[0] - hip[0])) / (
        np.linalg.norm(ab) + 1e-9
    )

    # Sign — use patient's lateral direction if available
    if other_hip_xy is not None:
        other_hip = np.asarray(other_hip_xy, dtype=float)
        lateral = hip - other_hip
        lat_n = np.linalg.norm(lateral)
        if lat_n > 1e-6:
            lateral /= lat_n
            mech = ank - hip
            mech_n = np.linalg.norm(mech)
            if mech_n < 1e-6:
                return float(abs_dev), float(mad)
            mech_u = mech / mech_n
            to_knee = knee - hip
            along = float(to_knee @ mech_u)
            knee_on_axis = hip + along * mech_u
            deflection = knee - knee_on_axis
            lat_component = float(deflection @ lateral)
            dev = abs_dev if lat_component > 0 else -abs_dev
            return float(dev), float(mad)

    # Image-plane fallback (Y-down): same convention as 3D version
    cross_z = float((ank - hip)[0] * (knee - hip)[1]
                     - (ank - hip)[1] * (knee - hip)[0])
    if side == "left":
        dev = abs_dev if cross_z < 0 else -abs_dev
    else:
        dev = abs_dev if cross_z > 0 else -abs_dev
    return float(dev), float(mad)


@dataclass
class AnteriorFrameAssessment:
    """2D HKA assessment from a single user-selected anterior frame."""
    frame_name: str
    frame_idx: Optional[int]

    # Per-leg 2D keypoints (in the same image's pixel coords)
    left_hip_xy: List[float] = field(default_factory=list)
    left_knee_xy: List[float] = field(default_factory=list)
    left_ankle_xy: List[float] = field(default_factory=list)
    right_hip_xy: List[float] = field(default_factory=list)
    right_knee_xy: List[float] = field(default_factory=list)
    right_ankle_xy: List[float] = field(default_factory=list)

    # Per-leg HKA
    left_hka_deg: Optional[float] = None
    left_hka_deviation_deg: Optional[float] = None
    left_mad_px: Optional[float] = None
    left_classification: Optional[str] = None
    left_severity: Optional[str] = None
    left_class_probabilities: Optional[dict] = None
    left_note: Optional[str] = None

    right_hka_deg: Optional[float] = None
    right_hka_deviation_deg: Optional[float] = None
    right_mad_px: Optional[float] = None
    right_classification: Optional[str] = None
    right_severity: Optional[str] = None
    right_class_probabilities: Optional[dict] = None
    right_note: Optional[str] = None

    # View-quality cross-check (hip X-sep on the chosen frame)
    hip_sep_ratio: Optional[float] = None
    view_quality_label: Optional[str] = None
    view_warning: Optional[str] = None

    # Pose-keypoint min confidence on the chosen frame
    min_keypoint_score: Optional[float] = None

    # Stance-symmetry diagnostics: if one leg's pixel-length is much shorter
    # than the other (e.g. patient stood with one leg back), HKA values from
    # the shorter leg are unreliable.
    left_leg_length_px: Optional[float] = None
    right_leg_length_px: Optional[float] = None
    leg_length_asymmetry_pct: Optional[float] = None
    stance_symmetry_warning: Optional[str] = None

    overall_assessment: Optional[str] = None


def measure_anterior_frame_2d(pose_frame_data: dict,
                                frame_name: str,
                                frame_idx: Optional[int] = None,
                                ) -> AnteriorFrameAssessment:
    """Single-frame 2D HKA + soft classification from one pose result.

    Args:
        pose_frame_data: dict like pose_results[<frame_name>] — must
            contain 'persons' with 'leg_keypoints'.
        frame_name: filename of the chosen frame.
        frame_idx: optional T-axis index.
    """
    out = AnteriorFrameAssessment(frame_name=frame_name, frame_idx=frame_idx)
    persons = pose_frame_data.get("persons", []) if pose_frame_data else []
    if not persons:
        out.view_warning = "No person detected in the chosen frame."
        return out
    person = max(persons, key=lambda p: p.get("mean_score", 0))
    lk = person.get("leg_keypoints", {})
    needed = ("left_hip", "right_hip", "left_knee", "right_knee",
              "left_ankle", "right_ankle")
    if not all(k in lk for k in needed):
        out.view_warning = "Pose detector missed a required leg keypoint."
        return out

    out.min_keypoint_score = float(min(lk[k]["score"] for k in needed))

    lh = (lk["left_hip"]["x"],  lk["left_hip"]["y"])
    rh = (lk["right_hip"]["x"], lk["right_hip"]["y"])
    lkn = (lk["left_knee"]["x"], lk["left_knee"]["y"])
    rkn = (lk["right_knee"]["x"], lk["right_knee"]["y"])
    la = (lk["left_ankle"]["x"], lk["left_ankle"]["y"])
    ra = (lk["right_ankle"]["x"], lk["right_ankle"]["y"])

    out.left_hip_xy = list(lh); out.right_hip_xy = list(rh)
    out.left_knee_xy = list(lkn); out.right_knee_xy = list(rkn)
    out.left_ankle_xy = list(la); out.right_ankle_xy = list(ra)

    # Per-leg pixel length (hip→ankle) — used for stance-symmetry check.
    # If one leg is much shorter in pixels, the patient is angled and the
    # HKA from that leg's projection is unreliable.
    l_len = float(np.hypot(la[0] - lh[0], la[1] - lh[1]))
    r_len = float(np.hypot(ra[0] - rh[0], ra[1] - rh[1]))
    out.left_leg_length_px = l_len
    out.right_leg_length_px = r_len
    if l_len > 1 and r_len > 1:
        avg = 0.5 * (l_len + r_len)
        asym = abs(l_len - r_len) / avg * 100.0
        out.leg_length_asymmetry_pct = float(asym)
        if asym > 15.0:
            out.stance_symmetry_warning = (
                f"Leg-length asymmetry on this frame is {asym:.0f}% — "
                f"the patient may be standing with one foot back. HKA "
                f"from the shorter leg has reduced reliability."
            )

    # View-quality sanity check on the chosen frame
    leg_h = max(l_len, r_len) or 1.0
    out.hip_sep_ratio = float(abs(lh[0] - rh[0]) / leg_h)
    if out.hip_sep_ratio >= ANTERIOR_VIEW_CLEAN_HIP_SEP:
        out.view_quality_label = "anterior"
    elif out.hip_sep_ratio >= ANTERIOR_VIEW_MIN_HIP_SEP:
        out.view_quality_label = "near_anterior"
        out.view_warning = (
            f"Hip X-separation {out.hip_sep_ratio:.2f} is below the clean "
            f"anterior threshold ({ANTERIOR_VIEW_CLEAN_HIP_SEP}). Measurement "
            f"still usable; classification has wider uncertainty band."
        )
    else:
        out.view_quality_label = "oblique"
        out.view_warning = (
            f"Hip X-separation {out.hip_sep_ratio:.2f} is below the anterior "
            f"threshold ({ANTERIOR_VIEW_MIN_HIP_SEP}) — this frame is too "
            f"oblique. Re-pick a frame closer to direct anterior view."
        )

    # If the view is sketchy, broaden the soft-classification sigma to
    # reflect the larger measurement uncertainty.
    sigma = MEASUREMENT_SIGMA_DEG
    if out.view_quality_label == "near_anterior":
        sigma *= 1.6
    elif out.view_quality_label == "oblique":
        sigma *= 2.4
    # Stance asymmetry adds more uncertainty (patient angled / one leg back)
    if out.leg_length_asymmetry_pct is not None:
        if out.leg_length_asymmetry_pct > 15:
            sigma *= 1.5
        if out.leg_length_asymmetry_pct > 25:
            sigma *= 1.5    # cumulative 2.25× for >25%

    # Per-leg HKA + soft classification
    for side, hip, knee, ank, other_hip in [
        ("left", lh, lkn, la, rh),
        ("right", rh, rkn, ra, lh),
    ]:
        hka = _angle_2d(hip, knee, ank)
        dev, mad = signed_hka_deviation_2d(hip, knee, ank, side,
                                              other_hip_xy=other_hip)
        cls, sev, probs, note = soft_classify_hka(dev, sigma_deg=sigma)
        if side == "left":
            out.left_hka_deg = hka
            out.left_hka_deviation_deg = dev
            out.left_mad_px = mad
            out.left_classification = cls
            out.left_severity = sev
            out.left_class_probabilities = probs
            out.left_note = note
        else:
            out.right_hka_deg = hka
            out.right_hka_deviation_deg = dev
            out.right_mad_px = mad
            out.right_classification = cls
            out.right_severity = sev
            out.right_class_probabilities = probs
            out.right_note = note

    # Overall summary
    L = out.left_classification or "—"; R = out.right_classification or "—"
    if L == "normal" and R == "normal":
        out.overall_assessment = "Bilateral normal alignment"
    elif L == R and L != "normal":
        out.overall_assessment = f"Bilateral {L}"
    elif L == "normal":
        out.overall_assessment = f"Right-side {R}"
    elif R == "normal":
        out.overall_assessment = f"Left-side {L}"
    else:
        out.overall_assessment = f"Mixed: left {L}, right {R}"
    return out


def classify_hka_deviation(deviation_deg: float, iqr_deg: Optional[float] = None
                            ) -> Tuple[str, str, str, str]:
    """Classify an HKA deviation with explicit margin zones.

    Args:
        deviation_deg: signed deviation (+ varus, − valgus).
        iqr_deg: optional inter-quartile range across frames; high IQR
            lowers confidence.

    Returns:
        (classification, severity, confidence, note)
            classification ∈ {"normal", "varus", "valgus"}
            severity       ∈ {"none", "borderline", "mild", "moderate", "severe"}
            confidence     ∈ {"high", "near_threshold", "low_variance"}
            note           is a human-readable comment summarising the call
    """
    abs_dev = abs(float(deviation_deg))

    # ── classification + severity ────────────────────────────────────
    if abs_dev <= NORMAL_MAX_DEG:
        cls, sev = "normal", "none"
        # Sub-classify the normal band for transparency:
        #   "centered"   = clearly mechanically straight (|dev| ≤ 2°)
        #   "physiologic" = within physiologic-normal range (2° < |dev| ≤ 5°)
        normal_tendency = (
            "centered" if abs_dev <= 2.0
            else ("varus_tendency" if deviation_deg > 0 else "valgus_tendency")
        )
    else:
        cls = "varus" if deviation_deg > 0 else "valgus"
        normal_tendency = None
        if abs_dev <= BORDERLINE_MAX_DEG:
            sev = "borderline"
        elif abs_dev <= MILD_MAX_DEG:
            sev = "mild"
        elif abs_dev <= MODERATE_MAX_DEG:
            sev = "moderate"
        else:
            sev = "severe"

    # ── confidence ───────────────────────────────────────────────────
    boundaries = [NORMAL_MAX_DEG, BORDERLINE_MAX_DEG, MILD_MAX_DEG, MODERATE_MAX_DEG]
    near_threshold = min(abs(abs_dev - b) for b in boundaries) < NEAR_THRESHOLD_MARGIN_DEG

    if iqr_deg is not None and iqr_deg > LOW_CONFIDENCE_IQR_DEG:
        confidence = "low_variance"
        note = (f"IQR across frames = {iqr_deg:.1f}° (> {LOW_CONFIDENCE_IQR_DEG}°). "
                f"Aggregated median is unreliable — capture more frames or check "
                f"pose-detection quality.")
    elif near_threshold:
        confidence = "near_threshold"
        note = (f"|dev| = {abs_dev:.2f}° is within {NEAR_THRESHOLD_MARGIN_DEG}° "
                f"of a category boundary. Classification could shift with a small "
                f"measurement change.")
    elif cls == "normal":
        confidence = "high"
        if normal_tendency == "centered":
            note = (f"|dev| = {abs_dev:.2f}° — mechanically straight, "
                    f"comfortably inside the normal range (≤{NORMAL_MAX_DEG:.0f}°).")
        else:
            direction = "varus" if deviation_deg > 0 else "valgus"
            note = (f"|dev| = {abs_dev:.2f}° — within the physiologic-normal "
                    f"range (≤{NORMAL_MAX_DEG:.0f}°) with a mild {direction} "
                    f"tendency; no deformity.")
    else:
        confidence = "high"
        note = f"|dev| = {abs_dev:.2f}°, comfortably inside the '{sev}' band."

    return cls, sev, confidence, note


def classify_genu_alignment(intercondylar_cm: Optional[float],
                              intermalleolar_cm: Optional[float]
                              ) -> Tuple[str, str, str]:
    """Classify genu varum (bow-legs) vs genu valgum (knock-knees) based on
    the knee and ankle gaps. This is an INDEPENDENT diagnostic from HKA —
    same patient can be flagged by both for stronger conviction.

    Logic:
        - Both gaps small (< 2 cm)            → 'normal_alignment'
        - Knee gap large, ankle gap small     → 'genu_varum'  (bow-legs)
        - Ankle gap large, knee gap small     → 'genu_valgum' (knock-knees)
        - Both large                          → 'ambiguous' (stance unclear)

    Returns (classification, severity, note).
    """
    if intercondylar_cm is None or intermalleolar_cm is None:
        return ("insufficient_data", "n/a",
                "knee/ankle gap measurements unavailable (need scale calibration)")

    knee = float(intercondylar_cm)
    ankle = float(intermalleolar_cm)

    def _severity(gap):
        if gap <= GAP_NORMAL_MAX_CM:
            return "none"
        if gap <= GAP_MILD_MAX_CM:
            return "mild"
        if gap <= GAP_MODERATE_MAX_CM:
            return "moderate"
        return "severe"

    knee_above = knee > GAP_NORMAL_MAX_CM
    ankle_above = ankle > GAP_NORMAL_MAX_CM

    if not knee_above and not ankle_above:
        return ("normal_alignment", "none",
                f"Both knee gap ({knee:.1f}cm) and ankle gap ({ankle:.1f}cm) "
                f"are within normal range (≤{GAP_NORMAL_MAX_CM}cm).")

    if knee_above and not ankle_above:
        sev = _severity(knee)
        return ("genu_varum", sev,
                f"Knee gap {knee:.1f}cm vs ankle gap {ankle:.1f}cm → "
                f"bow-legs ({sev}). Threshold for {sev}: "
                f"{GAP_NORMAL_MAX_CM}-{GAP_MILD_MAX_CM}cm mild, "
                f"{GAP_MILD_MAX_CM}-{GAP_MODERATE_MAX_CM}cm moderate, "
                f">{GAP_MODERATE_MAX_CM}cm severe.")

    if ankle_above and not knee_above:
        sev = _severity(ankle)
        return ("genu_valgum", sev,
                f"Ankle gap {ankle:.1f}cm vs knee gap {knee:.1f}cm → "
                f"knock-knees ({sev}). Threshold for {sev}: "
                f"{GAP_NORMAL_MAX_CM}-{GAP_MILD_MAX_CM}cm mild, "
                f"{GAP_MILD_MAX_CM}-{GAP_MODERATE_MAX_CM}cm moderate, "
                f">{GAP_MODERATE_MAX_CM}cm severe.")

    # Both gaps elevated — patient may not be in a clean stance, or the
    # pose detector is misplacing landmarks. Flag explicitly.
    return ("ambiguous", "low_confidence",
            f"Both knee gap ({knee:.1f}cm) AND ankle gap ({ankle:.1f}cm) are "
            f"elevated. Patient stance may be wide, or pose landmarks may be "
            f"mislocalized. Knee gap > ankle gap suggests varus; "
            f"the opposite suggests valgus.")


# Hip X-separation / leg-height ratio — the PRIMARY anterior-detection signal.
# Hips don't compress when feet move together, and pelvis width is bony (not
# affected by clothing), so this is the cleanest indicator of view direction.
# Anterior view: hip_sep_ratio ≈ 0.25–0.32 (pelvis width / leg length).
# Side view:    hip_sep_ratio ≈ 0  (one hip occludes the other).
ANTERIOR_VIEW_MIN_HIP_SEP = 0.16   # ≤ ~55° from anterior, still usable
ANTERIOR_VIEW_CLEAN_HIP_SEP = 0.22  # ≤ ~30° from anterior, clean
# Avg X-separation ratio kept as a secondary diagnostic only — confounded by
# stance width (feet together → low knee/ankle sep) and clothing artifacts.
ANTERIOR_VIEW_MIN_RATIO = 0.25
ANTERIOR_VIEW_CLEAN_RATIO = 0.35


def compute_frame_view_quality_2d(leg_keypoints: dict) -> Optional[dict]:
    """Compute view quality for a SINGLE frame's 2D pose.

    Returns None if any keypoint is missing or low-confidence; otherwise:
        {
          "avg_sep_ratio": ...,   # avg of (hip,knee,ankle) X-diff / leg height
          "is_anterior_ish":  bool,
          "is_clean_anterior": bool,
        }
    """
    try:
        lh = leg_keypoints["left_hip"]; rh = leg_keypoints["right_hip"]
        lk = leg_keypoints["left_knee"]; rk = leg_keypoints["right_knee"]
        la = leg_keypoints["left_ankle"]; ra = leg_keypoints["right_ankle"]
    except KeyError:
        return None
    if min(p.get("score", 0) for p in (lh, rh, lk, rk, la, ra)) < MIN_KEYPOINT_SCORE:
        return None
    leg_h = max(abs(la["y"] - lh["y"]), abs(ra["y"] - rh["y"]))
    if leg_h < 1.0:
        return None
    sep_hip = abs(lh["x"] - rh["x"]) / leg_h
    sep_knee = abs(lk["x"] - rk["x"]) / leg_h
    sep_ankle = abs(la["x"] - ra["x"]) / leg_h
    avg = (sep_hip + sep_knee + sep_ankle) / 3.0
    # Hip separation is the gating signal — stable across stance and clothing.
    # See ANTERIOR_VIEW_MIN_HIP_SEP comment above for rationale.
    return {
        "hip_sep_ratio": float(sep_hip),
        "knee_sep_ratio": float(sep_knee),
        "ankle_sep_ratio": float(sep_ankle),
        "avg_sep_ratio": float(avg),
        "is_anterior_ish": bool(sep_hip >= ANTERIOR_VIEW_MIN_HIP_SEP),
        "is_clean_anterior": bool(sep_hip >= ANTERIOR_VIEW_CLEAN_HIP_SEP),
    }


def compute_view_quality_from_2d_pose(pose_persons_per_frame: List[dict]) -> dict:
    """Aggregate per-frame view qualities into a session-level summary.

    For a 180° orbit capture, this reports:
      - How many frames are near-anterior (usable for HKA)
      - How many are clean-anterior (best for HKA)
      - The session's PEAK view quality (the best frame's avg_sep_ratio)
    """
    per_frame = []
    for f in pose_persons_per_frame:
        lk = f.get("leg_keypoints", {})
        q = compute_frame_view_quality_2d(lk)
        per_frame.append(q)

    valid = [q for q in per_frame if q is not None]
    if not valid:
        return {
            "view_quality": None, "view_label": "unknown",
            "warning": "Could not assess view — no high-confidence pose frames.",
            "per_frame": per_frame,
        }

    avg_seps = [q["avg_sep_ratio"] for q in valid]
    hip_seps = [q["hip_sep_ratio"] for q in valid]
    peak = max(avg_seps)
    peak_hip = max(hip_seps)
    avg = float(np.mean(avg_seps))
    mean_hip = float(np.mean(hip_seps))
    mean_knee = float(np.mean([q["knee_sep_ratio"] for q in valid]))
    mean_ankle = float(np.mean([q["ankle_sep_ratio"] for q in valid]))
    n_anterior_ish = sum(1 for q in valid if q["is_anterior_ish"])
    n_clean_anterior = sum(1 for q in valid if q["is_clean_anterior"])

    # Session label is driven by the BEST frame's HIP separation (the most
    # reliable view-direction signal — see ANTERIOR_VIEW_MIN_HIP_SEP comment).
    # A 180° orbit produces both near-anterior AND side frames; we classify
    # the session by what's usable, not by the mean.
    if peak_hip >= ANTERIOR_VIEW_CLEAN_HIP_SEP:
        label = "anterior"
        warning = None
    elif peak_hip >= ANTERIOR_VIEW_MIN_HIP_SEP:
        label = "oblique"
        warning = (
            f"Best frame is oblique (peak hip X-sep = {peak_hip:.2f}, "
            f"<{ANTERIOR_VIEW_CLEAN_HIP_SEP} required for clean anterior). "
            f"{n_anterior_ish}/{len(valid)} frames are usable for HKA. "
            f"For best results capture extra frames at the front of the patient."
        )
    else:
        label = "side"
        warning = (
            f"NO ANTERIOR FRAMES DETECTED (best hip X-sep = {peak_hip:.2f}, "
            f"required ≥{ANTERIOR_VIEW_MIN_HIP_SEP}). All frames are profile/oblique "
            f"— HKA-based varus/valgus is NOT geometrically valid here. "
            f"Re-record with the patient facing the camera directly."
        )

    quality = min(1.0, max(0.0, (peak_hip - 0.05) / (0.30 - 0.05)))

    return {
        "view_quality": float(quality),
        "view_label": label,
        "peak_hip_sep_ratio": float(peak_hip),
        "peak_avg_sep_ratio": float(peak),
        "mean_avg_sep_ratio": avg,
        "mean_hip_separation_ratio": mean_hip,
        "mean_knee_separation_ratio": mean_knee,
        "mean_ankle_separation_ratio": mean_ankle,
        "avg_separation_ratio": avg,
        "n_anterior_ish_frames": int(n_anterior_ish),
        "n_clean_anterior_frames": int(n_clean_anterior),
        "n_frames_assessed": int(len(valid)),
        "warning": warning,
        "per_frame": per_frame,
    }


def compute_reliability_score(n_used: int, n_total: int,
                                iqr_deg: Optional[float],
                                deviation_deg: float,
                                view_quality: Optional[float] = None
                                ) -> Tuple[float, str]:
    """Compute a 0–1 reliability score for a HKA classification.

    Combines FOUR factors:
      1. Sample size: more frames = more reliable
      2. Consistency: lower IQR across frames = more reliable
      3. Threshold proximity: far from any boundary = more reliable
      4. View quality: anterior view = reliable, side view = unreliable

    Returns (score, label) where label ∈ {'high', 'medium', 'low'}.
    """
    sample_score = min(1.0, n_used / 10.0)
    if iqr_deg is None:
        iqr_score = 0.5
    else:
        iqr_score = max(0.0, 1.0 - iqr_deg / 6.0)
    boundaries = [NORMAL_MAX_DEG, BORDERLINE_MAX_DEG, MILD_MAX_DEG, MODERATE_MAX_DEG]
    abs_dev = abs(deviation_deg)
    nearest = min(abs(abs_dev - b) for b in boundaries)
    threshold_score = min(1.0, nearest / 1.5)
    view_score = view_quality if view_quality is not None else 0.7

    score = sample_score * iqr_score * threshold_score * view_score
    if score >= 0.55:
        label = "high"
    elif score >= 0.30:
        label = "medium"
    else:
        label = "low"
    return float(score), label


def bootstrap_classification_margin(per_frame_devs: List[float],
                                     n_bootstrap: int = 1000,
                                     seed: int = 42) -> dict:
    """Bootstrap the median HKA deviation to get a margin-of-error.

    Resamples the per-frame deviations 1000× with replacement, computes the
    median each time, then returns the 95% interval. Also reports the
    fraction of bootstrap replicates that land in each classification bucket,
    so you can see "P(varus)=85%, P(normal)=12%, P(borderline-varus)=3%".

    Returns:
        {
            "median": float,
            "ci_low": float,  "ci_high": float,
            "class_probs": {  "normal": 0.85, "varus_borderline": 0.10, ... },
        }
    """
    if not per_frame_devs:
        return {}
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_frame_devs, dtype=float)
    n = len(arr)
    medians = np.empty(n_bootstrap)
    classes = []
    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, n, n)]
        med = float(np.median(sample))
        medians[i] = med
        cls, sev, _, _ = classify_hka_deviation(med, iqr_deg=None)
        if cls == "normal":
            label = "normal"
        else:
            label = f"{cls}_{sev}"
        classes.append(label)

    ci_low, ci_high = np.percentile(medians, [2.5, 97.5])
    unique, counts = np.unique(classes, return_counts=True)
    class_probs = {str(u): float(c / n_bootstrap) for u, c in zip(unique, counts)}
    return {
        "median": float(np.median(medians)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "class_probs": class_probs,
    }


def classify_lld(lld_pct: float) -> Tuple[str, str]:
    """Classify a leg-length discrepancy by percentage of average leg length.

    Returns (severity, note).
    """
    if lld_pct <= LLD_NORMAL_MAX_PCT:
        return "normal", f"LLD = {lld_pct:.1f}% — within normal range (≤{LLD_NORMAL_MAX_PCT}%)."
    elif lld_pct <= LLD_MILD_MAX_PCT:
        return "mild", f"LLD = {lld_pct:.1f}% — mild ({LLD_NORMAL_MAX_PCT}–{LLD_MILD_MAX_PCT}%)."
    elif lld_pct <= LLD_MODERATE_MAX_PCT:
        return "moderate", f"LLD = {lld_pct:.1f}% — moderate ({LLD_MILD_MAX_PCT}–{LLD_MODERATE_MAX_PCT}%)."
    else:
        return "severe", f"LLD = {lld_pct:.1f}% — severe (>{LLD_MODERATE_MAX_PCT}%)."


# ════════════════════════════════════════════════════════════════════
#  Per-leg measurement (single frame)
# ════════════════════════════════════════════════════════════════════

@dataclass
class LegFrameMeasurement:
    """All metrics for one leg at one frame."""
    side: str
    frame_idx: int
    hip_3d: List[float]
    knee_3d: List[float]
    ankle_3d: List[float]
    keypoint_score_min: float

    hka_angle_deg: float
    hka_deviation_deg: float
    mad_m: float                  # mechanical axis deviation, METERS
    femur_length_m: float
    tibia_length_m: float
    total_leg_length_m: float
    femur_tibia_ratio: float


def measure_leg_frame(hip_3d, knee_3d, ankle_3d, side: str,
                       frame_idx: int, score_min: float,
                       other_hip_3d=None) -> LegFrameMeasurement:
    """Compute all per-leg metrics for a single frame.

    Args:
        hip_3d, knee_3d, ankle_3d: 3D coordinates of this leg's joints.
        side: "left" or "right".
        frame_idx, score_min: bookkeeping.
        other_hip_3d: 3D coordinates of the OTHER leg's hip. Recommended —
            enables view-independent varus/valgus sign determination.
    """
    hka = _angle_3d(hip_3d, knee_3d, ankle_3d)
    dev, mad = signed_hka_deviation_3d(hip_3d, knee_3d, ankle_3d, side,
                                          other_hip=other_hip_3d)
    femur = _distance_3d(hip_3d, knee_3d)
    tibia = _distance_3d(knee_3d, ankle_3d)
    total = femur + tibia
    ratio = femur / (tibia + 1e-12)
    return LegFrameMeasurement(
        side=side, frame_idx=frame_idx,
        hip_3d=list(hip_3d), knee_3d=list(knee_3d), ankle_3d=list(ankle_3d),
        keypoint_score_min=float(score_min),
        hka_angle_deg=float(hka),
        hka_deviation_deg=float(dev),
        mad_m=float(mad),
        femur_length_m=float(femur),
        tibia_length_m=float(tibia),
        total_leg_length_m=float(total),
        femur_tibia_ratio=float(ratio),
    )


# ════════════════════════════════════════════════════════════════════
#  Multi-frame aggregation (robust statistics)
# ════════════════════════════════════════════════════════════════════

def _median_iqr(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return None, None
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return float(med), float(q3 - q1)


@dataclass
class LegAggregate:
    """Aggregated metrics for one leg across all valid frames."""
    side: str
    n_frames_used: int
    n_frames_total: int

    # Angle metrics (degrees, scale-invariant)
    hka_angle_deg_median: Optional[float] = None
    hka_angle_deg_iqr: Optional[float] = None
    hka_deviation_deg_median: Optional[float] = None
    hka_deviation_deg_iqr: Optional[float] = None

    # Distances (meters, requires metric calibration upstream)
    mad_cm_median: Optional[float] = None
    femur_length_cm_median: Optional[float] = None
    tibia_length_cm_median: Optional[float] = None
    total_leg_length_cm_median: Optional[float] = None
    femur_tibia_ratio_median: Optional[float] = None
    femur_tibia_ratio_iqr: Optional[float] = None

    # Classification
    classification: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[str] = None
    classification_note: Optional[str] = None

    # Reliability score (0..1) + label
    reliability_score: Optional[float] = None
    reliability_label: Optional[str] = None

    # Bootstrap 95% CI on the median HKA deviation
    hka_deviation_ci_low_deg: Optional[float] = None
    hka_deviation_ci_high_deg: Optional[float] = None
    # Probability mass across classification buckets (from bootstrap)
    class_probabilities: Optional[dict] = None


def aggregate_leg_frames(frames: List[LegFrameMeasurement], side: str,
                          n_frames_total: int,
                          view_quality: Optional[float] = None
                          ) -> LegAggregate:
    if not frames:
        return LegAggregate(side=side, n_frames_used=0, n_frames_total=n_frames_total)

    # Per-frame outlier filter: |dev| > IMPLAUSIBLE_DEVIATION_DEG in a single
    # frame is almost always a broken 3D point lookup (the keypoint hit
    # background depth or the wall instead of the patient's leg). Drop those
    # frames before computing the median, but keep at least 3 frames so we
    # don't over-prune a real severe-varus case.
    n_pre = len(frames)
    devs_abs = [(abs(f.hka_deviation_deg), f) for f in frames]
    devs_abs.sort(key=lambda x: x[0])
    plausible = [f for d, f in devs_abs if d <= IMPLAUSIBLE_DEVIATION_DEG]
    if len(plausible) < 3 and len(devs_abs) >= 3:
        plausible = [f for _, f in devs_abs[:max(3, len(devs_abs) // 2)]]
    if len(plausible) < n_pre:
        frames = plausible
        print(f"  [outlier-filter] {side}: dropped {n_pre - len(frames)} "
              f"frame(s) with |dev| > {IMPLAUSIBLE_DEVIATION_DEG}° "
              f"(broken 3D lookup); kept {len(frames)}.")

    hka_med, hka_iqr = _median_iqr([f.hka_angle_deg for f in frames])
    dev_med, dev_iqr = _median_iqr([f.hka_deviation_deg for f in frames])
    mad_med, _ = _median_iqr([f.mad_m for f in frames])
    fem_med, _ = _median_iqr([f.femur_length_m for f in frames])
    tib_med, _ = _median_iqr([f.tibia_length_m for f in frames])
    tot_med, _ = _median_iqr([f.total_leg_length_m for f in frames])
    rat_med, rat_iqr = _median_iqr([f.femur_tibia_ratio for f in frames])

    cls, sev, conf, note = classify_hka_deviation(dev_med, iqr_deg=dev_iqr)

    # Reliability + bootstrap CI on the median (view quality dampens it)
    rel_score, rel_label = compute_reliability_score(
        len(frames), n_frames_total, dev_iqr, dev_med,
        view_quality=view_quality,
    )
    boot = bootstrap_classification_margin([f.hka_deviation_deg for f in frames])

    return LegAggregate(
        side=side,
        n_frames_used=len(frames),
        n_frames_total=n_frames_total,
        hka_angle_deg_median=hka_med,
        hka_angle_deg_iqr=hka_iqr,
        hka_deviation_deg_median=dev_med,
        hka_deviation_deg_iqr=dev_iqr,
        mad_cm_median=(mad_med * 100) if mad_med is not None else None,
        femur_length_cm_median=(fem_med * 100) if fem_med is not None else None,
        tibia_length_cm_median=(tib_med * 100) if tib_med is not None else None,
        total_leg_length_cm_median=(tot_med * 100) if tot_med is not None else None,
        femur_tibia_ratio_median=rat_med,
        femur_tibia_ratio_iqr=rat_iqr,
        classification=cls,
        severity=sev,
        confidence=conf,
        classification_note=note,
        reliability_score=rel_score,
        reliability_label=rel_label,
        hka_deviation_ci_low_deg=boot.get("ci_low"),
        hka_deviation_ci_high_deg=boot.get("ci_high"),
        class_probabilities=boot.get("class_probs"),
    )


# ════════════════════════════════════════════════════════════════════
#  Bilateral assessment
# ════════════════════════════════════════════════════════════════════

@dataclass
class BilateralAssessment:
    """Full leg assessment across both legs."""
    left: LegAggregate
    right: LegAggregate

    # Bilateral metrics (medians across frames where both legs were valid)
    intercondylar_distance_cm: Optional[float] = None
    intermalleolar_distance_cm: Optional[float] = None
    leg_length_difference_cm: Optional[float] = None
    leg_length_difference_pct: Optional[float] = None
    leg_length_discrepancy_side: Optional[str] = None
    leg_length_classification: Optional[str] = None
    leg_length_note: Optional[str] = None

    # Stance-based diagnosis from knee/ankle gaps (independent from HKA)
    genu_alignment_classification: Optional[str] = None  # 'normal_alignment' / 'genu_varum' / 'genu_valgum' / 'ambiguous'
    genu_alignment_severity: Optional[str] = None
    genu_alignment_note: Optional[str] = None

    # Capture-quality diagnostics (computed from 2D pose layout)
    view_quality: Optional[float] = None
    view_label: Optional[str] = None  # 'anterior' / 'oblique' / 'side' / 'unknown'
    view_warning: Optional[str] = None
    view_separation_ratios: Optional[dict] = None

    overall_assessment: Optional[str] = None
    flags: list = field(default_factory=list)


def _bilateral_median(left_frames: List[LegFrameMeasurement],
                       right_frames: List[LegFrameMeasurement],
                       extractor):
    """Median over frames where BOTH legs were measured at the same frame."""
    left_by = {f.frame_idx: f for f in left_frames}
    right_by = {f.frame_idx: f for f in right_frames}
    shared = sorted(set(left_by) & set(right_by))
    if not shared:
        return None
    vals = [extractor(left_by[i], right_by[i]) for i in shared]
    return float(np.median(vals))


def build_bilateral_assessment(left_frames: List[LegFrameMeasurement],
                                right_frames: List[LegFrameMeasurement],
                                n_frames_total: int,
                                view_info: Optional[dict] = None
                                ) -> BilateralAssessment:
    view_quality = view_info.get("view_quality") if view_info else None
    left_agg = aggregate_leg_frames(left_frames, "left", n_frames_total,
                                      view_quality=view_quality)
    right_agg = aggregate_leg_frames(right_frames, "right", n_frames_total,
                                       view_quality=view_quality)

    inter_knee = _bilateral_median(
        left_frames, right_frames,
        lambda l, r: np.linalg.norm(np.asarray(l.knee_3d) - np.asarray(r.knee_3d)),
    )
    inter_ankle = _bilateral_median(
        left_frames, right_frames,
        lambda l, r: np.linalg.norm(np.asarray(l.ankle_3d) - np.asarray(r.ankle_3d)),
    )

    # Leg length discrepancy
    lld_cm = None
    lld_pct = None
    discrep_side = None
    lld_class = None
    lld_note = None
    if (left_agg.total_leg_length_cm_median is not None
            and right_agg.total_leg_length_cm_median is not None):
        L = left_agg.total_leg_length_cm_median
        R = right_agg.total_leg_length_cm_median
        diff = abs(L - R)
        lld_cm = float(diff)
        avg = (L + R) / 2.0
        lld_pct = float(100.0 * diff / max(avg, 1e-6))
        if L < R - 0.1:
            discrep_side = "left"
        elif R < L - 0.1:
            discrep_side = "right"
        else:
            discrep_side = "equal"
        lld_class, lld_note = classify_lld(lld_pct)

    # Genu varum / valgum classification from knee/ankle gaps
    intercond_cm = (inter_knee * 100) if inter_knee is not None else None
    intermal_cm = (inter_ankle * 100) if inter_ankle is not None else None
    genu_class, genu_sev, genu_note = classify_genu_alignment(
        intercond_cm, intermal_cm,
    )

    # Overall summary string
    flags = []
    if left_agg.severity and left_agg.severity != "none":
        flags.append(f"Left leg: {left_agg.classification} ({left_agg.severity}, "
                     f"|dev|={abs(left_agg.hka_deviation_deg_median or 0):.1f}°)")
    if right_agg.severity and right_agg.severity != "none":
        flags.append(f"Right leg: {right_agg.classification} ({right_agg.severity}, "
                     f"|dev|={abs(right_agg.hka_deviation_deg_median or 0):.1f}°)")
    if lld_class and lld_class != "normal":
        flags.append(f"Leg length discrepancy ({discrep_side} shorter): "
                     f"{lld_cm:.1f}cm ({lld_pct:.1f}%) — {lld_class}")
    if genu_class in ("genu_varum", "genu_valgum"):
        flags.append(f"Stance: {genu_class} ({genu_sev}) — "
                     f"knee gap {intercond_cm:.1f}cm, "
                     f"ankle gap {intermal_cm:.1f}cm")
    if genu_class == "ambiguous":
        flags.append(f"Stance ambiguous: {genu_note}")

    # Cross-check HKA vs stance — they should agree.
    # HKA says: dev > 0 → varus, dev < 0 → valgus.
    # Stance says: knee gap > ankle gap → varus; ankle gap > knee gap → valgus.
    def _hka_direction(leg_agg):
        d = leg_agg.hka_deviation_deg_median
        if d is None:
            return None
        if abs(d) <= NORMAL_MAX_DEG: return "normal"
        return "varus" if d > 0 else "valgus"

    hka_l_dir = _hka_direction(left_agg)
    hka_r_dir = _hka_direction(right_agg)
    if intercond_cm is not None and intermal_cm is not None:
        stance_dir = ("varus" if intercond_cm > intermal_cm + 1.0
                       else "valgus" if intermal_cm > intercond_cm + 1.0
                       else "normal")
        for side, hd in [("LEFT", hka_l_dir), ("RIGHT", hka_r_dir)]:
            if hd is None: continue
            if hd != "normal" and stance_dir != "normal" and hd != stance_dir:
                flags.append(
                    f"⚠ {side} HKA→{hd} but stance→{stance_dir} "
                    f"(knee gap {intercond_cm:.1f}cm vs ankle gap {intermal_cm:.1f}cm). "
                    f"Possible pose-detection bias (e.g. loose clothing pulling knee "
                    f"landmarks toward the OUTER pant fabric instead of the patella). "
                    f"Recapture with TIGHT-FITTING pants or shorts."
                )

    # Implausibility flag: severe deviations (>~25°) are biologically rare.
    for side, leg_agg in [("LEFT", left_agg), ("RIGHT", right_agg)]:
        d = leg_agg.hka_deviation_deg_median
        if d is not None and abs(d) > IMPLAUSIBLE_DEVIATION_DEG:
            flags.append(
                f"⚠ {side} |dev|={abs(d):.1f}° exceeds {IMPLAUSIBLE_DEVIATION_DEG}° "
                f"— biologically unusual and almost certainly a measurement "
                f"artifact (loose clothing, oblique capture, body rotation). "
                f"Treat the classification as UNRELIABLE."
            )
            # Force reliability down in dataclass
            leg_agg.reliability_label = "low"
            leg_agg.reliability_score = min(leg_agg.reliability_score or 1.0, 0.15)
            leg_agg.confidence = "low_variance"

    # Surface view-quality warnings ABOVE clinical findings so they're seen first
    if view_info and view_info.get("warning"):
        flags.insert(0, "⚠ " + view_info["warning"])

    l_cls = left_agg.classification or "unknown"
    r_cls = right_agg.classification or "unknown"
    if l_cls == "normal" and r_cls == "normal":
        overall = "Bilateral normal alignment"
    elif l_cls == r_cls and l_cls != "normal":
        overall = f"Bilateral {l_cls}"
    elif l_cls == "normal":
        overall = f"Right-side {r_cls}"
    elif r_cls == "normal":
        overall = f"Left-side {l_cls}"
    else:
        overall = f"Mixed: left {l_cls}, right {r_cls}"

    return BilateralAssessment(
        left=left_agg,
        right=right_agg,
        intercondylar_distance_cm=intercond_cm,
        intermalleolar_distance_cm=intermal_cm,
        leg_length_difference_cm=lld_cm,
        leg_length_difference_pct=lld_pct,
        leg_length_discrepancy_side=discrep_side,
        leg_length_classification=lld_class,
        leg_length_note=lld_note,
        genu_alignment_classification=genu_class,
        genu_alignment_severity=genu_sev,
        genu_alignment_note=genu_note,
        view_quality=view_info.get("view_quality") if view_info else None,
        view_label=view_info.get("view_label") if view_info else None,
        view_warning=view_info.get("warning") if view_info else None,
        view_separation_ratios=({
            "hip": view_info.get("mean_hip_separation_ratio"),
            "knee": view_info.get("mean_knee_separation_ratio"),
            "ankle": view_info.get("mean_ankle_separation_ratio"),
            "average": view_info.get("avg_separation_ratio"),
        } if view_info else None),
        overall_assessment=overall,
        flags=flags,
    )


# ════════════════════════════════════════════════════════════════════
#  Pipeline driver: 2D pose + 3D point map → bilateral assessment
# ════════════════════════════════════════════════════════════════════

def _lookup_3d(pts_per_frame_t, px: float, py: float, radius: int = 3):
    """Look up a 3D point at (px, py) in the H×W×3 point map.

    Averages over a small patch and ignores invalid (near-origin) entries.
    Returns None if no valid point is found.
    """
    H, W = pts_per_frame_t.shape[:2]
    ix = int(round(px))
    iy = int(round(py))
    ix = max(0, min(ix, W - 1))
    iy = max(0, min(iy, H - 1))
    y0 = max(0, iy - radius)
    y1 = min(H, iy + radius + 1)
    x0 = max(0, ix - radius)
    x1 = min(W, ix + radius + 1)
    patch = pts_per_frame_t[y0:y1, x0:x1]
    valid = np.linalg.norm(patch, axis=-1) > 0.01
    if not valid.any():
        return None
    return patch[valid].mean(axis=0)


def measure_from_pose_and_pointmap(pose_results: dict,
                                     points_per_frame: np.ndarray,
                                     image_order: List[str],
                                     ) -> BilateralAssessment:
    """Build a bilateral assessment by:
      1. Iterating frames in image_order (matches NPZ T-axis)
      2. For each frame, extracting leg keypoints from pose_results[name]
      3. Looking up 3D positions in points_per_frame[t]
      4. Computing LegFrameMeasurement for each leg
      5. Aggregating across frames

    Frames where any leg keypoint has score < MIN_KEYPOINT_SCORE are skipped.

    Args:
        pose_results: dict {image_name: {persons: [{leg_keypoints: {...}}]}}
        points_per_frame: (T, H, W, 3) — VGGT/AMB3R world points.
        image_order: list of image names in the order T-axis corresponds to.
    """
    JOINT_NAMES = ["left_hip", "right_hip", "left_knee", "right_knee",
                    "left_ankle", "right_ankle"]
    T = points_per_frame.shape[0]
    n_frames_total = min(T, len(image_order))

    left_frames: List[LegFrameMeasurement] = []
    right_frames: List[LegFrameMeasurement] = []

    # ── Pre-compute view quality from the 2D pose ───────────────────
    # This catches non-anterior captures (side / profile views) where the
    # HKA-based varus/valgus is geometrically invalid.
    persons_for_view = []
    for name in image_order[:T]:
        img_data = pose_results.get(name)
        if img_data and img_data.get("persons"):
            persons_for_view.append(
                max(img_data["persons"], key=lambda p: p.get("mean_score", 0))
            )
    view_info = compute_view_quality_from_2d_pose(persons_for_view)
    if view_info.get("warning"):
        print(f"  [view-quality] {view_info['warning']}")
    if view_info.get("n_anterior_ish_frames") is not None:
        print(f"  [view-quality] {view_info['n_anterior_ish_frames']}/"
              f"{view_info['n_frames_assessed']} frames pass the "
              f"'near-anterior' filter (hip X-sep ≥ {ANTERIOR_VIEW_MIN_HIP_SEP:.2f})")
    per_frame_view = view_info.get("per_frame", [])

    # We map per_frame_view to the same iteration order as persons_for_view,
    # which iterates over image_order[:T] WITH non-None pose results. Build a
    # lookup by name → view-quality dict.
    view_by_name = {}
    pf_iter = iter(per_frame_view)
    for name in image_order[:T]:
        img_data = pose_results.get(name)
        if img_data and img_data.get("persons"):
            view_by_name[name] = next(pf_iter, None)

    n_skipped_oblique = 0
    for t, name in enumerate(image_order[:T]):
        img_data = pose_results.get(name)
        if not img_data or not img_data.get("persons"):
            continue
        # Use the most-confident person (the patient, presumably foreground)
        persons = img_data["persons"]
        best_person = max(persons, key=lambda p: p.get("mean_score", 0))
        lk = best_person.get("leg_keypoints", {})

        scores = {n: lk.get(n, {}).get("score", 0) for n in JOINT_NAMES}
        # Drop the frame entirely if ANY joint is below threshold
        if min(scores.values()) < MIN_KEYPOINT_SCORE:
            continue

        # Per-frame view-quality filter — HKA is only valid for near-anterior
        # views, so we skip frames that are clearly oblique/side for the
        # purposes of HKA aggregation. The frames are still useful upstream
        # (for VGGT multi-view fusion), but we shouldn't pool their HKA
        # values with anterior ones.
        fview = view_by_name.get(name)
        if fview is not None and not fview.get("is_anterior_ish", False):
            n_skipped_oblique += 1
            continue

        l_hip_2d = (lk["left_hip"]["x"], lk["left_hip"]["y"])
        r_hip_2d = (lk["right_hip"]["x"], lk["right_hip"]["y"])
        l_knee_2d = (lk["left_knee"]["x"], lk["left_knee"]["y"])
        r_knee_2d = (lk["right_knee"]["x"], lk["right_knee"]["y"])
        l_ank_2d = (lk["left_ankle"]["x"], lk["left_ankle"]["y"])
        r_ank_2d = (lk["right_ankle"]["x"], lk["right_ankle"]["y"])

        pts_t = points_per_frame[t]
        l_hip_3d = _lookup_3d(pts_t, *l_hip_2d)
        r_hip_3d = _lookup_3d(pts_t, *r_hip_2d)
        l_knee_3d = _lookup_3d(pts_t, *l_knee_2d)
        r_knee_3d = _lookup_3d(pts_t, *r_knee_2d)
        l_ank_3d = _lookup_3d(pts_t, *l_ank_2d)
        r_ank_3d = _lookup_3d(pts_t, *r_ank_2d)
        if any(p is None for p in [l_hip_3d, r_hip_3d, l_knee_3d, r_knee_3d,
                                     l_ank_3d, r_ank_3d]):
            continue

        l_min = min(scores["left_hip"], scores["left_knee"], scores["left_ankle"])
        r_min = min(scores["right_hip"], scores["right_knee"], scores["right_ankle"])

        # Pass the OTHER hip into each leg's measurement so the varus/valgus
        # sign is determined from the patient's true lateral direction
        # (hip-to-hip vector), not from a fragile image-plane heuristic.
        left_frames.append(measure_leg_frame(
            l_hip_3d, l_knee_3d, l_ank_3d, "left", t, l_min,
            other_hip_3d=r_hip_3d,
        ))
        right_frames.append(measure_leg_frame(
            r_hip_3d, r_knee_3d, r_ank_3d, "right", t, r_min,
            other_hip_3d=l_hip_3d,
        ))

    if n_skipped_oblique > 0:
        print(f"  [view-filter] Excluded {n_skipped_oblique} oblique/side "
              f"frames from HKA aggregation (used {len(left_frames)} for left, "
              f"{len(right_frames)} for right).")

    view_info["n_skipped_oblique_frames"] = int(n_skipped_oblique)

    return build_bilateral_assessment(
        left_frames, right_frames, n_frames_total, view_info=view_info,
    ), left_frames, right_frames


# ════════════════════════════════════════════════════════════════════
#  Lower-leg volume estimation (slab-wise ellipse fit)
# ════════════════════════════════════════════════════════════════════
#
# Estimates the volume of the lower leg (knee → ankle) from a person
# point cloud + the 3D keypoints. The point cloud carries only surface
# samples, so we slice into thin slabs along the tibia axis, fit a 2D
# ellipse to each slab's perpendicular cross-section, and sum
# π·a·b·slab_height. This is much more accurate than raw voxel counting
# (which only sees the shell) and more anatomically faithful than a
# convex hull (which fills the posterior concavity behind the tibia).
#
# Robustness strategy (in order):
#   1. Discard points outside [knee_plane, ankle_plane] along tibia axis.
#   2. Discard points with radial distance > MAX_RADIAL_CM (12 cm) — the
#      person cloud may contain background bleed.
#   3. Side-isolate: drop points on the other leg's side of the midline.
#   4. Per-slab MAD outlier filter on radial distance — drops points
#      whose radial distance exceeds 3·MAD from the slab's median.
#   5. Robust ellipse axes from the inner 95th percentile of projections,
#      not the absolute max (so a single bad point can't blow up volume).
#   6. Slabs with too few points after cleaning are skipped and their
#      volume is interpolated from neighbouring slabs.

MAX_RADIAL_CM = 12.0      # any point > this from tibia axis is noise
DEFAULT_N_SLABS = 20
MIN_POINTS_PER_SLAB = 8
SLAB_AXIS_PCT = 95.0      # use 95th-percentile distance as ellipse semi-axis


@dataclass
class LowerLegVolume:
    """Volume estimate for one lower leg (knee → ankle).

    Uses ONLY the slab-wise ellipse-fit method — anatomically faithful
    (lower leg ≈ stack of elliptical cross-sections). Per-slab outlier
    rejection makes the integral robust to noisy/distorted clouds.
    """
    side: str
    method: str = "slab_ellipse_fit"

    # Primary (and only) volume estimate.
    volume_cm3: Optional[float] = None

    tibia_length_cm: Optional[float] = None           # from keypoints
    tibia_length_cloud_cm: Optional[float] = None     # from cloud axial extent
    mean_circumference_cm: Optional[float] = None
    max_circumference_cm: Optional[float] = None

    n_points_raw: int = 0
    n_points_used: int = 0       # after axial + radial + side + SOR filter
    n_points_after_sor: int = 0  # after statistical outlier removal
    n_slabs_total: int = 0
    n_slabs_with_data: int = 0
    # Per-slab profile for visualisation (sorted from knee→ankle).
    # Each entry is (axial_center_cm, semi_axis_a_cm, semi_axis_b_cm).
    slab_profile: Optional[list] = None
    # Sanity diagnostic kept so the viz can surface it; the user wanted
    # the high/medium/low reliability label removed, so we don't compute
    # it anymore.
    distortion_warning: Optional[str] = None
    note: Optional[str] = None


def _fit_slab_ellipse(slab_pts_2d: np.ndarray,
                       pct: float = SLAB_AXIS_PCT) -> Tuple[float, float]:
    """Fit a 2D ellipse to a slab's perpendicular cross-section.

    Uses PCA to find principal axes, then takes the `pct`-percentile of
    |projection along each axis| as the semi-axis length. This is more
    robust than max() because a single outlier point in the slab can't
    inflate the ellipse.
    """
    if len(slab_pts_2d) < 3:
        return 0.0, 0.0
    centered = slab_pts_2d - slab_pts_2d.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvecs columns are principal directions; project onto each
    proj0 = np.abs(centered @ eigvecs[:, 1])  # major axis
    proj1 = np.abs(centered @ eigvecs[:, 0])  # minor axis
    a = float(np.percentile(proj0, pct))
    b = float(np.percentile(proj1, pct))
    return a, b


def _robust_radial_filter(radial: np.ndarray, mad_factor: float = 3.0
                          ) -> np.ndarray:
    """Return boolean mask keeping points within mad_factor·MAD of the median."""
    if len(radial) == 0:
        return np.zeros(0, dtype=bool)
    med = np.median(radial)
    mad = np.median(np.abs(radial - med))
    if mad < 1e-6:
        return np.ones_like(radial, dtype=bool)
    return np.abs(radial - med) <= mad_factor * mad


def _knn_outlier_filter(pts: np.ndarray, *parallel_arrays: np.ndarray,
                         k: int = 12, std_ratio: float = 2.0):
    """KNN-based statistical outlier removal in-process (no open3d needed).

    For each point, compute the mean distance to its k nearest neighbours.
    Keep points whose mean-knn-distance is within `std_ratio * std` of the
    median mean-knn-distance. Drops sparse stragglers (which are typically
    background bleed, wall hits, or scattered noise from VGGT).

    Args:
        pts: (N, 3) point cloud.
        *parallel_arrays: any number of (N, ...) arrays that should be
            indexed the same way as pts (returned in the same order).
        k, std_ratio: hyperparameters.

    Returns:
        (pts_filtered, *parallel_filtered)
    """
    n = len(pts)
    if n <= k + 1:
        return (pts,) + parallel_arrays
    # Use scipy KDTree if available for speed, otherwise fall back to
    # a chunked broadcast distance computation.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=k + 1)
        mean_knn = d[:, 1:].mean(axis=1)
    except Exception:
        # Chunked computation to avoid n×n memory blow-up
        mean_knn = np.empty(n)
        chunk = 2000
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            diffs = pts[s:e, None, :] - pts[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
            d2.sort(axis=1)
            mean_knn[s:e] = np.sqrt(d2[:, 1:k + 1]).mean(axis=1)

    med = float(np.median(mean_knn))
    std = float(np.std(mean_knn))
    if std < 1e-9:
        keep = np.ones(n, dtype=bool)
    else:
        keep = mean_knn <= med + std_ratio * std
    return (pts[keep],) + tuple(arr[keep] for arr in parallel_arrays)


def compute_lower_leg_volume(person_points_cm: np.ndarray,
                               knee_3d_cm: np.ndarray,
                               ankle_3d_cm: np.ndarray,
                               side: str,
                               other_knee_3d_cm: Optional[np.ndarray] = None,
                               other_ankle_3d_cm: Optional[np.ndarray] = None,
                               n_slabs: int = DEFAULT_N_SLABS,
                               radial_max_cm: float = MAX_RADIAL_CM,
                               proximal_skip_cm: float = 2.0,
                               distal_skip_cm: float = 2.0,
                               ) -> LowerLegVolume:
    """Estimate one lower leg's volume from a person point cloud.

    Args:
        person_points_cm: (N, 3) — person-segmented points in metric (cm) space.
            Should already have basic outlier removal applied upstream.
        knee_3d_cm, ankle_3d_cm: (3,) — this leg's joint positions in cm.
        side: "left" or "right" — bookkeeping only.
        other_knee_3d_cm, other_ankle_3d_cm: the OTHER leg's joints. If
            provided, we drop points on the other side of the midline, so
            each leg's volume only sees its own points.
        n_slabs: number of slices along the tibia.
        radial_max_cm: discard points farther than this from the tibia axis.
        proximal_skip_cm: exclude this many cm at the knee end (avoid
            patella + soft-tissue bulge inflating the volume).
        distal_skip_cm: exclude this many cm at the ankle end (avoid
            talus / heel mass leaking into the lower-leg volume).
    """
    knee = np.asarray(knee_3d_cm, dtype=float).reshape(3)
    ankle = np.asarray(ankle_3d_cm, dtype=float).reshape(3)

    out = LowerLegVolume(side=side, n_points_raw=int(len(person_points_cm)))

    tibia = ankle - knee
    L = float(np.linalg.norm(tibia))
    if L < 1.0:
        out.note = "Tibia length too short — cannot fit slabs."
        return out
    tibia_unit = tibia / L
    out.tibia_length_cm = L

    # Effective range: knee + proximal_skip → ankle - distal_skip
    t_lo = proximal_skip_cm
    t_hi = L - distal_skip_cm
    if t_hi - t_lo < 4.0:
        out.note = "Lower-leg range too short after end-trims — cannot fit."
        return out

    if len(person_points_cm) == 0:
        out.note = "No person points provided."
        return out

    pts = np.asarray(person_points_cm, dtype=float)
    rel = pts - knee
    t = rel @ tibia_unit                           # axial coord (cm) from knee
    perp = rel - t[:, None] * tibia_unit           # perpendicular component
    radial = np.linalg.norm(perp, axis=1)          # radial distance (cm)

    # (1) axial-window mask
    m_axial = (t >= t_lo) & (t <= t_hi)
    # (2) radial-max mask
    m_radial = radial < radial_max_cm
    keep = m_axial & m_radial

    # (3) side-isolation using the midline between left and right knees
    if other_knee_3d_cm is not None:
        other_knee = np.asarray(other_knee_3d_cm, dtype=float).reshape(3)
        midpoint = 0.5 * (knee + other_knee)
        # lateral direction = THIS knee away from other knee (outward for this leg)
        lateral = knee - other_knee
        lat_norm = np.linalg.norm(lateral)
        if lat_norm > 1e-6:
            lateral /= lat_norm
            # Point is on THIS leg if its component along `lateral` (relative
            # to midpoint) is positive. We give a small inclusion buffer of
            # 1cm past the midline so the inner side of the tibia is included.
            lat_proj = (pts - midpoint) @ lateral
            keep = keep & (lat_proj > -1.0)

    pts_keep = pts[keep]
    t_keep = t[keep]
    perp_keep = perp[keep]
    radial_keep = radial[keep]

    if len(pts_keep) < n_slabs * MIN_POINTS_PER_SLAB // 2:
        out.note = (f"Too few points after filtering ({len(pts_keep)}) — "
                    f"need at least {n_slabs * MIN_POINTS_PER_SLAB // 2}.")
        out.n_points_used = int(len(pts_keep))
        return out

    out.n_points_used = int(len(pts_keep))

    # (4) Statistical outlier removal in the leg-local frame. We use a fast
    # numpy KNN-style filter (without open3d dependency here): for each
    # point, the mean distance to its K nearest neighbours should be close
    # to the cloud's overall median KNN distance — outliers stick out.
    # This catches background bleed, wall-points, and pose-detector mistakes
    # that survived the radial/axial filters.
    pts_keep, t_keep, perp_keep, radial_keep = _knn_outlier_filter(
        pts_keep, t_keep, perp_keep, radial_keep,
        k=12, std_ratio=2.0,
    )
    out.n_points_after_sor = int(len(pts_keep))
    if len(pts_keep) < n_slabs * MIN_POINTS_PER_SLAB // 2:
        out.note = (f"Too few points after SOR ({len(pts_keep)}).")
        return out

    # (5) Distortion sanity check: the cloud's axial extent should match the
    # keypoint-derived tibia length within ±20%. A bigger mismatch suggests
    # VGGT placed the keypoints and the surface points on different scales
    # (e.g. one frame at a different depth) — volume estimates won't be
    # trustworthy.
    t_lo_cloud = float(np.percentile(t_keep, 2))
    t_hi_cloud = float(np.percentile(t_keep, 98))
    cloud_extent = t_hi_cloud - t_lo_cloud
    out.tibia_length_cloud_cm = cloud_extent
    if abs(cloud_extent - L) / max(L, 1e-6) > 0.30:
        out.distortion_warning = (
            f"Cloud axial extent ({cloud_extent:.1f} cm) differs from "
            f"keypoint tibia length ({L:.1f} cm) by "
            f"{abs(cloud_extent - L) / L * 100:.0f}%. The point cloud may "
            f"be distorted relative to keypoints — treat volume with caution."
        )

    # 2D coordinate frame in the slab plane: any two unit vectors perpendicular
    # to the tibia axis. We use a stable construction (Gram-Schmidt off the
    # least-axis-aligned global axis).
    if abs(tibia_unit[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - tibia_unit * (ref @ tibia_unit)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(tibia_unit, e1)

    proj_2d = np.column_stack([perp_keep @ e1, perp_keep @ e2])

    # (4) per-slab MAD radial filter + (5) ellipse fit + slab volume sum
    edges = np.linspace(t_lo, t_hi, n_slabs + 1)
    slab_h = (t_hi - t_lo) / n_slabs
    out.n_slabs_total = n_slabs

    slab_axes = []         # list of (a, b) per slab; None where empty
    slab_volumes = []
    slab_circumferences = []
    n_with_data = 0
    for i in range(n_slabs):
        in_slab = (t_keep >= edges[i]) & (t_keep < edges[i + 1])
        slab_radial = radial_keep[in_slab]
        slab_pts2 = proj_2d[in_slab]
        if len(slab_pts2) < MIN_POINTS_PER_SLAB:
            slab_axes.append(None)
            slab_volumes.append(None)
            slab_circumferences.append(None)
            continue
        # Per-slab outlier filter on radial distance
        mask = _robust_radial_filter(slab_radial, mad_factor=3.0)
        if mask.sum() < MIN_POINTS_PER_SLAB:
            mask = np.ones_like(slab_radial, dtype=bool)
        slab_pts2 = slab_pts2[mask]
        a, b = _fit_slab_ellipse(slab_pts2, pct=SLAB_AXIS_PCT)
        if a < 0.5 or b < 0.5:
            slab_axes.append(None)
            slab_volumes.append(None)
            slab_circumferences.append(None)
            continue
        slab_axes.append((a, b))
        slab_volumes.append(math.pi * a * b * slab_h)
        # Ramanujan's approximation to ellipse circumference
        h_param = ((a - b) ** 2) / ((a + b) ** 2 + 1e-9)
        circ = math.pi * (a + b) * (1 + 3 * h_param / (10 + math.sqrt(4 - 3 * h_param)))
        slab_circumferences.append(circ)
        n_with_data += 1

    out.n_slabs_with_data = n_with_data
    if n_with_data == 0:
        out.note = "No slabs had enough points to fit."
        return out

    # Interpolate missing slabs from neighbours so the total isn't zero-biased.
    last_good = None
    next_good_idx = [None] * n_slabs
    for j in range(n_slabs - 1, -1, -1):
        if slab_volumes[j] is not None:
            last_good = j
        next_good_idx[j] = last_good
    prev_good_idx = [None] * n_slabs
    last_good = None
    for j in range(n_slabs):
        if slab_volumes[j] is not None:
            last_good = j
        prev_good_idx[j] = last_good

    filled_volumes = []
    for j in range(n_slabs):
        if slab_volumes[j] is not None:
            filled_volumes.append(slab_volumes[j])
            continue
        p, q = prev_good_idx[j], next_good_idx[j]
        if p is not None and q is not None:
            filled_volumes.append(0.5 * (slab_volumes[p] + slab_volumes[q]))
        elif p is not None:
            filled_volumes.append(slab_volumes[p])
        elif q is not None:
            filled_volumes.append(slab_volumes[q])
        else:
            filled_volumes.append(0.0)

    out.volume_cm3 = float(sum(filled_volumes))
    circs = [c for c in slab_circumferences if c is not None]
    if circs:
        out.mean_circumference_cm = float(np.mean(circs))
        out.max_circumference_cm = float(np.max(circs))

    # Persist per-slab profile so the visualisation can render the fitted
    # ellipses on top of the 3D cloud. Each entry: (t_center, a, b).
    profile = []
    for i in range(n_slabs):
        if slab_axes[i] is None:
            continue
        a_ax, b_ax = slab_axes[i]
        t_center = 0.5 * (edges[i] + edges[i + 1])
        profile.append([float(t_center), float(a_ax), float(b_ax)])
    out.slab_profile = profile

    # Plausibility check: typical adult lower leg ≈ 1.2-3.5 L; values
    # > 5 L are almost certainly inflated by loose clothing or noise.
    if out.volume_cm3 is not None and out.volume_cm3 > 5000:
        out.note = (f"Volume ({out.volume_cm3:.0f} cm³) exceeds physiologic "
                    f"upper bound (~5 L) — likely loose-clothing inflation "
                    f"or pose-detector placing keypoints outside the leg.")

    return out


def clean_person_pointcloud(points: np.ndarray,
                              max_passes: int = 2,
                              sor_k: int = 20,
                              sor_std: float = 1.8) -> np.ndarray:
    """Pre-clean the person point cloud before slab-fitting both legs.

    Two-pass approach so distant background bleed and isolated stragglers
    both get rejected:
      Pass 1: aggressive SOR (catches obvious outliers like wall hits)
      Pass 2: gentler SOR after centroid-trim (catches subtle stragglers)
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < sor_k + 1:
        return pts

    for _ in range(max_passes):
        (pts,) = _knn_outlier_filter(pts, k=sor_k, std_ratio=sor_std)
        if len(pts) < sor_k + 1:
            break

    # Centroid-trim: drop points that sit beyond the 99th-percentile
    # distance-from-centroid by a factor of 1.3 (handles distortion-induced
    # tail points where a few frames place the surface far from the rest).
    if len(pts) >= 10:
        c = pts.mean(axis=0)
        d = np.linalg.norm(pts - c, axis=1)
        thr = np.percentile(d, 99) * 1.3
        pts = pts[d < thr]
    return pts


def compute_bilateral_lower_leg_volumes(person_points_cm: np.ndarray,
                                          assessment: BilateralAssessment,
                                          left_frames: List[LegFrameMeasurement],
                                          right_frames: List[LegFrameMeasurement],
                                          metric_calibrated: bool,
                                          ) -> Tuple[Optional[LowerLegVolume],
                                                     Optional[LowerLegVolume]]:
    """Convenience wrapper: extract median knee/ankle 3D positions from the
    per-frame measurements and compute both lower-leg volumes from a shared
    person point cloud.

    Volumes are only meaningful when the pipeline ran with metric
    calibration AND the point cloud is in cm.
    """
    if not metric_calibrated:
        return None, None
    if person_points_cm is None or len(person_points_cm) == 0:
        return None, None

    def _robust_median_3d(frames: List[LegFrameMeasurement],
                            knee_key: str, ankle_key: str):
        """Find a trustworthy median (knee_3d, ankle_3d) pair across frames.

        Per-frame depth lookups can be wildly wrong when the 2D keypoint
        lands on the floor / wall (especially the ankle on oblique frames).
        Such frames produce tibia vectors that are too long or point
        sideways instead of vertically — both red flags for noisy depth.

        Filter:
            1. Tibia length must be within an anatomic range (the cloud's
               scale is unknown here — points are in cloud-native units,
               which after auto-scaling will become cm). We pre-scale
               outside; here we operate on whatever units `frames` carry,
               so we filter relative to the per-leg median tibia length.
            2. Tibia direction must be VERTICAL-dominant: the largest
               component (|x|,|y|,|z|) should be >= 0.55 × the magnitude.
               In VGGT's Y-down world frame, a standing leg has
               Y-dominant tibia. We don't hardcode the axis — we just
               require ONE axis dominate so the leg isn't diagonal.
        """
        if not frames:
            return None, None, []
        knees = np.array([getattr(f, knee_key) for f in frames])
        ankles = np.array([getattr(f, ankle_key) for f in frames])
        tibias = ankles - knees
        lengths = np.linalg.norm(tibias, axis=1)

        # Use the median tibia length as the anatomic anchor; reject
        # frames more than 35% off this median (catches depth artifacts).
        med_len = float(np.median(lengths))
        if med_len < 1e-6:
            return knees[len(knees) // 2], ankles[len(ankles) // 2], []
        len_ok = np.abs(lengths - med_len) / med_len < 0.35

        # Verticality: each frame's tibia should have ONE dominant axis
        # (>= 55% of the vector magnitude). Diagonal-tibia frames suggest
        # the ankle 3D was hijacked by a floor/wall depth.
        with np.errstate(invalid="ignore", divide="ignore"):
            normed = tibias / lengths[:, None]
        max_axis = np.max(np.abs(normed), axis=1)
        vert_ok = max_axis >= 0.55

        keep = len_ok & vert_ok
        if keep.sum() < max(3, len(frames) // 3):
            # Too few inliers — fall back to all frames (better than
            # nothing) and surface the issue.
            kept = np.ones(len(frames), dtype=bool)
        else:
            kept = keep

        idx_kept = list(np.flatnonzero(kept))
        knee_med = np.median(knees[kept], axis=0)
        ankle_med = np.median(ankles[kept], axis=0)
        return knee_med, ankle_med, idx_kept

    l_knee, l_ank, l_kept = _robust_median_3d(left_frames, "knee_3d", "ankle_3d")
    r_knee, r_ank, r_kept = _robust_median_3d(right_frames, "knee_3d", "ankle_3d")
    if l_knee is not None:
        print(f"  [volume] left:  {len(l_kept)}/{len(left_frames)} frames "
              f"survive anatomic/vertical filter")
    if r_knee is not None:
        print(f"  [volume] right: {len(r_kept)}/{len(right_frames)} frames "
              f"survive anatomic/vertical filter")

    # Convert metres → cm if the input keypoints were in metres. We can detect
    # this by checking the tibia length: a real tibia is 30–50 cm. If the value
    # < 1, the cloud is in metres.
    def _autoscale(p):
        return p * 100.0 if p is not None and np.linalg.norm(p) < 3.0 else p
    # `points_per_frame` and keypoints share the same units, so they scale together.
    if person_points_cm is not None and len(person_points_cm) > 0:
        # Heuristic: median point's distance from origin tells us the scale.
        med_r = float(np.median(np.linalg.norm(person_points_cm, axis=1)))
        if med_r < 5.0:    # < 5 m worth → it's in metres
            person_points_cm = person_points_cm * 100.0
            l_knee, l_ank = _autoscale(l_knee), _autoscale(l_ank)
            r_knee, r_ank = _autoscale(r_knee), _autoscale(r_ank)

    # Up-front cleanup of the shared cloud — single pass before slab fitting
    # so both legs see a less-noisy, less-distorted cloud.
    n_before = len(person_points_cm)
    cleaned = clean_person_pointcloud(person_points_cm)
    n_after = len(cleaned)
    print(f"  [volume] cleaned person cloud: {n_before} → {n_after} points")
    person_points_cm = cleaned

    left_vol = right_vol = None
    if l_knee is not None and l_ank is not None:
        left_vol = compute_lower_leg_volume(
            person_points_cm, l_knee, l_ank, "left",
            other_knee_3d_cm=r_knee, other_ankle_3d_cm=r_ank,
        )
    if r_knee is not None and r_ank is not None:
        right_vol = compute_lower_leg_volume(
            person_points_cm, r_knee, r_ank, "right",
            other_knee_3d_cm=l_knee, other_ankle_3d_cm=l_ank,
        )
    return left_vol, right_vol
