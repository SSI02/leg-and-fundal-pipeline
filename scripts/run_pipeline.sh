#!/bin/bash
# Quick-run script for the full leg deformity pipeline.
#
# Recommended workflow:
#   1. Run scale picker to set per-image cm/pixel from clicked points:
#      bash scripts/run_scale_picker.sh data/input/patient_001
#      → produces data/input/patient_001/scale_calibration.json
#
#   2. Run pipeline with that calibration:
#      SCALE_CALIB=data/input/patient_001/scale_calibration.json \
#        bash scripts/run_pipeline.sh data/input/patient_001 data/output/patient_001
#
# Usage:
#   bash scripts/run_pipeline.sh <image_dir> <output_dir> [marker_size_cm]
#
# Environment variables:
#   SCALE_CALIB=path  Path to scale_calibration.json (preferred over ArUco)
#   SKIP_3D=1         Skip 3D reconstruction (2D pose + measurements only)
#   SKIP_SAM3=1       Skip SAM3 person segmentation
#   RECON_MODEL=...   3D reconstruction model: amb3r (default) or vggt
#   POSE_MODEL=...    Pose model: human, vitpose, vitpose-l, vitpose-h, wholebody
#   MAX_IMAGES=N      Max images for 3D reconstruction (default: 4)
#   NO_OUTLIER=1      Skip statistical outlier removal in post-processing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_DIR="${1:?Usage: $0 <image_dir> <output_dir> [marker_size_cm]}"
OUTPUT_DIR="${2:?Usage: $0 <image_dir> <output_dir> [marker_size_cm]}"
MARKER_SIZE="${3:-}"

# Find conda base and leg_pipeline python directly
CONDA_BASE="${CONDA_PREFIX:-$HOME/miniconda3}"
# If CONDA_PREFIX points to an env (not base), go up
if [ -d "${CONDA_BASE}/envs" ]; then
    : # already at base
elif [ -d "$(dirname "$(dirname "$CONDA_BASE")")/envs" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_BASE")")"
fi
PYTHON="${CONDA_BASE}/envs/leg_pipeline/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: leg_pipeline env not found at $PYTHON"
    echo "Run: bash scripts/setup_pipeline.sh"
    exit 1
fi

# Build the command
CMD="$PYTHON ${PROJECT_DIR}/src/pipeline/orchestrator.py"
CMD="$CMD --image_dir $IMAGE_DIR"
CMD="$CMD --output_dir $OUTPUT_DIR"

if [ -n "$MARKER_SIZE" ]; then
    CMD="$CMD --marker_size_cm $MARKER_SIZE"
fi

if [ -n "$SCALE_CALIB" ]; then
    CMD="$CMD --scale_calibration $SCALE_CALIB"
fi

if [ "${SKIP_3D:-0}" = "1" ]; then
    CMD="$CMD --skip_3d"
fi

if [ "${SKIP_SAM3:-0}" = "1" ]; then
    CMD="$CMD --skip_sam3"
fi

if [ -n "$RECON_MODEL" ]; then
    CMD="$CMD --recon_model $RECON_MODEL"
fi

if [ -n "$POSE_MODEL" ]; then
    CMD="$CMD --pose_model $POSE_MODEL"
fi

if [ -n "$MAX_IMAGES" ]; then
    CMD="$CMD --max_images $MAX_IMAGES"
fi

if [ "${NO_OUTLIER:-0}" = "1" ]; then
    CMD="$CMD --no_outlier_removal"
fi

# Check which optional envs are available
SAM3_STATUS="not found"
if [ -f "${CONDA_BASE}/envs/vv_sam3/bin/python" ]; then
    SAM3_STATUS="available"
fi

echo "============================================"
echo "Leg Deformity Detection Pipeline"
echo "============================================"
echo "Image dir:   $IMAGE_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Scale calib: ${SCALE_CALIB:-<not set>}"
echo "ArUco size:  ${MARKER_SIZE:-<not set>}"
echo "Recon model: ${RECON_MODEL:-amb3r}"
echo "Pose model:  ${POSE_MODEL:-human}"
echo "Skip 3D:     ${SKIP_3D:-0}"
echo "Skip SAM3:   ${SKIP_SAM3:-0}  (env: $SAM3_STATUS)"
echo "Python:      $PYTHON"
echo "============================================"
echo ""

$CMD

echo ""
echo "============================================"
echo "Done! To view results:"
echo "  $PYTHON ${PROJECT_DIR}/src/visualization/server.py --output_dir $OUTPUT_DIR"
echo "============================================"
