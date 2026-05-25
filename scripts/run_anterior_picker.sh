#!/bin/bash
# Interactive picker for the most-anterior (front-facing) frame.
#
# Usage:
#   bash scripts/run_anterior_picker.sh <video_or_frames_dir> <output_json>
#
# Example:
#   bash scripts/run_anterior_picker.sh \
#        data/input/patient_005.mp4 \
#        data/input/patient_005_anterior.json
#
# Then feed the result into the leg pipeline:
#   ANTERIOR_FRAME=$(jq -r .anterior_frame  data/input/patient_005_anterior.json) \
#     SCALE_CALIB=data/input/patient_005_scale.json \
#     bash scripts/run_leg_pipeline.sh \
#         data/input/patient_005.mp4 \
#         data/output/patient_005
#
# Environment:
#   N_FRAMES=N  Frames to extract (default 30; MUST match the pipeline's
#               --n_frames so the chosen filename is consistent).
#   PORT=8091   Picker port.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT="${1:?Usage: $0 <video_or_frames_dir> <output_json>}"
OUTPUT="${2:?Usage: $0 <video_or_frames_dir> <output_json>}"

CONDA_BASE="${CONDA_PREFIX:-$HOME/miniconda3}"
if [ -d "${CONDA_BASE}/envs" ]; then :;
elif [ -d "$(dirname "$(dirname "$CONDA_BASE")")/envs" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_BASE")")"
fi
PYTHON="${CONDA_BASE}/envs/leg_pipeline/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: leg_pipeline env not found at $PYTHON"; exit 1
fi

CMD="$PYTHON ${PROJECT_DIR}/src/calibration/anterior_picker.py --output $OUTPUT --port ${PORT:-8091}"
if [ -n "$N_FRAMES" ]; then CMD="$CMD --n_frames $N_FRAMES"; fi

if [ -d "$INPUT" ]; then
    CMD="$CMD --image_dir $INPUT"
elif [ -f "$INPUT" ]; then
    CMD="$CMD --video $INPUT"
else
    echo "ERROR: input not found: $INPUT"; exit 1
fi

echo "============================================"
echo "Anterior frame picker"
echo "============================================"
echo "Input:   $INPUT"
echo "Output:  $OUTPUT"
echo "============================================"
echo ""

eval $CMD
