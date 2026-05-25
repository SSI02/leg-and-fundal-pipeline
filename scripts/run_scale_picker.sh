#!/bin/bash
# Launch the interactive scale picker tool.
#
# Click 2 points on a known-size object in each image (or just on frame 0
# for video, then "Track to all"). The tool computes per-image cm/pixel.
#
# Usage:
#   bash scripts/run_scale_picker.sh <image_dir_or_video> [output_path] [port]
#
# Examples:
#   # Image directory
#   bash scripts/run_scale_picker.sh data/input/patient_001
#   → saves data/input/patient_001/scale_calibration.json
#
#   # Video file (frames extracted automatically into <video>_frames/)
#   bash scripts/run_scale_picker.sh data/input/patient.mp4
#
#   # Custom output path and port
#   bash scripts/run_scale_picker.sh data/input/patient_001 my.json 8095
#
# Environment variables:
#   N_FRAMES=N  Number of frames to extract from video (default: 30)
#   MODE=...    Picker mode: 'scale' (default, 2 points + distance for cm/pixel)
#               or 'seed' (1 point on belly, propagated for SAM3 box prompt)
#
# Examples:
#   # Scale calibration (2-point mode):
#   bash scripts/run_scale_picker.sh data/input/belly.mp4 belly_scale.json
#
#   # Belly seed (1-point mode for SAM3):
#   MODE=seed bash scripts/run_scale_picker.sh data/input/belly.mp4 belly_seed.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT="${1:?Usage: $0 <image_dir_or_video> [output_path] [port]}"
OUTPUT="${2:-}"
PORT="${3:-8090}"

CONDA_BASE="${CONDA_PREFIX:-$HOME/miniconda3}"
if [ -d "${CONDA_BASE}/envs" ]; then
    :
elif [ -d "$(dirname "$(dirname "$CONDA_BASE")")/envs" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_BASE")")"
fi
PYTHON="${CONDA_BASE}/envs/leg_pipeline/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: leg_pipeline env not found at $PYTHON"
    exit 1
fi

# Detect input type
INPUT_KIND="dir"
if [ -f "$INPUT" ]; then
    case "$INPUT" in
        *.mp4|*.mov|*.avi|*.mkv|*.webm|*.m4v|*.MP4|*.MOV|*.AVI|*.MKV)
            INPUT_KIND="video"
            ;;
        *)
            echo "ERROR: file is not a recognized video format: $INPUT"
            exit 1
            ;;
    esac
elif [ ! -d "$INPUT" ]; then
    echo "ERROR: input not found: $INPUT"
    exit 1
fi

# Default output path
if [ -z "$OUTPUT" ]; then
    if [ "$INPUT_KIND" = "video" ]; then
        BASE="${INPUT%.*}"
        OUTPUT="${BASE}_scale_calibration.json"
    else
        OUTPUT="$INPUT/scale_calibration.json"
    fi
fi

# Free the port if needed
EXISTING=$(lsof -ti:$PORT 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Port $PORT already in use (pid $EXISTING). Killing..."
    kill $EXISTING 2>/dev/null || true
    sleep 1
fi

echo "============================================"
if [ "${MODE:-scale}" = "seed" ]; then
    echo "Belly Seed Picker (1 point on belly per frame)"
else
    echo "Scale Picker — Manual Per-Frame Calibration"
fi
echo "============================================"
echo "Input:        $INPUT  (${INPUT_KIND})"
echo "Mode:         ${MODE:-scale}"
echo "Output JSON:  $OUTPUT"
echo "Port:         $PORT"
echo ""
echo "Open in browser: http://localhost:$PORT"
echo "Press Ctrl+C when done."
echo "============================================"
echo ""

# Build command
CMD="$PYTHON ${PROJECT_DIR}/src/calibration/scale_picker.py --output $OUTPUT --port $PORT"
if [ -n "$MODE" ]; then
    CMD="$CMD --mode $MODE"
fi
if [ "$INPUT_KIND" = "video" ]; then
    # Only pass --n_frames if user set N_FRAMES explicitly. Otherwise let
    # the Python default (currently 30) take effect — keeping the picker and
    # orchestrator defaults in lock-step.
    CMD="$CMD --video $INPUT"
    if [ -n "$N_FRAMES" ]; then
        CMD="$CMD --n_frames $N_FRAMES"
    fi
else
    CMD="$CMD --image_dir $INPUT"
fi

$CMD
