#!/bin/bash
# Belly analysis pipeline: 3D mesh, volume, belly button, distance to feet.
#
# Recommended 3-step workflow:
#
#   1. Capture a video of the belly from multiple angles.
#
#   2. Calibrate scale: pick 2 points on a known object (ruler, box edge),
#      enter the real distance in cm. Use "Track to all frames" to propagate.
#
#        bash scripts/run_scale_picker.sh data/input/belly_002.mp4 \
#             data/input/belly_002_scale.json
#
#   3. Pick belly seed: click 1 point on the belly bump in any frame,
#      "Track seed to all frames" to propagate.
#
#        MODE=seed bash scripts/run_scale_picker.sh data/input/belly_002.mp4 \
#             data/input/belly_002_seed.json
#
#   4. Run the pipeline:
#
#        SCALE_CALIB=data/input/belly_002_scale.json \
#        SEED_POINTS=data/input/belly_002_seed.json \
#          bash scripts/run_belly_pipeline.sh data/input/belly_002.mp4 \
#               data/output/belly_002
#
# <input> can be a directory of images OR a video file.
#
# Environment variables:
#   SUBJECT=...       'pregnant' (default, runs pose for distance-to-ground) or
#                     'balloon' (skips pose, uses balloon-specific prompts)
#   SCALE_CALIB=path  Path to scale_calibration.json (REQUIRED for metric volume)
#   SEED_POINTS=path  Path to seed_points.json. Only USED when the subject's
#                     preset enables seed prompts (pregnant=YES, balloon=NO).
#                     Override with USE_SEED_POINTS=1 (force on) or 0 (force off).
#   USE_SEED_POINTS=0/1  Override the subject's seed-prompt default
#                        (1 = force-enable, 0 = force-disable)
#   RECON_MODEL=...   vggt (default) or amb3r
#   LOW_MEMORY=0/1    0 = VGGT batched multi-view (DEFAULT — proper fusion).
#                     1 = per-frame mode (fallback for OOM; clouds will NOT
#                         align across frames since each is in its own coords).
#                     Prefer reducing N_FRAMES first if you need less VRAM.
#   N_FRAMES=N        Frames to extract from video (default 30 — must match
#                     the picker setting so seed clicks align)
#   RECON_MAX_FRAMES=N  Cap on frames used for 3D reconstruction (default 20).
#                     Pipeline takes intersection of (extracted) and
#                     (frames where SAM3 succeeded), uniformly samples this many.
#   SAM_PROMPT=...    Override primary SAM3 prompt (default: from SUBJECT)
#   SAM_FALLBACK=...  Override fallback prompts (default: from SUBJECT)
#   SAM_CONF=0.25     SAM3 confidence threshold (default 0.25)
#   POISSON_DEPTH=8   Octree depth for surface mesh (default 8)
#   CONF_PCT_KEEP=75  Keep top X% confidence points (default 75)
#
# Examples:
#   # Pregnant patient with full distance computation:
#   SCALE_CALIB=scale.json SEED_POINTS=seed.json \\
#     bash scripts/run_belly_pipeline.sh data/input/belly_001.mp4 data/output/belly_001
#
#   # Balloon test (no person; volume + belly button only):
#   SUBJECT=balloon SCALE_CALIB=scale.json SEED_POINTS=seed.json \\
#     bash scripts/run_belly_pipeline.sh data/input/balloon_test.mp4 data/output/balloon_test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT="${1:?Usage: $0 <input_dir_or_video> <output_dir>}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir_or_video> <output_dir>}"

CONDA_BASE="${CONDA_PREFIX:-$HOME/miniconda3}"
if [ -d "${CONDA_BASE}/envs" ]; then :;
elif [ -d "$(dirname "$(dirname "$CONDA_BASE")")/envs" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_BASE")")"
fi
PYTHON="${CONDA_BASE}/envs/leg_pipeline/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: leg_pipeline env not found at $PYTHON"
    exit 1
fi

# Detect input type
INPUT_FLAG=""
if [ -f "$INPUT" ]; then
    case "$INPUT" in
        *.mp4|*.mov|*.avi|*.mkv|*.webm|*.MP4|*.MOV|*.AVI|*.MKV)
            INPUT_FLAG="--video $INPUT"
            ;;
        *)
            echo "ERROR: file is not a video format: $INPUT"; exit 1
            ;;
    esac
elif [ -d "$INPUT" ]; then
    INPUT_FLAG="--image_dir $INPUT"
else
    echo "ERROR: input not found: $INPUT"; exit 1
fi

CMD="$PYTHON ${PROJECT_DIR}/src/pipeline/belly_orchestrator.py"
CMD="$CMD $INPUT_FLAG --output_dir $OUTPUT_DIR"

if [ -n "$SUBJECT" ]; then
    CMD="$CMD --subject $SUBJECT"
fi
if [ -n "$SCALE_CALIB" ]; then
    CMD="$CMD --scale_calibration $SCALE_CALIB"
fi
if [ -n "$RECON_MODEL" ]; then
    CMD="$CMD --recon_model $RECON_MODEL"
fi
if [ -n "$N_FRAMES" ]; then
    CMD="$CMD --n_frames $N_FRAMES"
fi
if [ -n "$RECON_MAX_FRAMES" ]; then
    CMD="$CMD --recon_max_frames $RECON_MAX_FRAMES"
fi
if [ -n "$SAM_PROMPT" ]; then
    CMD="$CMD --sam_prompt \"$SAM_PROMPT\""
fi
if [ -n "$SAM_CONF" ]; then
    CMD="$CMD --sam_confidence $SAM_CONF"
fi
if [ -n "$SEED_POINTS" ]; then
    CMD="$CMD --seed_points $SEED_POINTS"
fi
# Optional explicit override of the subject preset's seed-prompt default
if [ "${USE_SEED_POINTS:-}" = "1" ]; then
    CMD="$CMD --use_seed_points"
elif [ "${USE_SEED_POINTS:-}" = "0" ]; then
    CMD="$CMD --no_seed_points"
fi
if [ -n "$POISSON_DEPTH" ]; then
    CMD="$CMD --poisson_depth $POISSON_DEPTH"
fi
if [ -n "$CONF_PCT_KEEP" ]; then
    CMD="$CMD --conf_pct_keep $CONF_PCT_KEEP"
fi
if [ -n "$SAM_FALLBACK" ]; then
    CMD="$CMD --sam_fallback_prompts \"$SAM_FALLBACK\""
fi
# Default: batched multi-view (proper cross-frame fusion).
# Set LOW_MEMORY=1 only if batched mode runs OOM and you can't reduce N_FRAMES.
if [ "${LOW_MEMORY:-0}" = "1" ]; then
    CMD="$CMD --low_memory"
fi

echo "============================================"
echo "Belly Analysis Pipeline"
echo "============================================"
echo "Input:        $INPUT"
echo "Output:       $OUTPUT_DIR"
echo "Subject:      ${SUBJECT:-pregnant}"
echo "Scale calib:  ${SCALE_CALIB:-<not set — volumes will NOT be metric>}"
echo "Seed points:  ${SEED_POINTS:-<not set — full mask from text prompt only>}"
echo "Recon model:  ${RECON_MODEL:-vggt}"
echo "Low memory:   ${LOW_MEMORY:-0}  (0 = batched multi-view, 1 = per-frame fallback)"
echo "N frames:     ${N_FRAMES:-30} (extracted)  →  ${RECON_MAX_FRAMES:-20} (recon cap)"
echo "SAM prompt:   ${SAM_PROMPT:-person}"
echo "Fallbacks:    ${SAM_FALLBACK:-belly,stomach,abdomen}"
echo "============================================"
echo ""

eval $CMD

echo ""
echo "Done. Results in: $OUTPUT_DIR"
echo "  - belly/belly_pointcloud.ply        (segmented belly point cloud)"
echo "  - belly/belly_mesh.ply              (closed surface mesh)"
echo "  - belly/belly_results.json          (volume, belly button, distances)"
