#!/bin/bash
# Leg deformity pipeline (clean, mirrors the belly pipeline).
#
# Workflow:
#   1. Scale calibration via picker (2 clicks on a known-size object):
#        bash scripts/run_scale_picker.sh data/input/patient.mp4 \
#             data/input/patient_scale.json
#
#   2. Run the leg pipeline:
#        SCALE_CALIB=data/input/patient_scale.json \
#          bash scripts/run_leg_pipeline.sh data/input/patient.mp4 \
#               data/output/patient
#
# <input> can be a directory of images OR a video file.
#
# Environment variables:
#   SUBJECT=...       Subject preset (default: 'standing')
#   SCALE_CALIB=path  scale_calibration.json (required for measurements in cm)
#   SEED_POINTS=path  Optional seed-point JSON (usually not needed for legs)
#   RECON_MODEL=...   vggt (default) or amb3r
#   LOW_MEMORY=0/1    1 → per-frame VGGT (slower but less VRAM). Default 0.
#   N_FRAMES=N        Frames to extract from video (default 30, must match picker)
#   RECON_MAX_FRAMES=N  Cap on frames used for 3D reconstruction (default 20)
#   POSE_MODEL=...    human (default), vitpose, vitpose-l, ...
#   SAM_PROMPT=...    Override primary SAM3 prompt (default: from subject)
#   SAM_FALLBACK=...  Override fallbacks (default: from subject)
#   SAM_CONF=0.25     SAM3 confidence threshold (default 0.25)
#   USE_SEED_POINTS=0/1  Override subject default for seed-prompt usage
#   ANTERIOR_FRAME=N|name  Frame to use for single-frame 2D HKA. Recommended
#                          for unclear varus/valgus cases. Pick the most-
#                          front-facing frame with the cleanest stance.
#                          Can be an integer index (0..n-1) or filename
#                          (e.g. 'frame_009.jpg').
#   ANTERIOR_PICKER_JSON=path  Alternative — path to the JSON written by
#                              the anterior picker. The script extracts
#                              the chosen frame from it. Use this if you
#                              don't have `jq` installed.

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
        *.mp4|*.mov|*.avi|*.mkv|*.webm|*.m4v|*.MP4|*.MOV|*.AVI|*.MKV)
            INPUT_FLAG="--video $INPUT"
            ;;
        *)
            echo "ERROR: not a recognized video format: $INPUT"; exit 1
            ;;
    esac
elif [ -d "$INPUT" ]; then
    INPUT_FLAG="--image_dir $INPUT"
else
    echo "ERROR: input not found: $INPUT"; exit 1
fi

CMD="$PYTHON ${PROJECT_DIR}/src/pipeline/leg_orchestrator.py"
CMD="$CMD $INPUT_FLAG --output_dir $OUTPUT_DIR"

if [ -n "$SUBJECT" ];          then CMD="$CMD --subject $SUBJECT"; fi
if [ -n "$SCALE_CALIB" ];      then CMD="$CMD --scale_calibration $SCALE_CALIB"; fi
if [ -n "$SEED_POINTS" ];      then CMD="$CMD --seed_points $SEED_POINTS"; fi
if [ -n "$RECON_MODEL" ];      then CMD="$CMD --recon_model $RECON_MODEL"; fi
if [ -n "$N_FRAMES" ];         then CMD="$CMD --n_frames $N_FRAMES"; fi
if [ -n "$RECON_MAX_FRAMES" ]; then CMD="$CMD --recon_max_frames $RECON_MAX_FRAMES"; fi
if [ -n "$POSE_MODEL" ];       then CMD="$CMD --pose_model $POSE_MODEL"; fi
if [ -n "$SAM_PROMPT" ];       then CMD="$CMD --sam_prompt \"$SAM_PROMPT\""; fi
if [ -n "$SAM_FALLBACK" ];     then CMD="$CMD --sam_fallback_prompts \"$SAM_FALLBACK\""; fi
if [ -n "$SAM_CONF" ];         then CMD="$CMD --sam_confidence $SAM_CONF"; fi
if [ "${LOW_MEMORY:-0}" = "1" ]; then CMD="$CMD --low_memory"; fi
if [ "${USE_SEED_POINTS:-}" = "1" ]; then CMD="$CMD --use_seed_points"; fi
if [ "${USE_SEED_POINTS:-}" = "0" ]; then CMD="$CMD --no_seed_points"; fi

# Anterior-frame resolution — two ways to pass it:
#   ANTERIOR_FRAME=frame_009.jpg            (direct filename or index)
#   ANTERIOR_PICKER_JSON=...anterior.json   (let us read the picker JSON)
#
# Treat literal "null" and the empty string as unset so we don't silently
# pass a bogus value through.
if [ -z "$ANTERIOR_FRAME" ] && [ -n "$ANTERIOR_PICKER_JSON" ]; then
    if [ -f "$ANTERIOR_PICKER_JSON" ]; then
        ANTERIOR_FRAME=$("$PYTHON" -c "
import json, sys
try:
    d = json.load(open('$ANTERIOR_PICKER_JSON'))
    print(d.get('anterior_frame', '') or '')
except Exception as e:
    sys.stderr.write('Could not read picker JSON: %s\n' % e)
    sys.exit(1)
")
        echo "  → loaded anterior frame '$ANTERIOR_FRAME' from $ANTERIOR_PICKER_JSON"
    else
        echo "ERROR: ANTERIOR_PICKER_JSON not found: $ANTERIOR_PICKER_JSON"
        exit 1
    fi
fi

case "$ANTERIOR_FRAME" in
    ""|"null"|"None")
        ANTERIOR_FRAME=""
        ;;
    *)
        CMD="$CMD --anterior_frame $ANTERIOR_FRAME"
        ;;
esac

if [ -n "$ANTERIOR_FRAME" ]; then
    MODE_BANNER="ANTERIOR-FRAME MODE — primary classification = single-frame 2D HKA on '$ANTERIOR_FRAME'"
else
    MODE_BANNER=$'MULTI-FRAME 3D MODE (LEGACY)\n              ⚠ For trustworthy classification, run:\n              ⚠   bash scripts/run_anterior_picker.sh <video> <patient>_anterior.json\n              ⚠ Then re-run this pipeline with:\n              ⚠   ANTERIOR_FRAME=$(jq -r .anterior_frame <patient>_anterior.json)'
fi

echo "============================================"
echo "Leg Deformity Pipeline"
echo "============================================"
echo "Input:        $INPUT"
echo "Output:       $OUTPUT_DIR"
echo "Subject:      ${SUBJECT:-standing}"
echo "Scale calib:  ${SCALE_CALIB:-<not set — distances will be in raw units>}"
echo "Recon model:  ${RECON_MODEL:-vggt}"
echo "Low memory:   ${LOW_MEMORY:-0}  (0 = batched multi-view)"
echo "N frames:     ${N_FRAMES:-30} (extracted)  →  ${RECON_MAX_FRAMES:-20} (recon cap)"
echo "Pose model:   ${POSE_MODEL:-human}"
echo "Mode:         $MODE_BANNER"
echo "============================================"
echo ""

eval $CMD

echo ""
echo "Done. Results in: $OUTPUT_DIR"
echo "  - leg_assessment.json      (HKA, MAD, lengths, classifications)"
echo "  - pipeline_results.json    (stage-by-stage summary)"
echo "  - debug/leg/               (visualizations)"
