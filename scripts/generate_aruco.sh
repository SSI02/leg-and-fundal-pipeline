#!/bin/bash
# Generate a printable ArUco marker for metric calibration.
#
# Usage:
#   bash scripts/generate_aruco.sh [output_path] [marker_id] [size_pixels]
#
# Default: generates aruco_marker.png (ID=0, 4x4_50 dict, 200px)
# Print at exactly 10cm x 10cm for the default calibration setup.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

OUTPUT="${1:-${PROJECT_DIR}/data/aruco_marker.png}"
MARKER_ID="${2:-0}"
SIZE="${3:-200}"

eval "$(conda shell.bash hook)"
conda activate leg_pipeline

python "${PROJECT_DIR}/src/calibration/aruco.py" generate \
    --id "$MARKER_ID" \
    --size "$SIZE" \
    --output "$OUTPUT"

echo ""
echo "IMPORTANT: Print the marker at exactly 10cm x 10cm."
echo "Then pass --marker_size_cm 10.0 to the pipeline."
