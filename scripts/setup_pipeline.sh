#!/bin/bash
# Setup script for the main orchestration pipeline environment
# This env handles: calibration, measurements, visualization, orchestration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_BASE="$(conda info --base)"
ENV_NAME="leg_pipeline"
ENV_DIR="${CONDA_BASE}/envs/${ENV_NAME}"
PIP="${ENV_DIR}/bin/pip"
PYTHON="${ENV_DIR}/bin/python"

echo "============================================"
echo "Setting up Pipeline Orchestration environment"
echo "============================================"

# Create conda environment
echo ""
echo "[1/2] Creating conda environment '${ENV_NAME}' (Python 3.10)..."
conda create -n "${ENV_NAME}" python=3.10 -y

# Install using the env's pip directly (avoids conda activate issues in scripts)
echo ""
echo "[2/2] Installing dependencies..."
"${PIP}" install \
    numpy \
    scipy \
    opencv-python \
    opencv-contrib-python \
    open3d \
    trimesh \
    Pillow \
    matplotlib \
    flask \
    flask-cors \
    reportlab

echo ""
echo "============================================"
echo "Pipeline Orchestration setup complete!"
echo "Activate with: conda activate ${ENV_NAME}"
echo "============================================"
