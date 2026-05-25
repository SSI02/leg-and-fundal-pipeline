#!/bin/bash
# Setup script for HMR2.0 / 4DHumans environment (SMPL body model fitting)
# Official repo: https://github.com/shubham-goel/4D-Humans

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$PROJECT_DIR/repos/4D-Humans"
CONDA_BASE="$(conda info --base)"
ENV_NAME="hmr_env"
ENV_DIR="${CONDA_BASE}/envs/${ENV_NAME}"
PIP="${ENV_DIR}/bin/pip"

echo "============================================"
echo "Setting up HMR2.0 / 4DHumans environment"
echo "Repo: $REPO_DIR"
echo "============================================"

if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: 4D-Humans repo not found at $REPO_DIR"
    exit 1
fi

echo "[1/4] Creating conda environment '${ENV_NAME}' (Python 3.10)..."
conda create -n "${ENV_NAME}" python=3.10 -y

echo "[2/4] Installing PyTorch 2.1.0 + CUDA 12.1..."
"${PIP}" install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

echo "[3/4] Installing 4DHumans..."
cd "$REPO_DIR"
"${PIP}" install -e ".[all]"

echo "[4/4] Installing smplx + SMPL-Anthropometry deps..."
"${PIP}" install smplx==0.1.28 trimesh scipy scikit-learn

echo ""
echo "============================================"
echo "HMR2.0 setup complete!"
echo "Activate with: conda activate ${ENV_NAME}"
echo ""
echo "IMPORTANT: Download SMPL model files from https://smpl.is.tue.mpg.de"
echo "Place basicModel_neutral_lbs_10_207_0_v1.0.0.pkl in:"
echo "  $REPO_DIR/data/smpl/SMPL_NEUTRAL.pkl"
echo "============================================"
