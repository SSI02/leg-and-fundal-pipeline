#!/bin/bash
# Setup script for AMB3R environment (includes VGGT as bundled thirdparty)
# Official repo: https://github.com/HengyiWang/amb3r

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$PROJECT_DIR/repos/amb3r"
CONDA_BASE="$(conda info --base)"
ENV_NAME="amb3r"
ENV_DIR="${CONDA_BASE}/envs/${ENV_NAME}"
PIP="${ENV_DIR}/bin/pip"
PYTHON="${ENV_DIR}/bin/python"

echo "============================================"
echo "Setting up AMB3R environment"
echo "Repo: $REPO_DIR"
echo "============================================"

# Check if repo exists
if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: AMB3R repo not found at $REPO_DIR"
    echo "Please run: cd $PROJECT_DIR/repos && git clone https://github.com/HengyiWang/amb3r.git"
    exit 1
fi

# Create conda environment
echo ""
echo "[1/7] Creating conda environment '${ENV_NAME}' (Python 3.9)..."
conda create -n "${ENV_NAME}" python=3.9 cmake=3.14.0 -y

echo ""
echo "[2/7] Installing PyTorch 2.5.0 + CUDA 11.8..."
"${PIP}" install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "[3/7] Installing torch-scatter..."
"${PIP}" install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cu118.html

echo ""
echo "[4/7] Installing PyTorch3D v0.7.8 (this may take a while)..."
"${PIP}" install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" --no-build-isolation

echo ""
echo "[5/7] Installing flash-attn 2.7.3 (this may take a while)..."
"${PIP}" install flash-attn==2.7.3 --no-build-isolation

echo ""
echo "[6/7] Installing remaining requirements..."
cd "$REPO_DIR"
"${PIP}" install -r requirements.txt

echo ""
echo "[7/7] Downloading AMB3R checkpoint..."
mkdir -p "$REPO_DIR/checkpoints"
if [ ! -f "$REPO_DIR/checkpoints/amb3r.pt" ]; then
    echo "Downloading amb3r.pt from Google Drive..."
    "${PIP}" install gdown
    "${ENV_DIR}/bin/gdown" "14x0WW2rUE_he2hUEouP6ywSRnlJDeLel" -O "$REPO_DIR/checkpoints/amb3r.pt"
else
    echo "Checkpoint already exists at $REPO_DIR/checkpoints/amb3r.pt"
fi

echo ""
echo "============================================"
echo "AMB3R setup complete!"
echo "Activate with: conda activate ${ENV_NAME}"
echo "Test with: cd $REPO_DIR && python demo.py"
echo "============================================"
