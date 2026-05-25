#!/bin/bash
# Setup script for Pose Estimation environment (MMPose + RTMPose/ViTPose++)
# Official repo: https://github.com/open-mmlab/mmpose

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$PROJECT_DIR/repos/mmpose"
CONDA_BASE="$(conda info --base)"
ENV_NAME="pose_env"
ENV_DIR="${CONDA_BASE}/envs/${ENV_NAME}"
PIP="${ENV_DIR}/bin/pip"
PYTHON="${ENV_DIR}/bin/python"
MIM="${ENV_DIR}/bin/mim"

echo "============================================"
echo "Setting up Pose Estimation environment"
echo "Repo: $REPO_DIR"
echo "============================================"

# Check if repo exists
if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: MMPose repo not found at $REPO_DIR"
    echo "Please run: cd $PROJECT_DIR/repos && git clone https://github.com/open-mmlab/mmpose.git"
    exit 1
fi

# Create conda environment
echo ""
echo "[1/5] Creating conda environment '${ENV_NAME}' (Python 3.9)..."
conda create -n "${ENV_NAME}" python=3.9 -y

echo ""
echo "[2/5] Installing PyTorch 2.1.0 + CUDA 11.8..."
"${PIP}" install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "[3/5] Installing OpenMMLab dependencies..."
"${PIP}" install -U openmim
"${MIM}" install mmengine
"${MIM}" install "mmcv>=2.0.1"
"${MIM}" install "mmdet>=3.1.0"

echo ""
echo "[4/5] Installing MMPose from source..."
cd "$REPO_DIR"
"${PIP}" install -r requirements.txt
"${PIP}" install -v -e .

echo ""
echo "[5/5] Installing additional dependencies for our pipeline..."
"${PIP}" install opencv-python-headless numpy scipy

echo ""
echo "============================================"
echo "Pose Estimation setup complete!"
echo "Activate with: conda activate ${ENV_NAME}"
echo ""
echo "Quick test:"
echo "  conda activate ${ENV_NAME}"
echo "  python -c \"from mmpose.apis import MMPoseInferencer; print('MMPose OK')\""
echo ""
echo "Available models (auto-downloaded on first use):"
echo "  - 'human' : RTMPose-m body 17 keypoints (RECOMMENDED)"
echo "  - 'vitpose' : ViTPose-B body 17 keypoints"
echo "  - 'vitpose-l' : ViTPose-L body 17 keypoints"
echo "  - 'wholebody' : RTMPose-m wholebody 133 keypoints"
echo "============================================"
