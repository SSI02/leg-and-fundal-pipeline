# `all_requirements/` — Conda Environment Specs

Locked dependency snapshots for every conda environment this pipeline needs.
Use these on a fresh Ubuntu machine to recreate each environment exactly.

## What's here

| Folder | conda env | Used for |
|--------|-----------|----------|
| [`leg_pipeline/`](leg_pipeline/) | `leg_pipeline` | Main orchestration, calibration tools, measurement maths, visualization, the web front end (the env you run the orchestrators in) |
| [`vv_sam3/`](vv_sam3/) | `vv_sam3` | SAM3 segmentation (`src/pipeline/run_sam3.py`) |
| [`vv_vggt/`](vv_vggt/) | `vv_vggt` | VGGT 3D reconstruction (`src/pipeline/run_vggt.py`) |
| [`amb3r/`](amb3r/) | `amb3r` | AMB3R metric reconstruction (`src/pipeline/run_amb3r.py`) |
| [`pose_env/`](pose_env/) | `pose_env` | MMPose / RTMPose 2D pose estimation (`src/pipeline/run_pose.py`) |

Each folder contains **two files** in the same format as the reference
`all_requirements_scoliosis_pipeline/`:

- **`environment.yml`** — `conda env export` output: full conda + pip
  dependency tree with versions and build strings, plus the channel list. This
  is the file you actually feed to `conda env create`. The
  machine-specific `prefix:` line has been stripped for portability.
- **`requirements.txt`** — `conda list --export` output: one line per package
  (`name=version=build`). Useful for inspection, diffs, and as an alternative
  recreation route via `conda create --file`.

## Quick start — recreate every environment

Run these one at a time from the project root on a clean Ubuntu install with
Miniconda or Anaconda already set up:

```bash
conda env create -f all_requirements/leg_pipeline/environment.yml
conda env create -f all_requirements/vv_sam3/environment.yml
conda env create -f all_requirements/vv_vggt/environment.yml
conda env create -f all_requirements/amb3r/environment.yml
conda env create -f all_requirements/pose_env/environment.yml
```

Each takes 5–20 minutes depending on bandwidth (the GPU-heavy envs pull
multi-GB CUDA / PyTorch wheels).

## Recreate one environment

```bash
conda env create -f all_requirements/<env>/environment.yml
# or, if you prefer the conda list --export form:
conda create --name <env> --file all_requirements/<env>/requirements.txt
```

## Update / refresh these files (after you change an env)

```bash
# from the project root, with each env name as appropriate
for env in leg_pipeline vv_sam3 vv_vggt amb3r pose_env; do
  conda env export -n "$env" \
    | grep -v '^prefix:' \
    > "all_requirements/$env/environment.yml"
  conda list -n "$env" --export \
    > "all_requirements/$env/requirements.txt"
done
```

## Portability notes (read this before recreating on a different machine)

These specs were captured on Ubuntu Linux 64-bit with a **CUDA-capable NVIDIA
GPU** and the matching NVIDIA driver. Practical caveats when recreating
elsewhere:

- **OS / arch.** The conda packages are `linux-64`. macOS or aarch64 will not
  resolve.
- **NVIDIA driver.** A driver matching the CUDA runtime baked into each env
  (CUDA 11.8 for `amb3r` / `pose_env`, CUDA 12.x for `vv_vggt`) must be
  present on the host. The CUDA runtime is installed as a conda package, but
  the driver is not — install it via `apt` first.
- **Channel pin solvers.** Specs include build strings (e.g.
  `pytorch=2.5.1=py3.12_cuda12.1_cudnn9.1.0_0`). If a build is no longer
  available on the channel, conda will emit a solver error. In that case,
  hand-relax the offending line to a version-only pin (`pytorch=2.5.1`) and
  retry, or fall back to the corresponding `scripts/setup_<env>.sh` script
  which uses unpinned/looser specs.
- **The vendored model repos.** Several envs assume the model source trees
  exist at `repos/<vggt|amb3r|sam3|mmpose|4D-Humans>/`. The `pose_env` env in
  particular is built by `scripts/setup_pose.sh` doing
  `pip install -e repos/mmpose` — make sure that submodule is cloned before
  recreating the env.
- **Model checkpoints.** These specs install the libraries, not the
  weights. After creating an env, run the matching `scripts/setup_*.sh` (or
  the worker script's first run) to download the model checkpoints
  separately. See `knowledge_transfer/README.md` § 5 for the checkpoint
  source list.

## Sanity check after creation

```bash
conda activate leg_pipeline   && python -c "import open3d, cv2, trimesh, flask; print('leg_pipeline OK')"
conda activate vv_sam3        && python -c "import torch; print('sam3 cuda:', torch.cuda.is_available())"
conda activate vv_vggt        && python -c "import torch; print('vggt cuda:', torch.cuda.is_available())"
conda activate amb3r          && python -c "import torch; print('amb3r cuda:', torch.cuda.is_available())"
conda activate pose_env       && python -c "import mmcv, mmpose; print('pose:', mmcv.__version__)"
```

---

For the original loose setup scripts (which install with looser version pins
and download model checkpoints), see [`../scripts/setup_*.sh`](../scripts/).
For the full project documentation see
[`../knowledge_transfer/README.md`](../knowledge_transfer/README.md).
