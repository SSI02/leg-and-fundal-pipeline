# Leg Deformity Detection & Fundal Height Estimation

Two medical computer-vision pipelines that turn ordinary phone **video** of a
patient into quantitative **3D clinical measurements** — no X-rays, depth
sensors, or specialised hardware.

| Pipeline | Input | Produces |
|----------|-------|----------|
| **Leg deformity detection** | Standing full-body video | Hip-Knee-Ankle (HKA) angle, varus/valgus classification + severity, mechanical-axis deviation, knee/ankle gaps, leg-length discrepancy, lower-leg volumes, 3D point cloud |
| **Fundal height / belly estimation** | Video of a pregnant belly (or a test balloon) | Bulge volume, belly-button apex location, protrusion height, belly-to-feet distance, watertight 3D mesh — as a **proxy** for symphysis-fundal height |

Both pipelines share a backbone of **SAM3** segmentation → **VGGT** /
**AMB3R** 3D reconstruction → manual metric scale calibration → **MMPose** 2D
pose → clinical measurement → debug visualizations. Each heavy model lives in
its own conda environment; the orchestrators shell out to per-model worker
scripts.

> [!IMPORTANT]
> **Status: research / proof-of-concept.** This is not a registered medical
> device, has no clinical validation, and **must not be used for diagnosis or
> treatment decisions**. Every number it produces is an estimate. The
> fundal-height pipeline currently outputs a belly-shape *proxy*, not a
> validated SFH measurement.

## Documentation

Start here, in order:

1. **[`knowledge_transfer/Knowledge_Transfer.pdf`](knowledge_transfer/Knowledge_Transfer.pdf)** —
   the conceptual / clinical walkthrough (34 pages). Read this first if you're
   new to the project.
2. **[`knowledge_transfer/README.md`](knowledge_transfer/README.md)** — the
   full code-level reference: every module, function, JSON schema, CLI option,
   environment variable, threshold, output file format.
3. **[`PIPELINE_DESIGN.md`](PIPELINE_DESIGN.md)** — the original design
   rationale (long).
4. **[`commands.txt`](commands.txt)** — copy-paste recipes for every common
   task.
5. **[`DATA_COLLECTION.md`](DATA_COLLECTION.md)** — protocol for collecting
   the paired dataset still needed to validate fundal-height estimation.

## Repository layout

```
src/                  pipeline source (orchestrators, workers, measurement engines, viz)
front end/app.py      single-file Flask web wrapper for the whole workflow
scripts/              setup_*.sh / run_*.sh / verify_*.py
configs/              legacy default parameters
viewer/               standalone Three.js 3D viewer
repos/                vendored model source trees (git submodules — see below)
knowledge_transfer/   the docs pack referenced above
```

## Vendored models (submodules)

The model source trees under `repos/` are tracked as **git submodules**
pinned to specific upstream commits. Clone with submodules included:

```bash
git clone --recurse-submodules https://github.com/SSI02/leg-and-fundal-pipeline.git
# or, after a plain clone:
git submodule update --init --recursive
```

| Submodule | Upstream |
|-----------|----------|
| `repos/vggt` | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| `repos/amb3r` | [HengyiWang/amb3r](https://github.com/HengyiWang/amb3r) |
| `repos/sam3` | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) |
| `repos/mmpose` | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) |
| `repos/4D-Humans` | [shubham-goel/4D-Humans](https://github.com/shubham-goel/4D-Humans) |

Each submodule respects its own upstream license.

## Environment setup

Run the setup scripts once each — they create the required conda envs and
install pinned dependencies (a CUDA GPU is required):

```bash
bash scripts/setup_pipeline.sh   # leg_pipeline (Py 3.10) — the main env
bash scripts/setup_amb3r.sh      # amb3r (Py 3.9)         — AMB3R + VGGT + checkpoint
bash scripts/setup_pose.sh       # pose_env (Py 3.9)      — MMPose
bash scripts/setup_hmr.sh        # hmr_env (Py 3.10)      — HMR2.0 (optional)
# vv_sam3 and vv_vggt envs are created separately (no bundled setup script)
```

See [`knowledge_transfer/README.md` §5](knowledge_transfer/README.md#5-environment-setup)
for what each script installs and where model checkpoints come from.

## Running a case

The fastest path is `commands.txt` (copy-paste recipes). The web front end
gives the same workflow in a browser:

```bash
python "front end/app.py" --port 8070   # then open http://127.0.0.1:8070
```

For the command-line workflow, see
[`knowledge_transfer/README.md` §6–§7](knowledge_transfer/README.md#6-how-to-run--leg-pipeline)
or the [Knowledge Transfer PDF §9](knowledge_transfer/Knowledge_Transfer.pdf).

## Data, weights & PHI

- Patient videos, extracted frames, picker JSONs, and pipeline outputs are
  **not** included in this repository. The `.gitignore` blocks the entire
  `data/` tree, every `*.mp4`/`*.mov`/etc., model-weight extensions, and
  reconstruction artefacts.
- Model checkpoints are downloaded by the setup scripts or auto-fetched on
  first run by VGGT / MMPose / HMR.
- See `DATA_COLLECTION.md` for the protocol used to collect the
  fundal-height paired dataset.
