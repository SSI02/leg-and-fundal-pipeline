# Code Documentation — Leg Deformity & Fundal Height Estimation

> **Scope.** This README is the developer-facing reference for the codebase. It documents
> every module, its functions, its inputs/outputs, the multi-environment architecture, the
> on-disk file formats, and how to run both pipelines. For the conceptual / clinical
> walkthrough see the companion **`Knowledge_Transfer.pdf`** in this folder.

---

## Table of Contents

1. [What this project is](#1-what-this-project-is)
2. [Repository layout](#2-repository-layout)
3. [The multi-environment architecture](#3-the-multi-environment-architecture)
4. [End-to-end data flow](#4-end-to-end-data-flow)
5. [Environment setup](#5-environment-setup)
6. [How to run — Leg pipeline](#6-how-to-run--leg-pipeline)
7. [How to run — Belly / Fundal pipeline](#7-how-to-run--belly--fundal-pipeline)
8. [The web front end](#8-the-web-front-end)
9. [Module reference: `src/calibration/`](#9-module-reference-srccalibration)
10. [Module reference: `src/pipeline/`](#10-module-reference-srcpipeline)
11. [Module reference: `src/measurements/`](#11-module-reference-srcmeasurements)
12. [Module reference: `src/visualization/`](#12-module-reference-srcvisualization)
13. [Scripts reference](#13-scripts-reference)
14. [Configuration files](#14-configuration-files)
15. [Output file formats](#15-output-file-formats)
16. [Clinical thresholds & constants](#16-clinical-thresholds--constants)
17. [Known issues, gotchas & dead code](#17-known-issues-gotchas--dead-code)

---

## 1. What this project is

Two medical computer-vision pipelines that turn ordinary phone **video** of a patient into
**clinical 3D measurements**:

| Pipeline | Input | Produces |
|----------|-------|----------|
| **Leg Deformity Detection** | Video of a standing patient | Hip-Knee-Ankle (HKA) angle, varus/valgus classification & severity, mechanical-axis deviation, knee/ankle gaps, leg-length discrepancy, lower-leg volumes, a 3D point cloud |
| **Fundal Height / Belly Estimation** | Video of a pregnant belly (or a test balloon) | Belly bulge volume, belly-button (apex) localization, protrusion height, belly-button-to-feet distance, a 3D mesh — all as a *proxy* for symphysis-fundal height |

Both share the same backbone: **video → frame extraction → person/belly segmentation
(SAM3) → 3D reconstruction (VGGT or AMB3R) → manual metric scale calibration → 2D pose
(MMPose) → measurement → debug visualizations**.

The project is **research / proof-of-concept** code. It is not a medical device and has no
clinical validation. The leg pipeline has the most published prior art and is the most
mature; the fundal-height pipeline is novel research and currently produces a *belly-shape
proxy*, not a validated SFH measurement.

---

## 2. Repository layout

```
leg_deformity_fundal_height/
├── configs/
│   └── default.json              # default pipeline parameters (legacy orchestrator)
├── data/
│   ├── input/                    # patient videos, *_scale.json, *_seed.json, *_anterior.json,
│   │                             #   and extracted <name>_frames/ directories
│   └── output/                   # one folder per run: <patient_or_balloon_id>/
├── front end/
│   └── app.py                    # single-file Flask web wrapper for the whole workflow
├── repos/                        # vendored model repositories (added to sys.path at runtime)
│   ├── vggt/                     # VGGT — Visual Geometry Grounded Transformer
│   ├── amb3r/                    # AMB3R — metric-scale 3D reconstruction
│   ├── sam3/                     # SAM3 — Segment Anything 3
│   ├── mmpose/                   # MMPose — 2D pose estimation
│   └── 4D-Humans/                # HMR2.0 — SMPL body-model recovery
├── scripts/                      # setup_*.sh (env install) and run_*.sh (launchers) + verify_*.py
├── src/
│   ├── calibration/              # frame extraction, scale picker, anterior picker, ArUco
│   ├── pipeline/                 # orchestrators + per-model worker scripts
│   ├── measurements/             # clinical measurement engines
│   ├── visualization/            # debug_viz.py + the viewer server
│   └── utils/                    # (empty placeholder)
├── viewer/
│   └── index.html                # standalone Three.js 3D point-cloud viewer
├── commands.txt                  # copy-paste workflow recipes (the canonical "how do I run it")
├── configs / environment.yml     # conda env spec for the pose environment
├── PIPELINE_DESIGN.md            # the original detailed design document
├── DATA_COLLECTION.md            # data-collection protocol for the fundal-height study
├── literature_review_fundal_height.md
└── research_report_leg_deformity_and_body_measurement.md
```

---

## 3. The multi-environment architecture

The single most important architectural fact: **the models have mutually incompatible
dependencies, so each runs in its own conda environment.** The orchestrators never
`import` the heavy models — they **shell out** to worker scripts, launching each with the
Python interpreter of the correct environment.

| conda env | Python | Purpose | Worker script(s) |
|-----------|--------|---------|------------------|
| `leg_pipeline` | 3.10 | Orchestration, calibration, measurements, visualization, front end. **This is the env you run the orchestrators in.** | orchestrators, `measurements/*`, `visualization/*`, calibration tools |
| `vv_sam3` | — | SAM3 segmentation | `run_sam3.py` |
| `vv_vggt` | — | VGGT reconstruction | `run_vggt.py` |
| `amb3r` | 3.9 | AMB3R reconstruction (bundles VGGT as a frontend) | `run_amb3r.py` |
| `pose_env` | 3.9 | MMPose / RTMPose pose estimation | `run_pose.py` |
| `hmr_env` | 3.10 | HMR2.0 / SMPL body-model fitting (optional, not wired into the main flow) | `run_hmr.py`, `run_hmr2.py` |

**How env switching works** (`orchestrator.py`, `leg_orchestrator.py`, `belly_orchestrator.py`):

1. `get_conda_prefix()` finds the conda base by inspecting `$CONDA_PREFIX` (climbing two
   directories to find an `envs/` folder), then falling back to `~/miniconda3`,
   `~/anaconda3`, `~/miniforge3`, `/opt/conda`, and finally `conda info --base`.
2. `run_in_env(env_name, script, args)` builds the path
   `<conda_base>/envs/<env_name>/bin/python`, runs `subprocess.run([...])`, captures
   stdout/stderr, raises `RuntimeError` on a non-zero exit.

If a worker fails, the error you see is the **last ~1000–1500 characters of that
subprocess's stderr** — to debug, re-run the worker script directly in its own env.

---

## 4. End-to-end data flow

```
                                   VIDEO  (data/input/patient_001.mp4)
                                     │
              ┌──────────────────────┼───────────────────────┐
              │  OPERATOR PRE-STEPS (interactive, browser)    │
              │  1. scale_picker.py   → patient_001_scale.json   (metric calibration)
              │  2. anterior_picker.py→ patient_001_anterior.json(leg: best front frame)
              │     OR scale_picker --mode seed → *_seed.json    (belly: SAM3 seed)
              └──────────────────────┬───────────────────────┘
                                     ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  ORCHESTRATOR  (leg_orchestrator.py | belly_orchestrator.py)          │
   │                                                                       │
   │  Stage 0  extract_frames.py     video → <name>_frames/frame_NNN.jpg    │
   │  Stage 1  run_sam3.py     [vv_sam3]   → segmentation/segmentation.json │
   │  Stage 2  frame selection             → recon_frames/  (symlinks)      │
   │  Stage 3  run_vggt.py | run_amb3r.py  → reconstruction/point_cloud.npz │
   │  Stage 4  manual_scale.py             → rescale npz to metric meters   │
   │  Stage 5  run_pose.py     [pose_env]  → pose/pose_results.json         │
   │  Stage 6  measurements/*              → leg_assessment.json |          │
   │                                         belly/belly_results.json      │
   │  Stage 7  debug_viz.py                → debug/                         │
   │                                                                       │
   │  → pipeline_results.json  (per-stage manifest)                        │
   └──────────────────────────────────────────────────────────────────────┘
```

**Three JSON files glue the operator's clicks to the pipeline:**

- `*_scale.json` — two clicked pixels + a real-world distance → **metric scale**.
- `*_anterior.json` — the single best front-facing frame for **leg classification**.
- `*_seed.json` — one clicked point on the belly → **SAM3 seed** for segmentation.

Because clicks are stored *per frame filename* (`frame_NNN.jpg`), the `--n_frames` value
**must be identical** across the picker tools and the orchestrator, or the saved clicks
will not line up with the extracted frames. `_video_frame_manifest.json` records the
mapping and the orchestrators hard-fail (`sys.exit(1)`) on a frame-count mismatch.

---

## 5. Environment setup

Run the setup scripts once each (they create conda envs and install dependencies):

```bash
bash scripts/setup_pipeline.sh   # creates  leg_pipeline  (Py 3.10) — the main env
bash scripts/setup_amb3r.sh      # creates  amb3r         (Py 3.9)  — AMB3R + VGGT
bash scripts/setup_pose.sh       # creates  pose_env      (Py 3.9)  — MMPose
bash scripts/setup_hmr.sh        # creates  hmr_env       (Py 3.10) — HMR2.0 (optional)
# vv_sam3 and vv_vggt envs must be created separately (no setup script bundled here)
```

What each installs:

- **`setup_pipeline.sh`** → `numpy scipy opencv-python opencv-contrib-python open3d
  trimesh Pillow matplotlib flask flask-cors reportlab`.
- **`setup_amb3r.sh`** → torch 2.5.0 (cu118), torch-scatter, PyTorch3D 0.7.8,
  flash-attn 2.7.3, `repos/amb3r/requirements.txt`, then downloads the `amb3r.pt`
  checkpoint via `gdown` into `repos/amb3r/checkpoints/`.
- **`setup_pose.sh`** → torch 2.1.0 (cu118), the OpenMMLab stack (`mmengine`, `mmcv>=2.0.1`,
  `mmdet>=3.1.0`) via `mim`, then `pip install -e .` for `repos/mmpose`.
- **`setup_hmr.sh`** → torch 2.1.0 (cu121), `repos/4D-Humans` (`pip install -e ".[all]"`),
  `smplx`, `trimesh`. SMPL body-model `.pkl` files must be downloaded manually.

`environment.yml` is a separate spec for the pose stack (it names the env `pose_env`,
Python 3.9, with the pinned OpenMMLab versions).

**Model checkpoints:**

| Model | Source |
|-------|--------|
| VGGT | HuggingFace `facebook/VGGT-1B` (auto-download) |
| AMB3R | local `repos/amb3r/checkpoints/amb3r.pt` (downloaded by `setup_amb3r.sh`) |
| SAM3 | local `repos/sam3/checkpoints/sam3.pt` if present, else HuggingFace |
| MMPose / RTMDet | auto-downloaded by MMPose at first run |
| HMR2.0 | auto-downloaded into `~/.cache/4DHumans` |

A CUDA GPU is required for SAM3, VGGT/AMB3R and pose. ~24 GB VRAM is comfortable; use
`LOW_MEMORY=1` / smaller `RECON_MAX_FRAMES` on tighter cards.

---

## 6. How to run — Leg pipeline

The leg workflow is **three required steps**. Skipping step 2 silently falls back to the
less-reliable legacy multi-frame mode.

```bash
# 1. Scale calibration — click 2 points on a known-size object, enter the cm
bash scripts/run_scale_picker.sh \
     data/input/patient_001.mp4 \
     data/input/patient_001_scale.json
#    Opens http://127.0.0.1:8090  — zoom, click 2 pixels, enter cm,
#    "Track to all frames", verify, then Ctrl+C the terminal.

# 2. Anterior-frame picker — click the cleanest front-facing frame
bash scripts/run_anterior_picker.sh \
     data/input/patient_001.mp4 \
     data/input/patient_001_anterior.json
#    Opens http://127.0.0.1:8091  — frames are auto-ranked best-first if a prior
#    pose run exists; click one card; writes the JSON.

# 3. Run the leg pipeline (the script reads the picker JSON for you)
ANTERIOR_PICKER_JSON=data/input/patient_001_anterior.json \
SCALE_CALIB=data/input/patient_001_scale.json \
  bash scripts/run_leg_pipeline.sh \
       data/input/patient_001.mp4 \
       data/output/patient_001
```

**Result:** `data/output/patient_001/leg_assessment.json` plus the visual report
`data/output/patient_001/debug/leg/anterior_assessment.jpg`.

The pipeline prints a banner telling you which mode it ran in:

- `ANTERIOR-FRAME MODE` — primary classification = single-frame 2D HKA (good — this is
  what a standing radiograph measures).
- `MULTI-FRAME 3D MODE (LEGACY)` — step 2 was skipped; classification is the aggregate of
  per-frame 3D measurements (less reliable; mixed/varus/valgus results).

**Useful environment-variable overrides** (read by `run_leg_pipeline.sh`):

| Var | Default | Effect |
|-----|---------|--------|
| `SUBJECT` | `standing` | subject preset (only `standing` exists) |
| `SCALE_CALIB` | — | path to `*_scale.json`; required for cm/volume output |
| `ANTERIOR_FRAME` / `ANTERIOR_PICKER_JSON` | — | the front frame (name/index, or a JSON to read it from) |
| `RECON_MODEL` | `vggt` | `vggt` (arbitrary scale) or `amb3r` (metric) |
| `LOW_MEMORY` | `0` | `1` → per-frame VGGT (less VRAM, worse fusion) |
| `N_FRAMES` | `30` | frames extracted — **must match the pickers** |
| `RECON_MAX_FRAMES` | `20` | cap on frames sent to reconstruction |
| `POSE_MODEL` | `human` | `human` (RTMPose-m) / `vitpose` / `vitpose-l` / `wholebody` |

---

## 7. How to run — Belly / Fundal pipeline

Two test modes exist: a **balloon** test (a balloon of known volume — a controlled
ground-truth target) and the **pregnant** workflow.

```bash
VIDEO=data/input/balloon_001.mp4

# 1. Scale calibration (identical tool to the leg pipeline)
bash scripts/run_scale_picker.sh "$VIDEO" data/input/balloon_001_scale.json

# 2. Belly seed — one click on the object, propagated to all frames
MODE=seed bash scripts/run_scale_picker.sh "$VIDEO" data/input/balloon_001_seed.json

# 3. Run the pipeline
SUBJECT=balloon \
SCALE_CALIB=data/input/balloon_001_scale.json \
SEED_POINTS=data/input/balloon_001_seed.json \
  bash scripts/run_belly_pipeline.sh "$VIDEO" data/output/balloon_001
```

For a real pregnant patient, omit `SUBJECT` (it defaults to `pregnant`, which also runs
pose estimation so the belly-button-to-feet distance can be computed):

```bash
SCALE_CALIB=data/input/belly_001_scale.json \
SEED_POINTS=data/input/belly_001_seed.json \
  bash scripts/run_belly_pipeline.sh data/input/belly_001.mp4 data/output/belly_001
```

**Result:** `data/output/<id>/belly/belly_results.json` plus debug images under
`data/output/<id>/debug/belly/`.

**Subject presets** (selected by `SUBJECT`, defined in `belly_orchestrator.py`):

| Subject | SAM3 prompt | Fallback prompts | Runs pose? | Seed points? |
|---------|-------------|------------------|:----------:|:------------:|
| `pregnant` | `person` | belly, stomach, abdomen | yes | yes |
| `balloon` | `balloon` | ball, round object, sphere | no | no |
| `balloon_held` | `balloon` | ball, round object, sphere | yes | no |

**Other env overrides:** `RECON_MODEL` (`vggt`/`amb3r`), `LOW_MEMORY`, `N_FRAMES`,
`RECON_MAX_FRAMES`, `SAM_PROMPT`, `SAM_FALLBACK`, `SAM_CONF` (default `0.25`),
`POISSON_DEPTH` (default `8`), `CONF_PCT_KEEP` (default `75`).

---

## 8. The web front end

`front end/app.py` is a single-file Flask app (~1875 lines, embedded HTML/CSS/JS) that
wraps the whole workflow in one browser UI — no terminal needed.

```bash
python "front end/app.py" --port 8070      # run inside the leg_pipeline env
# → open http://127.0.0.1:8070
```

It lets you: choose `leg` or `fundal`; point at a video; launch the scale / anterior /
seed pickers as embedded subprocesses (on auto-allocated ports starting 8090/8091/8092);
auto-derive the output and JSON paths; start the orchestrator as a background job; stream
its log live; and browse the results (summary cards, JSON viewer, image gallery, the 3D
viewer, debug previews).

Internally it spawns every Python tool with `sys.executable` (the same interpreter), and
on job completion runs `_validate_volume_outputs()` which **fails the job** if the expected
`leg_assessment.json` / `belly_results.json` volume fields or the volume-slab visualization
are missing.

---

## 9. Module reference: `src/calibration/`

### `extract_frames.py`

Extracts evenly-spaced frames from a video.

- **`extract_frames(video_path, output_dir, n_frames=8, prefix="frame", return_indices=False)`**
  — opens the video, computes `n_frames` evenly-spaced indices with `np.linspace`, saves
  each as `<prefix>_NNN.jpg` (JPEG quality 95), and **always** writes
  `_video_frame_manifest.json` (maps each saved frame to its original video frame index —
  essential for the scale picker's video tracking). Falls back to streaming all frames if
  video metadata is unavailable.
- **`is_video_file(path)`** — true for `.mp4 .mov .avi .mkv .webm .m4v .wmv .flv`.
- CLI default `--n_frames` is **8**, but every caller passes **30**.

### `scale_picker.py`

Interactive browser tool for **manual per-image pixel→metric calibration** (Flask).

- **Two modes:** `scale` (click 2 points + enter a real distance in cm) and `seed`
  (click 1 point on the belly bump to seed SAM3).
- **Per-image scale:** computes `scale_cm_per_pixel = real_distance / pixel_distance`
  *separately for each image*, because frames may be shot from different distances.
- **Click propagation:** "Track to all frames" propagates clicks automatically. Two
  back-ends — **video Lucas-Kanade** (`_track_video_lk`, frame-by-frame on the original
  full-rate video, `winSize=(31,31)`, `maxLevel=4`, accurate) when a video manifest
  exists; **ORB + RANSAC homography** (`nfeatures=2000`, Lowe ratio 0.7, 3 px RANSAC) for
  loose image directories, with strict sanity checks (≥25 inliers, ≥30% inlier ratio,
  ≤2 px reprojection error).
- **CLI:** `--image_dir` / `--video`, `--n_frames` (30), `--output` (required),
  `--mode` (`scale`/`seed`), `--host`, `--port` (default **8090**).
- **Routes:** `GET /`, `GET /api/files`, `GET /api/image/<name>`, `GET /api/calibrations`,
  `GET /api/source_info`, `POST /api/save`, `POST /api/track`, `POST /api/track_seed`.
- **Output:** see [§15](#15-output-file-formats).

### `anterior_picker.py`

Interactive browser **grid picker** for the single best front-facing leg frame (Flask).

- **`_auto_rank_by_hip_sep(image_dir)`** — if a prior `pose_results.json` exists for the
  patient, ranks every frame by its hip-X-separation ratio (via
  `leg_metrics.compute_frame_view_quality_2d`) so the best front-facing candidates appear
  first. Without a prior pose run, frames are shown in extraction order.
- **UI:** card grid; each card shows a thumbnail, a rank pip, and a colored badge —
  green "anterior" (`hip_sep ≥ 0.22`), gold "near-anterior" (`≥ 0.16`), red "oblique".
- **CLI:** `--image_dir` / `--video`, `--n_frames` (30), `--output` (required),
  `--host`, `--port` (default **8091**).
- **Routes:** `GET /`, `GET /frame/<name>` (downscaled thumbnail), `POST /select`.
- **Output:** `{anterior_frame, anterior_frame_idx, n_frames_total, auto_ranking_used,
  hip_sep_ratio, selected_at}`.

### `manual_scale.py`

Library (no UI) that **applies** scale-picker calibration to a 3D reconstruction.

- **`transform_point_to_recon_space(point_xy, transform)`** — re-applies VGGT/AMB3R
  preprocessing (scale + pad/crop) so a clicked original-image pixel maps onto the
  reconstruction canvas.
- **`load_manual_scale(path)`** / **`load_manual_scale_full(path)`** — read the picker JSON.
- **`compute_3d_scale_factor(amb3r_npz_path, scale_calib_path, image_names_in_order=None,
  recon_meta_path=None)`** — for each calibrated frame: projects the two clicked points
  into recon space, looks up their 3D coordinates in the per-frame point map (3×3 patch
  average), measures the 3D distance, and derives `factor = (real_cm/100) / d3_m`. Returns
  the **median** factor across frames (robust to a few bad depth lookups) plus mean/std and
  per-frame `details`. Warns loudly if mean and median diverge by >20%.
- **`apply_3d_scale_to_npz(npz_path, scale_factor, output_path=None)`** — multiplies all
  point arrays (`points`, `points_per_frame`, `depth_per_frame`) and pose translations by
  the factor; saves a compressed `.npz`.

### `aruco.py`

ArUco-marker metric calibration — an **alternative** to the manual scale picker.

- **`detect_aruco_markers(image_path, dict_type="4x4_50")`** — returns `[{id, corners,
  center}]`. Supported dicts: `4x4_50/100`, `5x5_50/100`, `6x6_50/100`, `original`.
- **`compute_scale_factor(image_dir, marker_real_size_cm, dict_type, target_marker_id=None)`**
  — `scale = marker_real_size_cm / marker_pixel_size` per detection; returns averaged
  `scale_cm_per_pixel`, `per_image_scale`, `scale_std`, `detections`.
- **`generate_aruco_marker(...)`** — renders a printable marker PNG (with a white border).
- **CLI subcommands:** `detect` and `generate`.

---

## 10. Module reference: `src/pipeline/`

This directory holds **three orchestrators** (run in `leg_pipeline`) and **six worker
scripts** (each run in its own env via subprocess).

### `leg_orchestrator.py` — the current leg pipeline driver

Clean rewrite modeled on the belly orchestrator. Accepts a video **or** an image
directory; defaults to **VGGT** reconstruction.

**CLI:** `--image_dir` | `--video` (mutually exclusive, required), `--n_frames` (30),
`--recon_max_frames` (20), `--output_dir` (required), `--scale_calibration`,
`--seed_points`, `--recon_model` (`vggt`/`amb3r`), `--subject` (`standing`),
`--sam_prompt`, `--sam_fallback_prompts`, `--sam_confidence` (0.25),
`--use_seed_points`/`--no_seed_points`, `--pose_model` (`human`), `--low_memory`,
`--skip_3d`, `--anterior_frame` (filename or integer index).

**Stages** (`run_leg_pipeline()`):

0. **Frame extraction** — video → `<name>_frames/`. Reuses existing frames if a manifest
   matches `n_frames`; **`sys.exit(1)`** on mismatch.
1. **SAM3 segmentation** — `run_sam3.py` in `vv_sam3`. **Mandatory** — exits if the env is
   missing.
2. **Frame selection** — reads `segmentation.json`, keeps frames where SAM3 found a
   person, uniformly subsamples down to `recon_max_frames`, **symlinks** them into
   `recon_frames/`.
3. **3D reconstruction** — `run_vggt.py` (`vv_vggt`) or `run_amb3r.py` (`amb3r`).
4. **Manual 3D scale** — `compute_3d_scale_factor` (median) → `apply_3d_scale_to_npz`;
   sets `metric_calibrated`.
5. **Pose** — `run_pose.py` in `pose_env`, on the **reconstruction-resolution images**
   (`amb3r_images/`), so keypoints map 1:1 onto the point map.
6. **Measurement + classification** — runs an integrity check (the `amb3r_images/` files,
   the `pose_results` keys, and the recon image order must all be identical sets), then:
   - **Anterior mode** (preferred, when `--anterior_frame` is set):
     `measure_anterior_frame_2d` on that one frame → single-frame 2D HKA.
   - Always also runs `measure_from_pose_and_pointmap` for per-frame 3D keypoints.
   - If metric-calibrated: `compute_bilateral_lower_leg_volumes` (slab ellipse fit).
   - Writes `leg_assessment.json` in one of two schemas (anterior vs. multi-frame).
7. **Debug viz** — `run_leg_debug`.

Writes `pipeline_results.json` (per-stage manifest).

### `belly_orchestrator.py` — the belly / fundal pipeline driver

**CLI:** `--image_dir` | `--video` (required), `--n_frames` (30), `--recon_max_frames`
(20), `--output_dir` (required), `--scale_calibration`, `--recon_model` (`vggt`),
`--subject` (`pregnant`/`balloon`/`balloon_held`), `--sam_prompt`,
`--sam_fallback_prompts`, `--sam_confidence` (0.25), `--seed_points`,
`--use_seed_points`/`--no_seed_points`, `--low_memory`/`--no_low_memory`,
`--poisson_depth` (8), `--conf_pct_keep` (75), `--skip_3d`.

**Stages** (`run_belly_pipeline()`): 0 frame extraction → 1 SAM3 segmentation → 1b frame
selection → 2 reconstruction → 3 apply manual 3D scale → 4 pose on recon-resolution images
(*skipped for `balloon`*) → 5 belly mesh/volume/button/distance (`measurements.belly`) →
6 debug viz → writes `pipeline_results.json`.

### `orchestrator.py` — legacy leg pipeline driver

The **older** leg orchestrator: image-directory input only (no video), defaults to
**AMB3R**, no explicit frame-selection stage, SAM3 optional. Computes both 2D
(`clinical_measurements.json`) and 3D (`clinical_measurements_3d.json`) measurements via
the older `clinical.py` / `clinical_3d.py` engines. Kept for reference; new work should
use `leg_orchestrator.py`. **CLI:** `--image_dir`, `--output_dir`, `--marker_size_cm`,
`--scale_calibration`, `--pose_model`, `--recon_model` (`amb3r`), `--max_images` (4),
`--skip_3d`, `--skip_sam3`, `--no_outlier_removal`, `--conf_threshold` (0.0).

### Worker scripts

All six follow the same skeleton: add `repos/<model>` to `sys.path`, parse
`--image_dir`/`--output_dir`, run `main()`. They require CUDA.

| Script | Env | Model | Key outputs |
|--------|-----|-------|-------------|
| **`run_vggt.py`** | `vv_vggt` | VGGT-1B (HuggingFace `facebook/VGGT-1B`) | `point_cloud.{ply,npz}`, `amb3r_images/`, `reconstruction_meta.json` — **arbitrary scale**. Resolution 518×518. `--max_images` (4), `--mode` (`pad`/`crop`), `--per_frame`. |
| **`run_amb3r.py`** | `amb3r` | AMB3R (local `amb3r.pt`) | same layout — **metric scale**. Resolution 518×392. Runs a frontend + backend pass, saves both. |
| **`run_sam3.py`** | `vv_sam3` | SAM3 | `masks/`, `overlays/`, `segmentation.json`. Multi-prompt cascade + adaptive confidence ladder (`0.40,0.25,0.15,0.08`) + quality gates + optional seed-point box prompt. Resolution 1008. |
| **`run_pose.py`** | `pose_env` | MMPose RTMPose-m + RTMDet-m (or ViTPose++) | `pose_results.json`, `vis/`. COCO-17 keypoints, with a `leg_keypoints` subset. `--model`, `--no_vis`. |
| **`run_hmr.py`** | `hmr_env` | HMR2.0 / 4D-Humans + RegNetY detector | `hmr_results.npz`, `hmr_meta.json`. SMPL: 6890 verts, 44 joints. *(not wired into the main flow)* |
| **`run_hmr2.py`** | `hmr_env` | HMR2.0 / 4D-Humans + ViTDet-H (or RegNetY) | `hmr2_results.{json,npz}`, with per-leg femur/tibia metrics. *(not wired into the main flow)* |

> `run_vggt.py` and `run_amb3r.py` deliberately write the **same NPZ key layout** and the
> same `amb3r_images/` + `reconstruction_meta.json` so every downstream stage is
> backend-agnostic. Both preserve **original input filenames** in `amb3r_images/` — a
> deliberate fix for a past frame/content-mismatch bug.

---

## 11. Module reference: `src/measurements/`

This package converts pose keypoints + 3D reconstruction into clinical numbers. It is
layered — `leg_metrics.py` is the **current authoritative engine**; `clinical.py` and
`clinical_3d.py` are the older 2D/3D paths used only by the legacy `orchestrator.py`.

> **Design philosophy.** VGGT/AMB3R metric scale is unreliable (the code estimates it can
> be ~40–67% of the true size), so the measurement code **trusts angles and ratios**
> (scale-invariant) and treats **absolute lengths** as suspect unless externally
> calibrated by the scale picker.

### `leg_metrics.py` — the current robust metrics engine (~1990 lines)

Computes per-frame 3D measurements, **aggregates across frames with the median + IQR**
(robust to outliers), and classifies with explicit margin zones and **soft (probabilistic)
classification** so measurement noise cannot flip a borderline case.

Key building blocks:

- **Geometry helpers** — `_angle_3d`, `_distance_3d`, `_point_to_line_distance_3d` (for
  Mechanical Axis Deviation), `_angle_2d`.
- **`signed_hka_deviation_3d` / `signed_hka_deviation_2d`** — HKA angle, *signed*
  deviation (**+ varus / − valgus**) and MAD. The robust 3D mode projects the knee's
  deflection onto the patient's true lateral direction (`this_hip − other_hip`), which is
  view-independent. (The older `clinical_3d.py` used a fragile cross-product Y-component.)
- **Soft classification** — `soft_class_probabilities` / `soft_classify_hka` model the
  measurement as a Gaussian `N(measured, σ²)` with σ = 2° and integrate the probability
  mass landing in each of 9 signed severity bands.
- **Hard classification** — `classify_hka_deviation` (normal / varus / valgus + severity),
  `classify_genu_alignment` (genu varum/valgum from knee vs. ankle gaps), `classify_lld`
  (leg-length discrepancy from %).
- **View quality** — `compute_frame_view_quality_2d` / `compute_view_quality_from_2d_pose`
  use **hip X-separation** (a bony, clothing-independent cue) to label a session
  anterior / oblique / side and reject frames that are too oblique for a valid HKA.
- **Reliability & uncertainty** — `compute_reliability_score` (a 0–1 product of sample
  size, IQR, threshold proximity, view quality) and `bootstrap_classification_margin`
  (1000× resample → 95% CI + class probabilities).
- **Per-frame & aggregation** — `LegFrameMeasurement` dataclass, `measure_leg_frame`,
  `aggregate_leg_frames` → `LegAggregate` (drops |deviation| > 25° as broken depth
  lookups), `build_bilateral_assessment` → `BilateralAssessment` (knee/ankle gaps, LLD,
  genu classification, an HKA-vs-stance cross-check that flags loose-clothing pose bias).
- **`measure_from_pose_and_pointmap(pose_results, points_per_frame, image_order)`** — the
  multi-frame driver: per frame picks the most-confident person, drops low-confidence and
  oblique frames, looks up 3D joints, builds per-leg measurements.
- **`measure_anterior_frame_2d(pose_frame_data, frame_name, frame_idx)`** — the
  **preferred single-frame 2D HKA** path → `AnteriorFrameAssessment`. Widens the
  classification σ for non-ideal views (oblique ×2.4) and for leg-length asymmetry.
- **Lower-leg volume** — `compute_lower_leg_volume` slices the knee→ankle region into 20
  slabs, fits a PCA ellipse to each slab's cross-section, and sums `π·a·b·slab_height`.
  Wrapped by `compute_bilateral_lower_leg_volumes` (returns `None` unless metric-calibrated).

### `clinical.py` — legacy 2D measurements

Pure-2D leg measurements from COCO-17 keypoints, used by `orchestrator.py`. Notable: its
normal HKA cutoff is **±3°** (vs. the 5° band in `leg_metrics.py`). `process_pose_results`
produces `clinical_measurements.json`.

### `clinical_3d.py` — legacy 3D measurements

Maps 2D keypoints into the AMB3R/VGGT point map and computes scale-invariant 3D
measurements. `measure_from_pointmap` produces `clinical_measurements_3d.json`. Its
`units` field is explicitly labelled `"amb3r_raw (NOT metric)"`. Superseded by
`leg_metrics.py`.

### `belly.py` — belly mesh / volume / button (~915 lines)

The geometric core of the belly pipeline. Assumes the input point cloud is already
metric-calibrated.

- **`_transform_mask_to_recon_space(...)`** — re-applies VGGT/AMB3R pad/crop so a SAM3
  mask aligns pixel-for-pixel with the point map.
- **`build_belly_pointcloud(...)`** — masks the reconstruction to belly points only,
  per-frame confidence filter (keep top `conf_pct_keep` %), two-pass cleanup (statistical
  outlier removal `nb_neighbors=30, std_ratio=1.5`, then centroid trim). → `belly_pointcloud.ply`.
- **`fit_back_plane(points, camera_position=None)`** — fits the plane closing the back of
  the belly: smallest-eigenvalue eigenvector of the point covariance, with the sign
  disambiguated toward the camera (origin).
- **`build_belly_mesh(...)`** — screened Poisson reconstruction (`depth=poisson_depth`,
  `scale=1.1`), density-trims and AABB-crops → `belly_mesh.ply`.
- **`compute_belly_volume(...)`** — produces several volume estimates: the **primary
  bulge volume** (convex hull of the cleaned cloud, sliced + capped at the back plane),
  a least-squares **sphere fit** (best for a balloon, recovers a full sphere from a
  partial capture), a PCA **ellipsoid**, the full hull, and the OBB. All in cm³ and litres.
- **`find_belly_button(...)`** — defines a height field `h(v) = (v − plane)·protrusion_dir`;
  the apex is `argmax(h)`, refined by **gradient-flow convergence** (every vertex walks
  uphill; the mode of all endpoints is the belly button). Reports `protrusion_height_cm`.
- **`find_feet_3d(...)`** + **`compute_distance_to_feet(...)`** — locate the ankles in 3D
  from pose keypoints and report the belly-button-to-mid-ankle distance.
- **`run_belly_pipeline(...)`** — the entry point called by `belly_orchestrator.py`;
  writes `belly_results.json`.

### `smpl_fit.py` — SMPL body-model fitting

Fits the SMPL parametric body model (6890 vertices, 24 joints) to a point cloud by
optimizing pose/shape/translation/scale to minimize bidirectional chamfer distance, to
recover anatomically-correct joint *centers*. Requires the SMPL `.pkl` model at
`data/body_models/smpl/SMPL_NEUTRAL.pkl`. **Not wired into the main pipelines** — provided
as an enhancement path (see `PIPELINE_DESIGN.md` Stage 4).

### `postprocess.py` — point-cloud post-processing

Used by the legacy `orchestrator.py`. `full_postprocess` runs: SAM3 mask filtering →
statistical outlier removal → normal estimation → screened Poisson reconstruction with
density filtering. Outputs `point_cloud_person.ply`, `point_cloud_clean.ply`,
`surface_mesh.ply`, `postprocess_meta.json`.

---

## 12. Module reference: `src/visualization/`

### `debug_viz.py` (~4060 lines)

A large debug-visualization library. Matplotlib runs headless (`Agg`); Open3D is imported
lazily. **Three top-level entry points:**

- **`run_leg_debug(output_dir, image_dir)`** — leg viz set. Reads `leg_assessment.json`'s
  `primary_method`: in **anterior mode** it deletes stale multi-frame artifacts and
  produces the scale overlay, segmentation overlay, point clouds, the **primary
  `anterior_assessment.jpg`**, the bilateral comparison and the lower-leg volume slabs;
  in multi-frame mode it additionally produces per-frame pose, 3D landmarks, the leg
  report, the classification chart, the HKA-per-frame chart, the HKA overlay, and the
  frame-quality dashboard.
- **`run_belly_debug(output_dir, image_dir)`** — belly viz set: scale overlay, SAM3
  overlay, point clouds + stats, mesh + back plane, the belly-button analysis (3D views,
  height field, cross-section, gradient flow, per-frame overlays), and the full belly
  scene with feet/distance lines.
- **`run_all_debug(output_dir, image_dir)`** — the legacy path used by `orchestrator.py`.

All artifacts are written under `<output_dir>/debug/` (`debug/leg/`, `debug/belly/`,
`debug/reconstruction/`, etc.). **CLI:** `--output_dir`, `--image_dir`, `--mode`
(`leg`/`leg_legacy`/`belly`/`both`) — can be re-run standalone without re-running the
whole pipeline.

### `server.py`

A small Flask server that serves the bundled `viewer/index.html` plus the files of one
output directory, and auto-opens a browser. **CLI:** `--output_dir` (required), `--port`
(8080), `--no_browser`. Routes: `GET /`, `/viewer/<file>`, `/api/files`,
`/api/file/<path>`, `/api/measurements`, `/api/pipeline_results`.

### `viewer/index.html`

A standalone Three.js single-page 3D viewer (point cloud + measurement annotations),
served by `server.py` and embedded by the front end.

---

## 13. Scripts reference

All `run_*.sh` scripts use `set -e`, derive the project root from `$BASH_SOURCE`, and
locate the `leg_pipeline` env's Python directly (`<conda_base>/envs/leg_pipeline/bin/python`).

| Script | Purpose | Key env vars |
|--------|---------|--------------|
| `run_scale_picker.sh` | Launch the scale (or seed) picker | `MODE` (`scale`/`seed`), `N_FRAMES` |
| `run_anterior_picker.sh` | Launch the anterior-frame picker | `N_FRAMES`, `PORT` |
| `run_leg_pipeline.sh` | Run `leg_orchestrator.py` | `SCALE_CALIB`, `ANTERIOR_FRAME`/`ANTERIOR_PICKER_JSON`, `RECON_MODEL`, `LOW_MEMORY`, `N_FRAMES`, `RECON_MAX_FRAMES`, `POSE_MODEL`, `SAM_*` |
| `run_belly_pipeline.sh` | Run `belly_orchestrator.py` | `SUBJECT`, `SCALE_CALIB`, `SEED_POINTS`, `RECON_MODEL`, `LOW_MEMORY`, `N_FRAMES`, `RECON_MAX_FRAMES`, `SAM_*`, `POISSON_DEPTH`, `CONF_PCT_KEEP` |
| `run_pipeline.sh` | Run the legacy `orchestrator.py` | `SCALE_CALIB`, `SKIP_3D`, `SKIP_SAM3`, `RECON_MODEL`, `POSE_MODEL`, `MAX_IMAGES`, `NO_OUTLIER` |
| `generate_aruco.sh` | Generate a printable ArUco marker | positional: output, id, size |
| `setup_pipeline.sh` | Create the `leg_pipeline` env | — |
| `setup_amb3r.sh` | Create the `amb3r` env + download checkpoint | — |
| `setup_pose.sh` | Create the `pose_env` env | — |
| `setup_hmr.sh` | Create the `hmr_env` env | — |

**Verification scripts** (run in `leg_pipeline`, reuse cached outputs — no model re-run):

- `verify_anterior_classification.py <patient_id> <frame_idx>` — re-runs the anterior-mode
  measurement + volume stages on cached data and rewrites `leg_assessment.json`.
- `verify_leg_measurement.py <patient_id>` — re-runs the multi-frame 3D measurement +
  volume stages, prints a detailed console report, and updates `leg_assessment.json`.

`commands.txt` in the repo root holds copy-paste recipes for all of the above.

---

## 14. Configuration files

### `configs/default.json`

Default parameters for the **legacy** `orchestrator.py` (the new orchestrators take their
config from CLI args / env vars instead). Notable keys: `reconstruction.model` (`amb3r`),
`reconstruction.resolution` (`[518, 392]`), `pose_estimation.model` (`human`),
`postprocessing` (`statistical_outlier_nb: 20`, `poisson_depth: 9`),
`clinical_thresholds` (`hka_normal_range_deg: 3.0`, `hka_mild_deg: 5.0`,
`hka_moderate_deg: 10.0`), `viewer.port` (8080).

### `environment.yml`

Conda spec for the **pose** environment (`pose_env`, Python 3.9) with pinned torch 2.1.0
(cu118) and the OpenMMLab stack — kept pinned because mmcv prebuilt wheels are
version-sensitive.

---

## 15. Output file formats

### Run directory layout (`data/output/<id>/`)

```
<id>/
├── segmentation/         segmentation.json, masks/, overlays/      (run_sam3.py)
├── recon_frames/         symlinks to the frames sent to reconstruction
├── reconstruction/       point_cloud.ply, point_cloud.npz,
│                         reconstruction_meta.json, amb3r_images/   (run_vggt/amb3r.py)
├── pose/                 pose_results.json, vis/                   (run_pose.py)
├── belly/                belly_pointcloud.ply, belly_mesh.ply,
│                         belly_results.json                        (belly pipeline only)
├── debug/                leg/ | belly/ | reconstruction/ | ...     (debug_viz.py)
├── leg_assessment.json   ← main clinical output (leg pipeline)
└── pipeline_results.json ← per-stage manifest
```

### `*_scale.json` (scale picker, `scale` mode)

One entry per image basename:

```json
{
  "frame_009.jpg": {
    "p1": [120, 450], "p2": [180, 450],
    "real_distance_cm": 10.0,
    "pixel_distance": 60.0,
    "scale_cm_per_pixel": 0.1667,
    "object_description": "ruler",
    "tracking": "manual | video_lk | orb_homography",
    "video_frame_index": 42
  }
}
```

### `*_seed.json` (scale picker, `seed` mode)

```json
{ "frame_000.jpg": { "p1": [310, 280], "object_description": "belly", "tracking": "manual" } }
```

### `*_anterior.json` (anterior picker)

```json
{ "anterior_frame": "frame_009.jpg", "anterior_frame_idx": 9,
  "n_frames_total": 30, "auto_ranking_used": true,
  "hip_sep_ratio": 0.27, "selected_at": "2026-05-12T18:40:11" }
```

### `point_cloud.npz` (reconstruction)

Keys: `points`, `colors`, `confidence`, `poses`, `points_per_frame (T,H,W,3)`,
`conf_per_frame (T,H,W,1)`, `images_per_frame (T,H,W,3)`; VGGT also adds
`depth_per_frame`, `intrinsic`, `extrinsic`; AMB3R adds `depth_metric` when available.

### `reconstruction_meta.json`

`model`, `num_images`, `resolution_hw`, `has_metric_depth`, `amb3r_images_dir`,
`preprocess_mode`, **`preprocess_transforms`** (per-frame pad/crop parameters — used to
re-align masks and clicks), **`image_files_in_order`** (the canonical T-axis ordering).

### `pose_results.json`

Per image: `image_path`, `num_persons`, `persons[]` with `keypoints` (17 named COCO
entries `{x, y, score, index}`), `leg_keypoints` (6 entries), `bbox`, `mean_score`.

### `leg_assessment.json` — main leg output

Two schemas, selected by `primary_method`:

- **`single_anterior_frame_2d`** (anterior mode) — `subject`, `metric_calibrated`,
  `anterior_frame_assessment` (full per-leg 2D HKA: keypoint pixels, `hka_deg`,
  `hka_deviation_deg`, `classification`, `severity`, `class_probabilities`, `note`,
  `view_quality_label`), `left`/`right` side blocks, `overall_assessment`,
  `lower_leg_volume_left`/`_right`, `per_frame_left`/`_right` (3D keypoints kept only for
  the volume viz), `notes`, `summary`.
- **`multi_frame_3d_aggregate`** (legacy mode) — full `LegAggregate` dicts for
  `left`/`right`, `intercondylar_distance_cm`, `intermalleolar_distance_cm`,
  `leg_length_difference_cm`/`_pct`, `genu_alignment_classification`, `view_quality`,
  `flags`, `summary`.

### `belly_results.json` — main belly output

`pointcloud` (`ply`, point counts), `mesh_path`, `plane_point`, `protrusion_direction`,
`volume` (`bulge_volume_cm3`/`_liters`, `sphere_volume_cm3` + radius + inlier fraction,
`ellipsoid_volume_cm3`, `obb_volume_cm3`, …), `belly_button` (`position_3d`,
`protrusion_height_cm`, `method`, gradient-flow convergence stats), `distances`
(`distance_belly_to_midfeet_cm`, ankle positions — present only when pose ran).

### `pipeline_results.json`

`output_dir`, `subject`, `stages` (one entry per stage with its inputs/outputs or an
`error`/`skipped` marker), `total_time_seconds`.

---

## 16. Clinical thresholds & constants

**HKA deviation** (degrees off a straight 180° leg; signed **+ varus / − valgus**) — the
authoritative bands are in `leg_metrics.py`:

| Band | `leg_metrics.py` | Legacy `clinical.py` / `clinical_3d.py` |
|------|------------------|------------------------------------------|
| normal | \|dev\| ≤ 5° | \|dev\| ≤ 3° |
| borderline | 5–7° | — |
| mild | 7–10° | 3–5° |
| moderate | 10–15° | 5–10° |
| severe | > 15° | > 10° |

The 5° normal band in `leg_metrics.py` is justified by anatomic literature (175°–185° HKA
is physiologically normal) and by per-frame measurement noise (~2–3° IQR). **Treat
`leg_metrics.py` as authoritative.**

Other constants: soft-classification Gaussian σ = **2°**; implausible deviation cutoff
**25°**; minimum pose keypoint score **0.30**; knee/ankle gap genu thresholds 2 / 5 / 8 cm;
leg-length-discrepancy bands 1.5 / 3 / 5 %; anterior-view hip-separation thresholds 0.16
(usable) / 0.22 (clean); lower-leg volume uses 20 slabs.

Belly constants: SAM3 confidence cascade `0.40, 0.25, 0.15, 0.08`; per-frame confidence
keep `75%`; statistical outlier removal `nb=30, std=1.5`; Poisson depth `8`;
back-plane percentile `5th`.

---

## 17. Known issues, gotchas & dead code

- **`--n_frames` must be consistent.** The scale picker, anterior picker and the
  orchestrator must all use the same value (default 30). Clicks are keyed by
  `frame_NNN.jpg`; a mismatch silently misaligns them. The orchestrators hard-fail on a
  manifest mismatch — if you change `N_FRAMES`, delete the `<name>_frames/` directory and
  re-do the picker steps.
- **Metric output requires `SCALE_CALIB`.** Without a `*_scale.json`, every `*_cm` /
  `*_cm3` value is in arbitrary reconstruction units. The orchestrators warn but do not
  fail.
- **VGGT/AMB3R scale is unreliable** even after calibration the absolute scale can be off;
  this is why the measurement code trusts angles/ratios over absolute lengths.
- **Skipping the anterior-frame step** drops the leg pipeline into the legacy multi-frame
  3D mode, which gives noticeably less reliable mixed/varus/valgus results. Watch the
  mode banner printed at the end of the run.
- **Two leg orchestrators exist.** `leg_orchestrator.py` is current; `orchestrator.py` is
  legacy (image-dir input, AMB3R default, older measurement engines). Don't confuse them.
- **HMR / SMPL fitting is not wired in.** `run_hmr.py`, `run_hmr2.py` and `smpl_fit.py`
  exist as an enhancement path but are not called by the main orchestrators.
- **Dead code:** `belly_orchestrator.run_belly_pipeline` accepts a `no_outlier_removal`
  parameter that is never used; `run_hmr.py`'s docstring advertises a `meshes/` OBJ output
  that the code never writes; `postprocess.radius_outlier_removal` is defined but unused.
- **GPU memory.** Batched VGGT fuses views correctly but uses more VRAM. `LOW_MEMORY=1`
  switches to per-frame VGGT — lower VRAM, but per-frame point clouds live in independent
  coordinate systems and overlap incorrectly when merged. Prefer lowering
  `RECON_MAX_FRAMES` first.
- **The fundal-height number is a proxy.** The belly pipeline currently outputs belly
  shape descriptors (volume, protrusion height, apex). A validated symphysis-fundal-height
  regression model still requires the paired dataset described in `DATA_COLLECTION.md`.

---

*Companion document: `Knowledge_Transfer.pdf` (conceptual & clinical walkthrough).
Original design rationale: `PIPELINE_DESIGN.md`. Data-collection protocol:
`DATA_COLLECTION.md`.*
