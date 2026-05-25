# Prompt for the new Claude Code chat

> Copy everything between the `===` lines below as your first message in a fresh chat.
> The new chat will have no memory of the previous one — this prompt carries the full
> context across.

---

```
============================================================================
TASK: Help me clean and publish this codebase to GitHub.

CONTEXT
=======

This repo holds two medical computer-vision pipelines that share a backbone of
video -> SAM3 segmentation -> VGGT/AMB3R 3D reconstruction -> manual metric
scale calibration -> MMPose 2D pose -> clinical measurement -> debug viz:

  1) Leg deformity detection (HKA angle, varus/valgus classification,
     mechanical-axis deviation, knee/ankle gaps, leg-length discrepancy,
     lower-leg volumes) — driven by src/pipeline/leg_orchestrator.py.
  2) Fundal-height / belly estimation (belly bulge volume, belly-button apex,
     protrusion height, belly-to-feet distance) — driven by
     src/pipeline/belly_orchestrator.py. Currently produces a belly-shape
     PROXY, not a validated SFH measurement (the regression model needs
     paired patient data that hasn't been collected yet).

The architecture is multi-environment: each heavy model lives in its own conda
env and the orchestrators shell out to worker scripts. The envs are
`leg_pipeline` (main / orchestration), `vv_sam3`, `vv_vggt`, `amb3r`,
`pose_env`, and the optional `hmr_env`.

A previous chat produced a comprehensive knowledge-transfer pack in this same
repo under `knowledge_transfer/`:

  - knowledge_transfer/README.md            — full code reference (every
                                              module, function, JSON schema,
                                              CLI option, env var, threshold)
  - knowledge_transfer/Knowledge_Transfer.pdf — 34-page conceptual / clinical
                                                walkthrough
  - knowledge_transfer/NEW_CHAT_PROMPT.md   — this prompt (you can ignore it)

Other authoritative docs already in the repo root:
  - PIPELINE_DESIGN.md      — original design rationale (long)
  - DATA_COLLECTION.md      — protocol for the fundal-height dataset
  - commands.txt            — copy-paste workflow recipes (REFERENCES REAL
                              PATIENT IDS — see "PHI" warning below)
  - configs/default.json    — legacy orchestrator defaults
  - environment.yml         — pose_env conda spec

PLEASE READ knowledge_transfer/README.md FIRST. It documents the whole codebase
and will let you make informed decisions about what is code vs. what is
data/weights/outputs.

REPOSITORY LAYOUT (high-level)
==============================

  src/                      ← the project code (KEEP all of it)
    calibration/  pipeline/  measurements/  visualization/  utils/
  front end/app.py          ← Flask web wrapper (KEEP)
  scripts/                  ← setup_*.sh, run_*.sh, verify_*.py (KEEP)
  configs/                  ← KEEP
  viewer/index.html         ← KEEP (standalone Three.js viewer)
  knowledge_transfer/       ← KEEP (the docs pack)
  *.md  environment.yml     ← KEEP (the design docs)

  repos/                    ← vendored model source trees (vggt/, amb3r/,
                              sam3/, mmpose/, 4D-Humans/). The SETUP SCRIPTS
                              expect these. DO NOT silently delete them.
                              They may contain their own .git directories;
                              check before pushing — see "Decisions" below.

  data/                     ← MIX of patient PHI and small test assets:
    input/*.mp4             ← patient/balloon VIDEOS — likely PHI, REMOVE
    input/*_frames/         ← extracted frames — REMOVE
    input/*.json            ← picker click files referencing those videos
                              — REMOVE
    input/*.jpg.jpeg, aruco.png ← small static assets — discuss with user
    output/*                ← all pipeline run outputs — REMOVE entirely

  Model weights to find and remove:
    repos/amb3r/checkpoints/amb3r.pt      (~ large file)
    repos/sam3/checkpoints/*              (if present)
    Any *.pt, *.pth, *.pkl, *.ckpt, *.safetensors, *.bin under repos/
    The HuggingFace / 4D-Humans caches live OUTSIDE the repo
    (~/.cache/4DHumans, ~/.cache/huggingface) — do NOT touch those.

  Always-remove caches:
    **/__pycache__/         ← Python bytecode caches
    **/*.pyc, **/*.pyo

THE WORK YOU NEED TO DO
=======================

The user is about to copy this codebase into a NEW folder (a clean working
copy) and wants to:
  1. Remove all model weights, run outputs, extracted frames, and patient
     videos from the new folder.
  2. Add a proper .gitignore, top-level README, and LICENSE.
  3. Initialise git in the new folder and push to GitHub.

Before doing anything destructive, CONFIRM with the user:

  A. Which folder is the working copy? (cd there before doing anything)
  B. Have they already done the copy, or do you need to help with that?
  C. Vendored repos under `repos/`: keep them in the new git repo (large but
     self-contained), remove entirely (rely on the setup_*.sh scripts to
     re-clone — but those scripts currently assume `repos/<x>/` already
     exists), or convert to git submodules? Recommend SUBMODULES if each
     vendored repo has its own upstream — check each `repos/*/.git` and pick
     a default proposal to suggest.
  D. The static test images data/aruco.png and data/2026*.jpg.jpeg —
     keep as test fixtures or remove?
  E. commands.txt — it references real patient IDs (patient_001..007). Even
     though it's just filenames, the user should decide whether to scrub
     them or keep as-is. Suggest scrubbing to patient_NNN placeholders.
  F. Existing GitHub remote already configured, or create a new repo? If new:
     name, visibility (public / private), license (MIT / Apache-2.0 / other).
  G. The bundled docs reference internal file paths like
     `<absolute path>/leg_deformity_fundal_height/` — fine to leave, or want
     them genericised?

STEPWISE PLAN — do them in this order, asking before each destructive step:

  1. Verify you're in the new working copy (NOT the original) — confirm with
     `pwd` and show it to the user. Refuse to proceed if the path looks like
     the original location.
  2. Read knowledge_transfer/README.md to ground yourself in what's code vs.
     not.
  3. Inventory what would be removed: list (a) every file under data/output,
     data/input/*_frames, data/input/*.mp4, data/input/*.json; (b) every
     model checkpoint matched by the glob patterns above (use `find . -size
     +50M` to also catch any stray big files); (c) every __pycache__. Show
     the user the TOTAL count + total size before deleting anything.
  4. Get explicit user approval, then delete.
  5. Resolve the repos/ question per the user's choice (keep / remove /
     submodule). If submodules: detect each repo's upstream from
     `repos/<x>/.git/config`, remove `repos/<x>`, `git submodule add` the
     upstream URL at the same path. If removing: update setup_*.sh to git-
     clone the upstream during setup (don't just delete and break setup).
  6. Write a thorough .gitignore covering: model-weight extensions, run
     outputs, extracted frames, patient videos, picker click JSONs,
     __pycache__, .env, conda envs, IDE folders. Make sure it would block
     re-adding anything you just deleted.
  7. Write a top-level README.md that introduces the project briefly and
     points at knowledge_transfer/Knowledge_Transfer.pdf for the conceptual
     overview and knowledge_transfer/README.md for the code reference. Do
     not duplicate their content. Include a "Status: research / not a
     medical device" disclaimer prominently.
  8. Add a LICENSE file matching the user's choice.
  9. `git init` (if not already a repo), make a clean initial commit (or
     a small series of logical commits — the user prefers logical commits).
     Run `git status` and show the user before committing.
 10. Help create the GitHub repo (`gh repo create` if `gh` is available;
     otherwise give the user the manual steps). Push.

SAFETY RULES
============

  - You are operating on a USER'S CODEBASE. Confirm before every `rm`,
    `git rm`, `git push`, or any other destructive action. Show the user
    the list of files first.
  - Never `rm -rf` anything matched by a glob without showing the matches
    first.
  - Refuse to operate on the ORIGINAL folder (the user's source location for
    `leg_deformity_fundal_height/`) — only on the user's working copy.
  - PHI: patient videos and extracted frames are sensitive. If anything
    might still contain identifying info (faces, EXIF GPS, hospital
    metadata) after cleanup, flag it.
  - Don't blanket-skip pre-commit hooks (no --no-verify) and don't
    force-push.
  - If the new folder happens to be the same as the original, STOP and
    ask the user before continuing — they may not have done the copy yet.

START BY:
  1. Asking the user for the new working-copy path (Decision A).
  2. cd-ing there and running `pwd && ls -la` to confirm.
  3. Reading knowledge_transfer/README.md.
  4. Walking through Decisions B–G with the user before any deletion.
============================================================================
```

---

**How to use this:**

1. Copy the codebase to a new folder, e.g. `~/leg_pipeline_public/`.
2. Start a fresh Claude Code session in that new folder (or anywhere — the new chat will `cd` based on the prompt).
3. Paste everything inside the ``` block above as your first message.
4. The new chat will walk you through the decisions (vendored repos, license, GitHub repo) and then do the cleanup + push, confirming before each destructive step.

**If something needs tweaking** before you paste — for example you've already decided to use Apache-2.0, you want submodules, and you want a public repo — you can simply append `"My choices: license=Apache-2.0, vendored repos=submodules, visibility=public, scrub patient IDs=yes"` to the bottom of the prompt to skip the question round-trip.
