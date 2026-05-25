# Detailed Pipeline Design: Leg Deformity Detection & Fundal Height Estimation

---

## Table of Contents
1. [Technology Selection & Justification](#1-technology-selection--justification)
2. [Pipeline 1: Leg Deformity Detection](#2-pipeline-1-leg-deformity-detection)
3. [Pipeline 2: Fundal Height Estimation](#3-pipeline-2-fundal-height-estimation)
4. [Shared Infrastructure](#4-shared-infrastructure)
5. [Key Papers & References](#5-key-papers--references)

---

## 1. Technology Selection & Justification

### 1.1 Why VGGT + AMB3R (Not VGGT Alone)

**VGGT** (Meta, CVPR 2025 Best Paper) is a 1.2B-parameter feed-forward transformer that produces dense 3D point maps, depth maps, camera poses, and point tracks from 1-200+ images in a single forward pass (0.04s for 1 image, 3.12s for 100). However, **VGGT does NOT produce metric-scale output** — it normalizes to an arbitrary coordinate system.

**AMB3R** (CVPR 2026) solves this by using VGGT as a frozen front-end and adding a lightweight metric scale head + sparse voxel backend (Point Transformer v3). It achieves:
- 1.7% relative depth error (vs 3.7% for MASt3R)
- SOTA in camera pose, depth, and metric-scale estimation
- Essentially "VGGT with real-world measurements"

### 1.2 Alternatives Considered

| Method | Metric Scale | Multi-View | Speed | Why Not Primary |
|--------|:---:|:---:|:---:|:---|
| **Depth Anything v3** | Yes | Yes | Fast | 44% better pose than VGGT, but newer/less validated; strong backup option |
| **MASt3R** | Yes (approx) | Pairs only | ~10s | O(N²) pair processing; metric scale unreliable across views |
| **MUSt3R** | Partial | Yes | Fast | Non-commercial license; less metric accuracy |
| **Depth Pro** (Apple) | Yes | Single only | 0.3s | No multi-view fusion; best for single-image fallback |
| **Metric3D v2** | Yes | Single only | Fast | Single-image only; excellent for supplementary depth |
| **MoGe-2** (Microsoft) | Yes | Single only | 60ms | Single-image; strong for rapid screening |
| **COLMAP** | With ArUco | Yes | Minutes-hours | Too slow for clinical workflow |

### 1.3 Final Technology Stack

| Component | Primary Choice | Backup/Alternative |
|-----------|---------------|-------------------|
| **3D Reconstruction** | VGGT + AMB3R | Depth Anything v3 |
| **Metric Scale** | AMB3R scale head + ArUco markers | Depth Pro / MoGe-2 |
| **2D Pose Estimation** | Sapiens-2B (308 keypoints) | RTMPose (real-time) / ViTPose++ |
| **3D Body Model** | SMPL-X + TokenHMR/SMPLer-X | HMR2.0 / WHAM (video) |
| **Body Measurements** | SMPL-Anthropometry | Custom landmark-based |
| **Visualization** | Potree (web) + 3D Slicer (desktop) | Three.js custom viewer |
| **Surface Mesh** | Poisson reconstruction / SuGaR | Screened Poisson via Open3D |

---

## 2. Pipeline 1: Leg Deformity Detection

### 2.1 Overview

**Goal:** From 1-5 images of a standing person, estimate:
- Person height
- Gap between knees (intercondylar distance) and between ankles (intermalleolar distance)
- Postural abnormalities (varus/valgus, genu recurvatum, limb length discrepancy)
- Hip-Knee-Ankle (HKA) angle
- Generate a 3D point cloud for doctor analysis

**Clinical Context:**
- **Genu Varum (bow-legged):** HKA < 0° (mechanical axis lateral to knee center)
- **Genu Valgum (knock-kneed):** HKA > 0° (mechanical axis medial to knee center)
- **Normal HKA:** 0° ± 3°
- **Clinical threshold:** Deviations > 3° warrant further investigation

### 2.2 Image Capture Protocol

```
CAPTURE REQUIREMENTS:
├── Patient standing upright, feet together or shoulder-width apart
├── Full body visible (head to feet)
├── Views needed:
│   ├── REQUIRED: Front view (anterior-posterior)
│   ├── RECOMMENDED: Side view (lateral, 90°)
│   └── OPTIONAL: 45° oblique views (2 additional)
├── Calibration:
│   ├── ArUco marker (10×10 cm printed) placed on floor near patient's feet
│   ├── OR known-height reference object in frame
│   └── OR patient's self-reported height as fallback
├── Lighting: Even, minimal shadows
├── Background: Plain/uncluttered preferred
└── Distance: 2-4 meters from patient
```

**Why multiple views help:** A single frontal image can measure varus/valgus angle (2D projection of HKA). Adding a lateral view enables detection of genu recurvatum (hyperextension) and sagittal plane deformities. Oblique views improve 3D reconstruction quality for the point cloud.

### 2.3 Detailed Pipeline Stages

```
INPUT: 1-5 RGB images of standing person + optional ArUco marker
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1: Preprocessing & Person Detection          │
│  ─────────────────────────────────────────           │
│  1a. Person detection (YOLOv8/RTMDet)               │
│  1b. Person segmentation (SAM2 / Sapiens-Seg)       │
│  1c. Image quality validation                       │
│  1d. ArUco marker detection (OpenCV aruco module)    │
│  Output: Cropped person images + segmentation masks  │
│          + ArUco pose (if present)                   │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: 2D Pose Estimation                        │
│  ──────────────────────────                         │
│  Model: Sapiens-2B (308 keypoints, Meta 2024)       │
│    - Detects: hip, knee, ankle, toe, heel           │
│    - Also: femoral condyles, tibial plateau,        │
│      malleoli (if 308-keypoint model used)          │
│  Fallback: ViTPose++ (17 COCO keypoints)            │
│    - Sufficient for basic HKA if 308-kp unavailable │
│  Output: 2D keypoint coordinates + confidence       │
│          per image per person                        │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 3: 3D Reconstruction (Point Cloud)           │
│  ────────────────────────────────────               │
│  Path A — Multi-image (2-5 images):                 │
│    3a. VGGT forward pass → point maps, depth maps,  │
│        camera poses for all views                   │
│    3b. AMB3R scale head → metric scale factor        │
│    3c. Fuse point maps into unified point cloud     │
│    3d. Apply person segmentation mask to isolate     │
│        body points from background                  │
│    3e. Optional: Poisson surface reconstruction      │
│        for smooth mesh                              │
│                                                     │
│  Path B — Single image:                             │
│    3a. Depth Pro / MoGe-2 → metric depth map        │
│    3b. Back-project to 3D using estimated intrinsics │
│    3c. Apply person mask                            │
│    3d. Result: single-view metric point cloud        │
│        (partial, front-facing only)                 │
│                                                     │
│  Output: Metric-scale 3D point cloud of person      │
│          (PLY/PCD format)                           │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 4: 3D Body Model Fitting                     │
│  ──────────────────────────────                     │
│  4a. Run TokenHMR / SMPLer-X on input images        │
│      → SMPL-X mesh + pose parameters                │
│  4b. Align SMPL-X mesh to metric point cloud        │
│      (ICP registration + scale alignment)           │
│  4c. Extract SMPL-X joint positions in metric space │
│      → 3D coordinates of: hip_L, hip_R, knee_L,    │
│        knee_R, ankle_L, ankle_R                     │
│  4d. Run SMPL-Anthropometry to extract:             │
│      → Height, leg length, thigh length,            │
│        shank length, segment ratios                 │
│  Output: Fitted SMPL-X mesh + joint 3D positions    │
│          + body measurements (cm)                   │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 5: Clinical Measurements                     │
│  ──────────────────────────────                     │
│  From 3D joint positions (SMPL-X aligned):          │
│                                                     │
│  5a. HKA Angle (per leg):                           │
│      hip_center → knee_center → ankle_center        │
│      angle = arccos(dot(v_femur, v_tibia))          │
│      Normal: 180° ± 3° (0° deviation)              │
│      Varus: angle < 177° (deviation > +3°)          │
│      Valgus: angle > 183° (deviation < -3°)         │
│                                                     │
│  5b. Mechanical Axis Deviation (MAD):               │
│      Line from hip center to ankle center           │
│      Perpendicular distance from this line to       │
│      knee center, in mm                             │
│      Normal: < 10mm                                 │
│                                                     │
│  5c. Intercondylar Distance (knee gap):             │
│      Distance between medial femoral condyles       │
│      (knee_L to knee_R medial surface)              │
│      Measured from SMPL mesh or pose keypoints      │
│                                                     │
│  5d. Intermalleolar Distance (ankle gap):           │
│      Distance between medial malleoli               │
│      (ankle_L to ankle_R medial surface)            │
│                                                     │
│  5e. Tibiofemoral Angle:                            │
│      Angle between femoral and tibial anatomical    │
│      axes in the coronal plane                      │
│      Normal: 5-7° valgus                            │
│                                                     │
│  5f. Person Height:                                 │
│      From SMPL-Anthropometry OR                     │
│      vertex_top_of_head.z - vertex_bottom_of_foot.z │
│      in metric point cloud                          │
│                                                     │
│  5g. Leg Length (per side):                          │
│      hip_center → knee_center distance +            │
│      knee_center → ankle_center distance            │
│      Compare L vs R for limb length discrepancy     │
│                                                     │
│  5h. Genu Recurvatum Check (if lateral view):       │
│      Sagittal plane angle at knee                   │
│      Normal: 0-5° flexion                           │
│      Recurvatum: > 5° hyperextension                │
│                                                     │
│  5i. Foot Progression Angle (if sufficient views):  │
│      Angle of foot relative to direction of gait    │
│      In-toeing vs out-toeing                        │
│                                                     │
│  Output: Structured clinical measurements JSON      │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 6: Abnormality Classification                │
│  ─────────────────────────────────                  │
│  Rule-based + ML classification:                    │
│                                                     │
│  6a. Rule-based flags:                              │
│      IF HKA_deviation > 3°: flag varus/valgus       │
│      IF MAD > 10mm: flag mechanical axis deviation   │
│      IF |leg_L - leg_R| > 10mm: flag LLD            │
│      IF recurvatum > 5°: flag hyperextension        │
│      IF intercondylar_dist > threshold: flag        │
│                                                     │
│  6b. Severity grading:                              │
│      Mild: 3-5° HKA deviation                       │
│      Moderate: 5-10° HKA deviation                  │
│      Severe: > 10° HKA deviation                    │
│                                                     │
│  6c. Optional ML classifier:                        │
│      Train on clinical dataset to classify:         │
│      Normal / Genu Varum / Genu Valgum /            │
│      Genu Recurvatum / Mixed deformity              │
│      Input: measurement vector from Stage 5         │
│      Model: Simple MLP or gradient boosting         │
│                                                     │
│  Output: Classification + severity + confidence     │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 7: Visualization & Report Generation         │
│  ──────────────────────────────────────             │
│  7a. 3D Point Cloud Viewer (Potree/Three.js web):   │
│      - Colored point cloud of full body             │
│      - Overlaid skeleton with joint positions       │
│      - Mechanical axis lines drawn                  │
│      - Angle annotations at knee joints             │
│      - Interactive measurement tools (distance,     │
│        angle, cross-section)                        │
│      - Doctor can rotate, zoom, measure             │
│                                                     │
│  7b. 2D Annotated Images:                           │
│      - Original images with skeleton overlay        │
│      - HKA angle annotation                        │
│      - Mechanical axis lines                        │
│      - Color-coded severity (green/yellow/red)      │
│                                                     │
│  7c. Clinical Report (PDF):                         │
│      - Patient info                                 │
│      - All measurements in table format             │
│      - Normal ranges for comparison                 │
│      - Annotated images                             │
│      - 3D viewer link                               │
│      - Classification result + confidence           │
│      - Recommendation (further imaging, referral)   │
│                                                     │
│  Output: Web viewer URL + PDF report                │
└─────────────────────────────────────────────────────┘
```

### 2.4 Expected Accuracy

Based on published literature:

| Measurement | Expected Accuracy | Gold Standard Comparison |
|-------------|------------------|------------------------|
| HKA angle | 1.5-3° MAE | Radiographic HKA |
| Height | ±1-3 cm | Stadiometer |
| Intercondylar distance | ±5-10 mm | Physical measurement |
| Leg length | ±5-10 mm | Radiographic measurement |
| Varus/Valgus classification | 75-85% | Clinical diagnosis |

**Key references for accuracy claims:**
- OpenPose achieves 1.58° absolute error for HKA vs radiography in walking videos (Scientific Reports, 2025)
- MORA Vu software correlates r=0.754 with radiographic HKA from smartphone photos (PMC, 2025)
- RTMPose with adversarial training achieves 75% accuracy for genu valgum classification (ScienceDirect, 2024)
- SMPL-X dual-view achieves <0.8 cm error for body measurements (2024)

### 2.5 Single Image vs Multi-Image Decision

```
IF num_images == 1:
    → Use Path B (single-image metric depth)
    → Can measure: HKA (frontal), height, knee/ankle gaps
    → Cannot measure: recurvatum, 3D rotational deformities
    → Point cloud: partial (front surface only)
    → Recommendation: "Sufficient for screening, additional views recommended for full assessment"

IF num_images == 2 (front + side):
    → Use Path A (VGGT/AMB3R multi-view)
    → Can measure: ALL measurements including recurvatum
    → Point cloud: partial but includes frontal + lateral surfaces
    → Recommendation: "Good for clinical assessment"

IF num_images >= 3:
    → Use Path A with full reconstruction
    → Point cloud: near-complete body surface
    → Recommendation: "Comprehensive assessment with high-quality 3D model"
```

---

## 3. Pipeline 2: Fundal Height Estimation

### 3.1 Overview

**Goal:** From multiple images of a pregnant woman's belly, estimate the symphysis-fundal height (SFH) — the distance from the top of the pubic bone (symphysis pubis) to the top of the uterus (fundus).

**Clinical Context:**
- SFH is measured at every prenatal visit from ~20 weeks onward
- Rule of thumb: SFH in cm ≈ gestational age in weeks (±2 cm) for weeks 20-36
- SFH too large → macrosomia, polyhydramnios, multiple pregnancy
- SFH too small → intrauterine growth restriction (IUGR), oligohydramnios
- Manual measurement has inter-observer variability of 1.36-3.60 cm
- **No published work exists on estimating SFH from RGB images** — this is novel research

**Critical Challenge:** SFH measures internal anatomy (uterus position) through the abdominal wall. The external belly surface is only an indirect proxy. The symphysis pubis is not externally visible, and the fundus is detected by palpation, not by surface geometry alone.

### 3.2 Approach: Surface-to-SFH Regression

Since direct external measurement of SFH is not possible (the landmarks are internal), we propose a **learned regression approach**: reconstruct the 3D belly surface with metric accuracy, extract surface features, and train a model to predict SFH from these features using paired ground truth data.

### 3.3 Image Capture Protocol

```
CAPTURE REQUIREMENTS:
├── Patient standing or semi-reclined (standardized position)
├── Belly exposed from below breasts to upper thighs
├── Views needed (multi-view for 3D reconstruction):
│   ├── REQUIRED: Front view (centered on belly)
│   ├── REQUIRED: Left oblique (45°)
│   ├── REQUIRED: Right oblique (45°)
│   ├── RECOMMENDED: Left lateral (90°)
│   ├── RECOMMENDED: Right lateral (90°)
│   └── OPTIONAL: Slightly elevated angle view
├── Calibration:
│   ├── ArUco marker (5×5 cm) placed on patient's hip/thigh
│   ├── OR two markers at known distance apart
│   └── Patient height as secondary reference
├── Anatomical markers (sticker dots) placed by clinician:
│   ├── Symphysis pubis (top of pubic bone)
│   ├── Umbilicus (belly button - natural landmark)
│   └── Fundus (palpated and marked by clinician)
│       [Note: marking the fundus requires clinical skill]
├── Lighting: Even, diffuse (avoid harsh shadows on belly)
└── Distance: 0.5-1.5 meters from patient
```

### 3.4 Detailed Pipeline Stages

```
INPUT: 3-6 RGB images of pregnant belly + ArUco marker
       + optional anatomical sticker markers
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1: Preprocessing                             │
│  ──────────────────                                 │
│  1a. Belly region detection & segmentation          │
│      (SAM2 with belly prompt / fine-tuned detector) │
│  1b. Skin segmentation (exclude clothing)           │
│  1c. ArUco marker detection for scale               │
│  1d. Anatomical sticker detection (if present)      │
│      → Color-based detection of placed markers      │
│  1e. Image quality validation                       │
│  Output: Segmented belly images + marker positions  │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: 3D Belly Surface Reconstruction           │
│  ────────────────────────────────────               │
│  Path A — Multi-image (3-6 images, PREFERRED):      │
│    2a. VGGT forward pass on all belly images        │
│        → point maps, depth maps, camera poses       │
│    2b. AMB3R metric scale recovery                  │
│    2c. Mask point maps to belly region only         │
│    2d. Fuse into unified metric point cloud         │
│    2e. Surface reconstruction:                      │
│        - Screened Poisson reconstruction (Open3D)    │
│        - OR SuGaR (Surface-Aligned Gaussians)       │
│        → Clean, watertight belly surface mesh        │
│    2f. Extract mesh at belly region with proper      │
│        boundaries (below breasts to pubic area)     │
│                                                     │
│  Path B — Enhanced with body model:                 │
│    2a-2d. Same as Path A                            │
│    2e. Fit SMPL-X body model to full torso          │
│    2f. Use SMPL-X to establish body coordinate      │
│        system (pelvis orientation, spine axis)       │
│    2g. Difference between SMPL-X neutral torso      │
│        and actual belly surface = pregnancy          │
│        deformation map                              │
│                                                     │
│  Output: Metric 3D belly surface mesh +             │
│          body coordinate system                      │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 3: Anatomical Landmark Localization          │
│  ────────────────────────────────────               │
│  3a. If sticker markers present:                    │
│      → Detect markers in 3D (triangulate across     │
│        views or find on surface mesh)               │
│      → Symphysis pubis location (3D)                │
│      → Fundus location (3D) if marked               │
│                                                     │
│  3b. If sticker markers NOT present:                │
│      → Use SMPL-X body model landmarks:             │
│        - Pelvis joint → estimate symphysis pubis     │
│          (offset ~5-7 cm anterior and inferior       │
│          from SMPL pelvis joint)                    │
│        - Umbilicus → detect belly button in mesh     │
│          (geometric feature: local concavity)       │
│      → Fundus estimation:                           │
│        - Find highest point of belly curvature      │
│          along the midline (sagittal plane)         │
│        - This is an approximation; true fundus      │
│          requires palpation                         │
│                                                     │
│  3c. Establish measurement coordinate system:       │
│      → Midline axis (from symphysis upward along    │
│        belly surface, following curvature)          │
│      → This is the path along which SFH is measured │
│                                                     │
│  Output: 3D positions of symphysis pubis,           │
│          umbilicus, estimated fundus + midline path  │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 4: Surface Feature Extraction                │
│  ──────────────────────────────                     │
│  Extract features from the 3D belly surface:        │
│                                                     │
│  4a. Geometric measurements:                        │
│      - Belly circumference at N levels              │
│        (e.g., 64 levels like GWU Fit3D study)       │
│      - Maximum anterior protrusion (sagittal)       │
│      - Belly volume (from watertight mesh)          │
│      - Surface area of belly region                 │
│      - Curvature map (Gaussian + mean curvature)    │
│                                                     │
│  4b. Profile measurements:                          │
│      - Midline sagittal profile curve               │
│      - Cross-sections at umbilicus, 5cm above/below │
│      - Maximum width at each cross-section level    │
│                                                     │
│  4c. Relative measurements:                         │
│      - Umbilicus height relative to symphysis       │
│      - Maximum protrusion height relative to symph. │
│      - Belly surface distance: symphysis → highest  │
│        curvature point (surface geodesic distance)  │
│                                                     │
│  4d. Shape descriptors:                             │
│      - Belly shape coefficients (spherical harmonics│
│        or PCA of cross-section profiles)            │
│      - Asymmetry measures (L vs R comparison)       │
│                                                     │
│  Output: Feature vector (50-200 dimensions)         │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 5: SFH Prediction                            │
│  ────────────────────                               │
│  Two approaches (train both, evaluate which is      │
│  better):                                           │
│                                                     │
│  5a. Direct regression:                             │
│      Input: Surface feature vector + patient        │
│             metadata (height, weight, gestational    │
│             age if known, parity)                   │
│      Model: Gradient Boosting (XGBoost/LightGBM)    │
│             OR small MLP                            │
│      Output: Predicted SFH in cm                    │
│      Training data: Paired (3D belly scan, clinician│
│                     measured SFH) — MUST COLLECT     │
│                                                     │
│  5b. Surface geodesic measurement:                  │
│      If anatomical markers (symphysis + fundus)     │
│      are placed by clinician:                       │
│      → Compute geodesic distance on belly surface   │
│        from symphysis marker to fundus marker       │
│      → This is a direct digital measurement of SFH  │
│      → No ML needed for this path                   │
│      → Accuracy depends on marker placement         │
│                                                     │
│  5c. Hybrid approach (RECOMMENDED):                 │
│      - If markers present: use geodesic (5b)        │
│      - If markers absent: use regression (5a)       │
│      - Report confidence based on which path used   │
│                                                     │
│  Output: Predicted SFH (cm) + confidence interval   │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 6: Clinical Assessment                       │
│  ────────────────────────                           │
│  6a. Compare SFH to gestational age:               │
│      Expected SFH = gestational_weeks ± 2 cm        │
│      (for weeks 20-36)                              │
│                                                     │
│  6b. Flag abnormalities:                            │
│      IF SFH > gestational_weeks + 3: "Large for     │
│         dates — consider macrosomia, polyhydramnios, │
│         multiple pregnancy, wrong dates"            │
│      IF SFH < gestational_weeks - 3: "Small for     │
│         dates — consider IUGR, oligohydramnios,     │
│         wrong dates"                                │
│                                                     │
│  6c. Longitudinal tracking (if serial measurements):│
│      Plot SFH growth curve over time                │
│      Compare to standard growth curves              │
│      Flag if growth velocity abnormal               │
│                                                     │
│  6d. Additional predictions (from GWU study model): │
│      - Estimated fetal weight (from belly shape)    │
│      - Risk indicators (if trained on outcomes)     │
│                                                     │
│  Output: Clinical assessment + flags + growth chart  │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 7: Visualization & Report                    │
│  ──────────────────────────                         │
│  7a. 3D Belly Surface Viewer (Potree/Three.js):     │
│      - Colored belly surface mesh                   │
│      - Anatomical landmarks annotated               │
│      - Midline measurement path shown               │
│      - Cross-section profiles interactive           │
│      - Circumference levels visualized              │
│      - Doctor can rotate, zoom, measure             │
│                                                     │
│  7b. 2D Annotated Images:                           │
│      - Original images with landmarks overlaid      │
│      - Measurement line from symphysis to fundus    │
│                                                     │
│  7c. Clinical Report (PDF):                         │
│      - Patient info + gestational age               │
│      - SFH measurement + method used                │
│      - Comparison to expected range                 │
│      - Growth curve (if longitudinal data)          │
│      - Belly circumference profile                  │
│      - 3D viewer link                               │
│      - Clinical flags + recommendations             │
│                                                     │
│  Output: Web viewer URL + PDF report                │
└─────────────────────────────────────────────────────┘
```

### 3.5 Data Collection Strategy (Critical)

**No existing dataset exists for this task.** You must collect paired data:

```
REQUIRED DATASET:
├── Minimum viable: 100-200 pregnant women
├── Ideal: 500+ across gestational ages 20-40 weeks
├── Per patient, collect:
│   ├── 3-6 belly images (per protocol above)
│   ├── Clinician-measured SFH (tape measure, cm)
│   ├── Gestational age (from dating ultrasound)
│   ├── Patient height and pre-pregnancy weight
│   ├── Parity (number of previous pregnancies)
│   ├── BMI
│   └── Optional: ultrasound fetal biometrics
├── Longitudinal: same patients at multiple visits ideal
└── Ethics: IRB approval required for pregnant subjects
```

### 3.6 Expected Accuracy & Clinical Relevance

**Target:** ±2 cm accuracy (matching manual measurement inter-observer variability)

**Why this is achievable:**
- Manual SFH measurement has mean error of ~1.25 cm and inter-observer limits of ±3.6 cm
- 3D surface reconstruction achieves <2.5 mm accuracy for body measurements
- GWU Fit3D study achieved 72.22% accuracy for fetal weight estimation within 10% error from optical body scanning
- If we can achieve ±2 cm, we match clinical practice

**Why this is valuable even with limited accuracy:**
- Standardized measurement (eliminates inter-observer variability)
- Non-contact (more comfortable for patient)
- Generates 3D data for longitudinal tracking
- Potential for additional biomarkers from belly shape analysis
- Novel research direction with no competition

---

## 3.6.1 Noise Handling & Post-Processing (Both Pipelines)

### Why Noise Occurs

VGGT predicts point maps **independently per frame**. When merged, each frame's 3D points are slightly inconsistent with others, causing:
- Duplicate/overlapping surfaces (thickened walls, double layers)
- Misaligned edges and floating noise points
- Inconsistent depth at occlusion boundaries

The AMB3R backend (sparse voxel + Point Transformer v3) reduces ~60-70% of this noise by reasoning in 3D space. However, **human bodies are harder than rigid objects** because:
- Skin is textureless and uniform → harder for feature matching
- Patient may slightly move between captures (breathing, swaying)
- Skin can be shiny/sweaty → depth errors from specularity
- Thin structures (fingers, leg gaps) produce extra noise
- Clothing edges create discontinuities

### Post-Processing Pipeline (After AMB3R)

```
AMB3R output (metric point cloud, partially cleaned)
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: Segmentation Masking                       │
│  Remove any points outside body/belly region        │
│  using SAM2 segmentation masks projected to 3D      │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: Statistical Outlier Removal (Open3D)       │
│  For each point, compute mean distance to           │
│  k-nearest neighbors. Remove points where           │
│  distance > mean + 2*std_dev                        │
│  Typical params: k=20, std_ratio=2.0                │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: Screened Poisson Surface Reconstruction    │
│  Converts noisy point cloud → smooth watertight     │
│  mesh. This is the clean surface for measurements.  │
│  Library: Open3D (create_from_point_cloud_poisson)  │
│  Typical params: depth=9, scale=1.1                 │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: SMPL-X Fitting (Leg Pipeline)              │
│  Parametric body model acts as strong regularizer.  │
│  Fitted mesh is anatomically plausible by design,   │
│  even if raw point cloud is noisy.                  │
│  Clinical measurements come from SMPL-X, not raw    │
│  point cloud.                                       │
└─────────────────────────────────────────────────────┘
```

### What Doctors See vs What We Measure On

| Layer | Purpose | Quality |
|-------|---------|---------|
| Raw VGGT point cloud | Visual inspection of reconstruction quality | Noisy |
| AMB3R refined cloud | Better visual, shows metric scale | Partially cleaned |
| Poisson mesh | Smooth surface for visualization + circumferences | Clean |
| SMPL-X fitted mesh | Anatomical measurements (angles, lengths) | Anatomically constrained |

Doctors can toggle between all four layers in the 3D viewer to understand reconstruction confidence.

---

## 4. Shared Infrastructure

### 4.1 Technical Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FRONTEND (Web App)                      │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Image    │  │   3D Viewer  │  │  Report Viewer   │   │
│  │  Upload   │  │  (Potree /   │  │  (PDF + Charts)  │   │
│  │  Interface│  │  Three.js)   │  │                  │   │
│  └──────────┘  └──────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    BACKEND (API Server)                    │
│  ┌──────────────────────────────────────────────────┐    │
│  │              Pipeline Orchestrator                │    │
│  │  - Receives images                               │    │
│  │  - Routes to correct pipeline (leg / fundal)     │    │
│  │  - Manages async processing                      │    │
│  │  - Returns results                               │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              GPU INFERENCE SERVER(S)                       │
│  ┌───────────┐  ┌───────────┐  ┌───────────────────┐    │
│  │   VGGT /  │  │  Sapiens  │  │   TokenHMR /      │    │
│  │   AMB3R   │  │  2B Pose  │  │   SMPLer-X        │    │
│  │  (3D Rec) │  │  (2D Pose)│  │  (Body Model)     │    │
│  └───────────┘  └───────────┘  └───────────────────┘    │
│  ┌───────────┐  ┌───────────┐  ┌───────────────────┐    │
│  │ Depth Pro │  │   SAM2    │  │   ArUco Det.      │    │
│  │ (Fallback)│  │  (Segment)│  │  (Calibration)    │    │
│  └───────────┘  └───────────┘  └───────────────────┘    │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    POST-PROCESSING                        │
│  ┌───────────────┐  ┌────────────────┐  ┌────────────┐  │
│  │  Point Cloud  │  │  Measurement   │  │  Report    │  │
│  │  Processing   │  │  Extraction    │  │  Generation│  │
│  │  (Open3D)     │  │  (NumPy/SciPy) │  │  (PDF/HTML)│  │
│  └───────────────┘  └────────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1x RTX 3090 (24GB) | 1x A100 (80GB) or 2x RTX 4090 |
| RAM | 32 GB | 64 GB |
| Storage | 500 GB SSD | 1 TB NVMe SSD |
| CPU | 8-core | 16-core |

**Model VRAM requirements:**
- VGGT (10 images): ~4 GB
- VGGT (100 images): ~21 GB
- AMB3R: ~8-12 GB (estimated, on top of VGGT)
- Sapiens-2B: ~8 GB
- TokenHMR: ~4 GB
- SAM2: ~4 GB
- Total concurrent: 20-40 GB (can be serialized to fit in 24GB)

### 4.3 Key Dependencies

```
# Core reconstruction
vggt                    # Meta's VGGT (pip install from GitHub)
amb3r                   # Metric scale (pip install from GitHub)

# Pose estimation
sapiens                 # Meta Sapiens-2B (or use Hugging Face)
# OR mmpose + rtmpose   # MMPose for RTMPose (faster alternative)

# Body model
smplx                   # SMPL-X body model
tokenhmr                # TokenHMR for SMPL fitting
# OR smpler_x           # SMPLer-X for whole-body

# Depth (fallback for single-image)
depth-pro               # Apple Depth Pro
# OR moge               # Microsoft MoGe-2

# Segmentation
segment-anything-2      # SAM2 for person/belly segmentation

# Point cloud processing
open3d                  # Point cloud ops, Poisson reconstruction, ICP
trimesh                 # Mesh operations
numpy, scipy            # Numerical computation

# Calibration
opencv-python           # ArUco detection, camera calibration

# Visualization
potree                  # Web point cloud viewer (JS)
three.js                # Web 3D viewer (JS)
matplotlib              # Charts for reports
reportlab / weasyprint  # PDF generation

# Body measurements
smpl-anthropometry      # Extract measurements from SMPL
```

### 4.4 3D Viewer Feature Set for Doctors

```
INTERACTIVE 3D VIEWER FEATURES:
├── Navigation: Rotate, pan, zoom (mouse/touch)
├── Measurement tools:
│   ├── Point-to-point distance (click 2 points → mm)
│   ├── Angle measurement (click 3 points → degrees)
│   ├── Cross-section tool (define plane → see profile)
│   ├── Circumference tool (at specified height)
│   └── Free-form path measurement (geodesic distance)
├── Visualization modes:
│   ├── Point cloud (raw reconstruction)
│   ├── Surface mesh (smooth)
│   ├── Wireframe overlay
│   ├── Skeleton overlay (joint positions)
│   ├── Heatmap overlay (curvature, deviation from normal)
│   └── Measurement annotations
├── Annotations:
│   ├── Pre-computed anatomical landmarks
│   ├── Mechanical axis lines (leg pipeline)
│   ├── Midline measurement path (fundal pipeline)
│   ├── Doctor can add custom annotations
│   └── Save/load annotation sessions
├── Comparison:
│   ├── Side-by-side with previous scan (longitudinal)
│   ├── Overlay with normal template
│   └── Difference visualization (color-coded deviation)
└── Export:
    ├── PLY/OBJ/STL (3D formats)
    ├── Screenshot (PNG)
    ├── Measurements (CSV/JSON)
    └── DICOM-compatible (if needed)
```

---

## 5. Key Papers & References

### 5.1 Core 3D Reconstruction

| Paper | Venue | Relevance |
|-------|-------|-----------|
| VGGT: Visual Geometry Grounded Transformer | CVPR 2025 (Best Paper) | Primary 3D reconstruction backbone |
| AMB3R: Metric-Scale 3D Reconstruction | CVPR 2026 | Metric scale recovery for VGGT |
| Depth Anything v3 | ICLR 2026 | Strong alternative (44% better pose than VGGT) |
| DUSt3R: Geometric 3D Vision Made Easy | CVPR 2024 | Foundational pairwise reconstruction |
| MASt3R: Matching and Stereo 3D Reconstruction | ECCV 2024 | Metric pairwise reconstruction |
| Depth Pro (Apple) | 2024 | Single-image metric depth fallback |
| MoGe-2 (Microsoft) | 2025 | Single-image metric 3D point maps |
| MapAnything (Meta) | 3DV 2026 | Universal metric reconstruction framework |

### 5.2 Human Body & Pose

| Paper | Venue | Relevance |
|-------|-------|-----------|
| Sapiens: Foundation for Human Vision Models | 2024 (Meta) | 308-keypoint pose estimation |
| ViTPose++: Vision Transformer for Pose Estimation | 2023 | High-accuracy body keypoints |
| RTMPose: Real-Time Multi-Person Pose Estimation | 2023 (MMPose) | Real-time clinical deployment |
| TokenHMR: Token-based HMR | CVPR 2024 | SMPL mesh recovery from images |
| SMPLer-X: Scaling Up Expressive Body Capture | CVPR 2024 | Whole-body SMPL-X fitting |
| SMPL-Anthropometry | GitHub | Body measurements from SMPL |

### 5.3 Leg Deformity Detection

| Paper | Venue | Relevance |
|-------|-------|-----------|
| MORA Vu AI Posture Estimation | PMC 2025 | r=0.754 correlation with radiographic HKA from photos |
| OpenPose for HKA in Walking Videos | Scientific Reports 2025 | 1.58° error vs radiography |
| Adversarially Trained RTMPose for Genu Valgum | ScienceDirect 2024 | 75% classification accuracy from photos |
| Automatic Lower Limb Deformity Assessment | BMC 2025 | 0.45° angle MAE on radiographs |
| ChatGPT + Body Landmarks for Genu Valgum | ScienceDirect 2024 | LLM-based classification approach |

### 5.4 Fundal Height & Pregnancy Body Scanning

| Paper | Venue | Relevance |
|-------|-------|-----------|
| GWU Fit3D Pregnancy Body Scanning | Med & Bio Eng & Computing 2025 | 3D optical scanning of pregnant women, closest prior work |
| SFH Measurement Systematic Review (37 studies, n=33,346) | PMC 2022 | Clinical accuracy of manual SFH |
| 3D Body Shapes for Pregnant Women | MDPI Sensors 2022 | 3D body shape modeling in pregnancy |
| Handheld US vs SFH for Fetal Growth Restriction | PMC | Ultrasound 100% vs SFH 42.86% sensitivity |

### 5.5 Visualization & Measurement

| Tool/Paper | Type | Relevance |
|------------|------|-----------|
| Potree | WebGL Point Cloud Viewer | Primary web viewer for doctors |
| 3D Slicer | Desktop Medical Visualization | Detailed offline analysis |
| Open3D | Point Cloud Library | Processing, ICP, Poisson reconstruction |
| CloudCompare | Desktop Point Cloud Tool | Cloud-to-cloud comparison (longitudinal) |
| aruco-estimator | GitHub Tool | Metric scale from ArUco markers in COLMAP |

---

## Summary: Recommended Implementation Order

### Phase 1: Proof of Concept (Leg Deformity — simpler, more prior work)
1. Set up VGGT + AMB3R for multi-view reconstruction
2. Integrate Sapiens/ViTPose++ for 2D pose estimation
3. Implement metric HKA angle measurement from 2D pose (single frontal image)
4. Add ArUco-based metric calibration
5. Build basic web viewer with Potree
6. Validate against manual measurements on 20-30 subjects

### Phase 2: Enhanced Leg Pipeline
7. Add SMPL-X body model fitting (TokenHMR)
8. Implement full 3D clinical measurements (MAD, leg length, etc.)
9. Add abnormality classification
10. Build clinical report generation
11. Validate on larger clinical dataset

### Phase 3: Fundal Height Pipeline (requires data collection)
12. Design and get IRB approval for data collection protocol
13. Begin paired data collection (images + clinician SFH)
14. Adapt 3D reconstruction pipeline for belly surface
15. Implement surface feature extraction
16. Train SFH regression model
17. Validate and iterate

### Phase 4: Production
18. API server with async processing
19. Full web application with both pipelines
20. DICOM integration (if needed for hospital systems)
21. Clinical validation study
