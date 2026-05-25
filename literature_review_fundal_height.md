# Comprehensive Literature Review: Fundal Height Estimation & Related Technologies

**Date:** March 2026
**Scope:** Clinical fundal height measurement, AI-based estimation, 3D surface reconstruction, clinical imaging approaches, and technical challenges

---

## Table of Contents

1. [Fundal Height Measurement: Clinical Background](#1-fundal-height-measurement-clinical-background)
2. [Automated/AI-Based Fundal Height Estimation](#2-automatedai-based-fundal-height-estimation)
3. [3D Surface Reconstruction of the Human Body/Torso](#3-3d-surface-reconstruction-of-the-human-bodytorso)
4. [Related Clinical Imaging Approaches](#4-related-clinical-imaging-approaches)
5. [Challenges Specific to Fundal Height Estimation from Images](#5-challenges-specific-to-fundal-height-estimation-from-images)
6. [Identified Research Gaps & Opportunities](#6-identified-research-gaps--opportunities)
7. [Key References & Resources](#7-key-references--resources)

---

## 1. Fundal Height Measurement: Clinical Background

### 1.1 Clinical Definition

**Symphysis-fundal height (SFH)** is the distance measured from the **symphysis pubis** (the cartilaginous joint between the pubic bones) to the **uterine fundus** (the top of the uterus). It is a core component of routine antenatal care worldwide, used as a simple, inexpensive screening tool for monitoring fetal growth.

### 1.2 Anatomical Landmarks

| Landmark | Description |
|----------|-------------|
| **Symphysis pubis** | Cartilaginous joint at the anterior junction of the two pubic bones; palpated as the bony prominence at the lower midline of the abdomen |
| **Uterine fundus** | The uppermost, dome-shaped portion of the uterus; its position rises progressively through pregnancy |

### 1.3 Measurement Technique (Standard Clinical Protocol)

1. Patient lies **supine** with bladder emptied
2. Clinician palpates the **uterine fundus** with one hand
3. Using a **flexible tape measure** (paper or plastic), the zero end is placed at the superior border of the symphysis pubis
4. The tape is extended along the **longitudinal axis** of the uterus to the fundus
5. The measurement is recorded in **centimeters**

**Technique variations documented in literature (19 distinct methods identified):**
- **Instrument:** Tape measure vs. calipers vs. finger-width vs. ultrasound
- **Superior landmark:** Top of fundus vs. fetal pole
- **Axis:** Midline vertical vs. diagonal to highest point of fundus
- **Tape contact:** Along skin surface vs. straight line between two hands
- **Blinding:** Some protocols recommend reading tape after placement (to avoid bias from knowing gestational age)

### 1.4 Normal Ranges by Gestational Age

The fundamental clinical rule of thumb: **SFH in cm approximately equals gestational age in weeks (from 20-36 weeks)**, with a tolerance of +/- 2 cm.

| Gestational Age (weeks) | Expected SFH (cm) | Normal Range (cm) |
|--------------------------|-------------------|-------------------|
| 20 | ~19-20 | 17-22 |
| 24 | ~24 | 22-26 |
| 28 | ~28 | 26-30 |
| 32 | ~32 | 30-34 |
| 36 | ~36 | 34-38 |
| 40 | ~35-37 | 33-38 |

**Growth rate patterns:**
- Weeks 20-32: ~1.0 cm/week increase
- Weeks 33-36: ~0.7 cm/week
- Weeks 37-40: ~0.3 cm/week (growth slows as fetus descends into pelvis)

### 1.5 Clinical Significance

SFH measurement is used to screen for:
- **Intrauterine Growth Restriction (IUGR):** SFH measuring small for dates (>3 cm below expected)
- **Macrosomia / Large for Gestational Age (LGA):** SFH measuring large for dates
- **Polyhydramnios / Oligohydramnios:** Abnormal amniotic fluid volumes
- **Multiple gestation:** Unexpectedly large measurements
- **Malpresentation:** Transverse lie may alter measurement

### 1.6 Accuracy and Limitations of Traditional Measurement

**Systematic review findings (PMC9409500, 37 studies, n=33,346):**

| Metric | Value |
|--------|-------|
| Simple 1cm=1week accuracy | 95% LOA: +/- 42.8 days |
| Statistical model (single SFH) | 95% prediction error: +/- 45.5 days |
| Statistical model (3 serial SFH) | 95% LOA: +/- 33 days |
| Percentage within +/- 14 days of ultrasound dating | 71% (95% CI: 66-77%) |
| Inter-rater mean differences | 0.66 cm to 2.06 cm |
| Sensitivity for preterm birth (3 serial) | 43% |
| Specificity for preterm birth (3 serial) | 96% |

**Key limitations:**
- Highly operator-dependent: inter-examiner differences of 1.36-3.60 cm (max 11.5 cm)
- Knowledge of gestational age biases the measurement (clinicians "find" what they expect)
- Mean absolute error of 1.25 cm, with 42.1% of errors exceeding 1 cm
- Not reliable in: multiple pregnancies, polyhydramnios, uterine fibroids, transverse lie, obesity
- 19 different measurement techniques create inconsistency across clinical settings

---

## 2. Automated/AI-Based Fundal Height Estimation

### 2.1 Current State of the Art

**Critical finding: There are NO published papers (as of March 2026) that directly address automated fundal height estimation from RGB images of the pregnant belly.** This represents a significant research gap.

The closest related work falls into several categories:

### 2.2 AI for Ultrasound-Based Fetal Biometrics

| Study | Year | Method | Key Finding |
|-------|------|--------|-------------|
| NEJM Evidence (Blind US sweeps) | 2022 | CNN frame classification + Bayesian aggregation | Human-level biometric measurement from untrained operator scans |
| npj Digital Medicine (20-week scans) | 2024 | Whole-examination AI | Automated biometric extraction from every frame; 3-second real-time processing |
| Horgan et al., Prenatal Diagnosis | 2023 | Scoping review | AI in obstetric US is rapidly maturing for measurement automation |
| npj Digital Medicine (Gestational age) | 2023 | ML on standard US planes | Accurate GA estimation from image analysis alone without measurement inputs |

### 2.3 Optical 3D Body Scanning for Pregnancy (The GWU Study)

**The most directly relevant work** is from George Washington University (2025):

**Paper:** "Maternal and fetal health status assessment by using machine learning on optical 3D body scans"
**Published in:** Medical & Biological Engineering & Computing, 2025
**Authors:** GWU Departments of Computer Science, Statistics, and Obstetrics/Gynecology

**Key details:**
- **Scanner:** Fit3D Proscanner (commercial optical scanner, ~$5,000; similar technology to iPhone 3D scanning)
- **Sample:** 144 pregnant women scanned at 18-24 gestational weeks
- **Data extraction:** 64 horizontal circumference levels from pubic bone to breast
- **ML architecture:** Dual-stream hybrid neural network:
  - Stream 1: Elman RNN processing sequential circumference data (supervised)
  - Stream 2: PCA extracting 3 global shape descriptors capturing 98.1% of variance (unsupervised)
  - Joint layer: Concatenated features through fully connected layers

**Clinical prediction results:**

| Prediction Target | Accuracy |
|-------------------|----------|
| Preterm labor | 89.74% |
| Gestational diabetes mellitus (GDM) | 92.68% |
| Preeclampsia | <=84.62% |
| Cesarean delivery | <=84.62% |
| Fetal weight (within 10% error) | 72.22% |

**Limitations:** Small sample size (n=144), single timepoint (18-24 weeks only), abdominal region analysis only, no longitudinal tracking.

### 2.4 3D Body Shape Simulation for Pregnancy (Z-Size Ladies)

**Paper:** "Simulation of 3D Body Shapes for Pregnant and Postpartum Women" (Sensors, 2022)

- **Data:** 98 pregnant women tracked across 587 data points (12-36 weeks)
- **Method:** Multiple linear regression + morphing technique using avatar bodies
- **Input variables:** Age, pre-pregnancy weight, height, gestational age, weight gain
- **Accuracy:** Relative errors <3% for body circumferences; weight prediction within 0.725% error
- **Application:** Web-based real-time 3D visualization (Z-Size Ladies); Thai population only
- **Note:** Used tape measurements, NOT 3D scanning (pregnant women concerned about scanner safety)

### 2.5 Related Body Measurement from Images (Non-Pregnancy)

**Comprehensive review (PMC12193998, 2025):** "Inferring Body Measurements from 2D Images"

Key benchmarks for body measurement from images:

| Condition | Height MAE | Weight MAE | Waist Circ. MAE |
|-----------|-----------|-----------|-----------------|
| Synthetic 3D renders | 0.90 cm | -- | 59 mm |
| Clinical depth images | 1.40 cm | -- | -- |
| In-the-wild RGB images | 6.20-6.94 cm | 3.20-9.80 kg | -- |
| Pediatric (ARAN dataset) | 2.54 cm | 1.51 kg | 25.3 mm |

**Conclusion:** Controlled environments yield substantially lower errors than unconstrained real-world conditions.

### 2.6 Relevant GitHub Repositories

| Repository | Description | Relevance |
|-----------|-------------|-----------|
| [farazBhatti/Human-Body-Measurements-using-Computer-Vision](https://github.com/farazBhatti/Human-Body-Measurements-using-Computer-Vision) | Single-image body measurements using OpenCV + TensorFlow | Could be adapted for belly measurement |
| [DavidBoja/SMPL-Anthropometry](https://github.com/DavidBoja/SMPL-Anthropometry) | Measure the SMPL body model | SMPL-based body measurements |
| [DavidBoja/Landmarks2Anthropometry](https://github.com/DavidBoja/Landmarks2Anthropometry) | Direct 3D body measurement from sparse landmarks (VISAPP 2024) | Landmark-based approach |
| [ankesh007/Body-Measurement-using-Computer-Vision](https://github.com/ankesh007/Body-Measurement-using-Computer-Vision) | 2D image to real-world body measurements with checkerboard calibration | Calibrated measurement pipeline |
| [maria-korosteleva/Body-Shape-Estimation](https://github.com/maria-korosteleva/Body-Shape-Estimation) | Estimate body shape under clothing from 3D scans | Under-clothing estimation |
| [AI-Machine-Vision-Lab/body-measure](https://github.com/AI-Machine-Vision-Lab/body-measure) | Smartphone-based body measurement using AR + ML | Mobile deployment |
| [meyerls/aruco-estimator](https://github.com/meyerls/aruco-estimator) | Automatic scale factor estimation for COLMAP using ArUco markers | Metric scale recovery |
| [MrNeRF/awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) | Curated 3DGS papers list | Comprehensive resource |
| [chenweikai/Body_Reconstruction_References](https://github.com/chenweikai/Body_Reconstruction_References) | Paper/dataset/code collection for human body reconstruction | Research index |
| [Anttwo/SuGaR](https://github.com/Anttwo/SuGaR) | Surface-Aligned Gaussian Splatting mesh extraction (CVPR 2024) | Mesh from 3DGS |
| [yanivw12/gs2mesh](https://github.com/yanivw12/gs2mesh) | GS2Mesh: surface reconstruction from Gaussian Splatting (ECCV 2024) | Alternative mesh extraction |

---

## 3. 3D Surface Reconstruction of the Human Body/Torso

### 3.1 Multi-View Stereo (MVS) Reconstruction

**Foundational approach for body measurement:**

**Paper:** "Anthropometric body measurements based on multi-view stereo image reconstruction" (PMC3812429)
- **Setup:** Single NIKON D600 camera + rotating disk platform; images at 20-degree intervals
- **Pipeline:** SIFT features -> Structure from Motion -> Visual Hull -> PDE-based mesh refinement
- **Measurements:** Hip, waist, neck, chest, arm circumferences
- **Accuracy:** Mannequin: 0.95-4.57% error; Human subjects: <4.56% error; waist error 1.91%

**Recent high-precision systems:**
- **Binocular stereo + deep learning:** Average reconstruction accuracy within 2.5 mm
- **Multi-view RGB + RGB-D:** Measurement errors below 0.8 cm for waist and chest circumference
- **Compact multi-view imaging (2025):** Using depth-assisted stitching with SMPL parameter optimization

### 3.2 Shape-from-Silhouette / Visual Hull

**Core principle:** For each camera view, the object silhouette defines a "visual cone" in 3D space. Intersecting cones from multiple views produces the **visual hull** -- an upper bound on the object's shape.

**Key characteristics:**
- Fast computation (suitable for real-time)
- Requires only binary silhouettes (robust to texture/lighting)
- Cannot reconstruct concavities
- Accuracy improves with more views

**Recent advance -- sSfS (Segmented Shape from Silhouette, 2022):**
- Extends voxel-based SfS with body segment-aware silhouette segmentation
- Reconstructs body parts separately, improving concave area estimation
- Significantly better human body shape results than standard visual hull

### 3.3 Neural Surface Reconstruction

#### 3.3.1 NeRF (Neural Radiance Fields)

**Core concept:** Learns a continuous volumetric scene representation (density + color) from posed images using an MLP. Renders novel views via volume rendering.

**Human body variants:**
- **H-NeRF:** Constrains NeRF with structured implicit human body model (SDF-based); reconstructs humans in motion from sparse views
- **A-NeRF:** Equips NeRF with a skeleton for articulated motion; learns from unlabeled monocular video
- **Animatable NeRF (TPAMI 2024):** Creates realistic avatars from video

**Limitation for measurement:** NeRF produces density fields, not explicit surfaces. Extracting accurate meshes for measurement requires additional processing (e.g., marching cubes on density thresholds).

#### 3.3.2 NeuS (Neural Implicit Surfaces)

**Core concept:** Uses SDF (Signed Distance Function) representation instead of density, enabling direct surface extraction. Learns via volume rendering with a logistic density distribution derived from SDF.

**Advantage:** Produces clean, watertight meshes suitable for measurement.

**NeuSG variant:** Jointly optimizes NeuS + 3D Gaussian Splatting; uses 3DGS point clouds to regulate NeuS, and NeuS normals to refine 3DGS quality.

#### 3.3.3 3D Gaussian Splatting (3DGS)

**Core concept:** Represents scenes as collections of 3D Gaussian primitives with position, covariance, opacity, and color. Renders via differentiable rasterization (not ray marching). Dramatically faster than NeRF.

**Surface reconstruction from 3DGS:**

| Method | Venue | Approach | Key Innovation |
|--------|-------|----------|----------------|
| **SuGaR** | CVPR 2024 | Regularize Gaussians to align with surfaces; Poisson reconstruction | Fast mesh extraction in minutes on single GPU |
| **GS2Mesh** | ECCV 2024 | Pre-trained stereo model as geometric prior | Accurate depth from every view for smooth meshes |
| **2DGS** | 2024 | 2D Gaussians on oriented planes (disks) | Better geometric consistency than 3DGS |
| **MILo** | SIGGRAPH Asia 2025 | Differentiable mesh extraction during training | Bidirectional consistency; fewer vertices, higher quality |

**Critical caveat for measurement:** "3D Gaussian Splatting is optimized for visualization, not measurement. For engineering work requiring precise measurements, use point clouds or survey-grade mesh outputs from photogrammetry or LiDAR." -- However, recent methods like SuGaR and MILo are closing this gap.

### 3.4 Achieving Metric Scale from Reconstruction

**The scale ambiguity problem:** Structure-from-Motion and neural reconstruction methods produce reconstructions at **unknown scale**. Metric measurements require resolving this.

**Solutions:**

| Method | Description | Accuracy |
|--------|-------------|----------|
| **ArUco markers** | Place known-size fiducial markers in scene; detect in images; compute scale factor. Tool: `aruco-estimator` for COLMAP | Sub-centimeter with good marker detection |
| **Calibrated scale bars** | Physical bars with known distance between targets placed in scene | Professional-grade accuracy |
| **Known reference object** | Any object with known dimensions in the scene (e.g., checkerboard) | Depends on measurement precision |
| **Stereo baseline** | Calibrated stereo camera with known baseline distance | Accurate at close range |
| **Depth sensor fusion** | iPhone LiDAR, Intel RealSense, Azure Kinect provide metric depth | iPhone LiDAR: height error 0.55%, waist 6.90% |
| **SMPL body model** | Fit parametric body model (known scale) to observations | Indirect; depends on fitting accuracy |

**Best practice for clinical measurement:** Use multiple scale references; place them at the same depth as the measurement target; verify accuracy against known ground truth.

---

## 4. Related Clinical Imaging Approaches

### 4.1 Ultrasound-Based Fundal Height

**Standard of care:** Ultrasound is considered more accurate than manual palpation for assessing fetal growth, though manual SFH and ultrasound SFH show equivalent predictive power for fetal age.

**Comparison: Manual SFH vs. Ultrasound Abdominal Circumference (AC):**

| Metric | Manual SFH | Handheld US (AC) |
|--------|-----------|-----------------|
| Sensitivity for FGR | 42.86% | 100% |
| Specificity for FGR | 85.24% | 92.62% |
| Birth weight prediction | Comparable | Superior for extremes |

### 4.2 RGB Camera-Based Approaches for Prenatal Monitoring

**Current state: Extremely limited published work.** The field is nascent.

- **GWU Pregnancy-3D-Scan project** (2025): Only published study using optical scanning of the maternal body surface for clinical prediction
- **No published work** on fundal height estimation from standard RGB photographs
- **Smartphone-based scanning** explored using Polycam app at GWU (feasibility demonstrated but not clinically validated)

### 4.3 Body Surface Scanning for Prenatal Assessment

| Technology | Example | Pregnancy Application | Accuracy |
|-----------|---------|----------------------|----------|
| **Fit3D optical scanner** | GWU study | Predict GDM, preterm labor, fetal weight from 64 circumferences | 72-93% for various outcomes |
| **Artec Eva** (structured light) | Dutch pregnancy statue company | 3D scanning during third trimester (~2000 scans) | Commercial quality |
| **Xbox Kinect** (structured light) | DIY belly scanning projects | 3D printing of pregnant belly | Consumer quality |
| **iPhone LiDAR** | Multiple clinical studies | Breast measurement, wound assessment, body scanning | rTEM 1.43-5.19% for linear measurements |

### 4.4 Structured Light / Depth Sensor Approaches

**Structured light** projects coded patterns onto the subject and captures the deformed pattern to compute depth. Advantages: higher measurement precision than passive stereo. Disadvantages: complex setup, cost, ambient light interference.

**Documented pregnancy applications:**
- Monitoring maternal abdomen changes during pregnancy
- Quantifying abdominal deformation for evaluating fetal kicking activity
- 3D body scans for future telehealth prenatal care applications

**iPhone LiDAR accuracy for clinical body measurement:**
- Height: 0.55% error
- Hip circumference: 3.84% error
- Waist circumference: 6.90% error
- Linear breast measurements: rTEM 1.43-5.19%
- Complex curved surfaces (nipple-to-IMF): >10% error (poor)

---

## 5. Challenges Specific to Fundal Height Estimation from Images

### 5.1 Why This Is Harder Than Regular Body Measurement

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Landmark identification** | Symphysis pubis is not visible externally; must be inferred from body surface | Requires anatomical knowledge or proxy landmarks |
| **Fundus localization** | The fundal position is detected by palpation, not visual appearance; no surface marker | Cannot be directly seen in images |
| **Soft tissue deformation** | The pregnant abdomen is highly deformable; posture, breathing, fetal position all change the surface shape | Same patient can measure differently in same session |
| **Individual variation** | Maternal BMI, body habitus, parity, fetal lie all affect the relationship between surface shape and SFH | Population-level models may not generalize |
| **Surface vs. internal anatomy** | SFH measures the uterus position through the abdominal wall; the surface curvature is only an indirect proxy | The measurement fundamentally requires knowing internal anatomy |

### 5.2 Clothing Considerations

- Pregnant women typically wear loose-fitting clothing that obscures body contours
- Tight-fitting clothing (compression garments, fitted tops) preserves shape but introduces compression artifacts
- For accurate measurement, the subject would ideally be imaged in minimal clothing or standardized thin garments
- Under-clothing body shape estimation is an active research area (see: Body-Shape-Estimation repo)

### 5.3 Body Position and Posture

- **Clinical standard:** Supine position with empty bladder
- **Image-based approach** would likely use standing position (more practical for imaging)
- Standing position changes the belly contour due to gravity (fundus tilts forward)
- Breathing phase affects measurements (expiration vs. inspiration)
- Fetal position (vertex vs. breech vs. transverse) significantly alters surface shape

### 5.4 Lighting and Image Quality

- Specular reflections on skin surface can confuse 3D reconstruction
- Even, diffuse lighting required for consistent photogrammetry
- Skin tone variations affect feature matching in multi-view stereo
- For structured light: ambient lighting must be controlled

### 5.5 Required Accuracy for Clinical Relevance

**The bar is actually not very high, given how inaccurate manual measurement already is:**

| Comparison | Tolerance |
|-----------|-----------|
| Clinical SFH tolerance | +/- 2 cm from gestational weeks |
| Inter-examiner variability (manual) | 1.36-3.60 cm (max 11.5 cm) |
| Intra-examiner variability | ~1.25 cm mean absolute error |
| Ultrasound vs manual equivalence | Comparable predictive power |
| SMART guidelines for height (clinical) | MAE below 1.2-1.4 cm |

**Implication:** An automated system achieving **+/- 1-2 cm accuracy** would be **clinically competitive** with manual measurement. Even +/- 2.5 cm accuracy with high consistency (low variance) could be valuable given the high inter-observer variability of manual methods.

### 5.6 Calibration Requirements

For any image-based system to produce clinically meaningful metric measurements:

1. **Scale calibration:** Must convert pixel/reconstruction units to centimeters
   - Recommended: ArUco markers or known-size reference object in the scene
   - Alternative: Known camera intrinsics + depth estimation
   - iPhone LiDAR: Built-in metric depth but with waist-level errors of ~6.9%

2. **Spatial calibration:** Camera pose must be known relative to the subject
   - For multi-view: Structure from Motion provides relative poses; scale bar provides metric scale
   - For single-view: Requires depth estimation or known reference

3. **Anatomical calibration:** Must map between surface geometry and clinical landmarks
   - Symphysis pubis location: Requires pose estimation or manual annotation
   - Fundus location: Could potentially be inferred from surface curvature analysis

---

## 6. Identified Research Gaps & Opportunities

### 6.1 Major Gaps

1. **No published method for estimating SFH from RGB images** of the pregnant belly surface
2. **No validated non-contact SFH measurement system** exists
3. **No public dataset** of pregnant belly images with corresponding SFH ground truth measurements
4. **No validated mapping** between belly surface 3D shape and internal SFH measurement
5. **Longitudinal tracking** of belly shape changes from images has not been studied

### 6.2 Opportunities for Novel Research

1. **3D belly reconstruction + surface analysis pipeline:**
   - Multi-view images (or video) -> 3D reconstruction (COLMAP/3DGS/NeuS) -> mesh extraction -> surface curvature analysis -> SFH estimation
   - ArUco markers for metric scale
   - Potential accuracy target: +/- 2 cm

2. **Regression from 3D belly shape to SFH:**
   - Following the GWU approach but with explicit SFH as the target variable
   - Extract circumference profiles at multiple levels
   - Train regression model (SFH = f(circumferences, demographics))

3. **Single-image belly assessment:**
   - Side-view photograph -> depth estimation -> belly protrusion analysis -> gestational age/SFH estimation
   - Lower accuracy but maximal accessibility (smartphone only)

4. **Longitudinal tracking:**
   - Serial belly scans across pregnancy
   - Track growth rate rather than absolute measurement
   - Growth velocity may be more clinically useful than single-point accuracy

5. **Transfer learning from the GWU dataset:**
   - The GWU study demonstrated that 3D body shape at 18-24 weeks predicts clinical outcomes
   - A similar approach could be extended with SFH as an explicit output

### 6.3 Proposed Technical Pipeline

```
Input: Multi-view images/video of pregnant belly + scale reference (ArUco marker)
  |
  v
Step 1: 3D Surface Reconstruction
  - Option A: COLMAP + Multi-View Stereo -> Dense point cloud -> Mesh
  - Option B: 3D Gaussian Splatting (SuGaR/MILo) -> Mesh extraction
  - Option C: NeuS -> SDF -> Mesh
  |
  v
Step 2: Metric Scale Recovery
  - Detect ArUco markers -> Compute scale factor
  - Validate with known reference measurement
  |
  v
Step 3: Anatomical Landmark Detection
  - Detect body pose (MediaPipe/OpenPose)
  - Estimate symphysis pubis location from hip keypoints
  - Detect fundus as the highest point of anterior belly curvature
  |
  v
Step 4: Surface Measurement
  - Compute geodesic distance along belly surface from pubis to fundus
  - OR compute straight-line distance (matching clinical tape measure protocol)
  - Extract belly circumference profiles at multiple heights
  |
  v
Step 5: Clinical Estimation
  - Direct measurement output (SFH in cm)
  - Regression model for gestational age
  - Classification for growth abnormalities
```

---

## 7. Key References & Resources

### 7.1 Clinical Fundal Height

- [Cleveland Clinic: Fundal Height Measurement](https://my.clevelandclinic.org/health/diagnostics/22294-fundal-height)
- [Wikipedia: Fundal Height](https://en.wikipedia.org/wiki/Fundal_height)
- [PMC9409500: SFH for Gestational Age Estimation in LMICs -- Systematic Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9409500/)
- [PMC6465049: Cochrane Review -- SFH for Detecting Abnormal Fetal Growth](https://pmc.ncbi.nlm.nih.gov/articles/PMC6465049/)
- [PMC7032650: Symphysis-Fundal Height Measurement in Pregnancy](https://pmc.ncbi.nlm.nih.gov/articles/PMC7032650/)
- [Springer: SFH to Predict Small-for-Gestational-Age](https://link.springer.com/article/10.1186/s12884-015-0461-z)
- [Mayo Clinic: Fundal Height Accuracy](https://www.mayoclinic.org/healthy-lifestyle/pregnancy-week-by-week/expert-answers/fundal-height/faq-20057962)

### 7.2 AI/ML for Pregnancy Monitoring

- [NEJM Evidence: AI Estimation of GA from Blind US Sweeps](https://evidence.nejm.org/doi/full/10.1056/EVIDoa2100058)
- [Nature npj Digital Medicine: Whole-Examination AI for Fetal Biometrics](https://www.nature.com/articles/s41746-024-01406-z)
- [Springer: Maternal/Fetal Health from Optical 3D Body Scans (GWU)](https://link.springer.com/article/10.1007/s11517-025-03473-0)
- [ArXiv: GWU Study Full Text](https://arxiv.org/html/2504.05627)
- [MDPI Sensors: 3D Body Shapes for Pregnant Women (Z-Size Ladies)](https://www.mdpi.com/1424-8220/22/5/2036)
- [Nature npj Digital Medicine: ML for Gestational Age from US](https://www.nature.com/articles/s41746-023-00774-2)
- [PMC12193998: Inferring Body Measurements from 2D Images -- Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12193998/)

### 7.3 3D Surface Reconstruction

- [PMC3812429: Anthropometric Measurements via Multi-View Stereo](https://pmc.ncbi.nlm.nih.gov/articles/PMC3812429/)
- [ScienceDirect: Accurate 3D Anthropometric Measurement Using Compact Multi-View Imaging](https://www.sciencedirect.com/science/article/abs/pii/S0263224125001368)
- [PMC8840191: sSfS -- Segmented Shape from Silhouette](https://pmc.ncbi.nlm.nih.gov/articles/PMC8840191/)
- [PyImageSearch: 3DGS vs NeRF Comparison](https://pyimagesearch.com/2024/12/09/3d-gaussian-splatting-vs-nerf-the-end-game-of-3d-reconstruction/)
- [SuGaR: Surface-Aligned Gaussian Splatting (CVPR 2024)](https://github.com/Anttwo/SuGaR)
- [MILo: Mesh-In-the-Loop Gaussian Splatting (SIGGRAPH Asia 2025)](https://github.com/Anttwo/MILo)
- [GS2Mesh: Surface Reconstruction from Gaussian Splatting (ECCV 2024)](https://github.com/yanivw12/gs2mesh)

### 7.4 Metric Scale Recovery

- [GitHub: aruco-estimator -- Scale Factor Estimation for COLMAP](https://github.com/meyerls/aruco-estimator)
- [OpenCV: ArUco Marker Detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [Agisoft: Calibrated Scale Bar Guidelines](https://www.agisoft.com/pdf/tips_and_tricks/CHI_Calibrated_Scale_Bar_Placement_and_Processing.pdf)

### 7.5 Body Measurement Tools & Datasets

- [GitHub: Human-Body-Measurements-using-Computer-Vision](https://github.com/farazBhatti/Human-Body-Measurements-using-Computer-Vision)
- [GitHub: SMPL-Anthropometry](https://github.com/DavidBoja/SMPL-Anthropometry)
- [GitHub: Landmarks2Anthropometry (VISAPP 2024)](https://github.com/DavidBoja/Landmarks2Anthropometry)
- [GitHub: Body-Shape-Estimation (under clothing)](https://github.com/maria-korosteleva/Body-Shape-Estimation)
- [GitHub: Body-Measurement-using-Computer-Vision (with calibration)](https://github.com/ankesh007/Body-Measurement-using-Computer-Vision)
- [GitHub: awesome-3dbody-papers](https://github.com/3DFaceBody/awesome-3dbody-papers)
- [SMPL Body Model](https://smpl.is.tue.mpg.de/)

### 7.6 Clinical Imaging

- [PMC4465094: Diagnostic Accuracy of Fundal Height vs Handheld US](https://pmc.ncbi.nlm.nih.gov/articles/PMC4465094/)
- [Structure.io: Healthcare 3D Scanning Platform](https://structure.io/)
- [Artec 3D: Digitizing Pregnancy with Artec Eva](https://www.artec3d.com/cases/digitizing-your-pregnancy-with-artec-eva)
- [MDPI: iPhone LiDAR for 3D Body Measurement](https://www.mdpi.com/2076-3417/15/4/2001)
- [PubMed: iPhone LiDAR Breast Scanning Accuracy](https://pubmed.ncbi.nlm.nih.gov/39749942/)

---

## Summary of Key Takeaways

1. **Fundal height measurement is simple but inaccurate:** Inter-observer variability (1.36-3.60 cm) and 95% LOA of +/- 42.8 days for gestational age estimation make it a coarse screening tool.

2. **No one has published on RGB-image-based SFH estimation:** This is a completely open research area. The closest work is the GWU 3D optical scanning study (2025), which predicts clinical outcomes from belly shape but does not directly estimate SFH.

3. **3D reconstruction is mature enough for this application:** Multi-view stereo achieves <2.5 mm reconstruction accuracy; 3DGS methods like SuGaR and MILo can extract high-quality meshes; ArUco markers solve metric scale.

4. **The accuracy bar is achievable:** Given that manual SFH has ~1.25 cm mean error and ~3 cm inter-observer variability, an automated system achieving +/- 2 cm accuracy would be clinically useful.

5. **The main technical challenges are anatomical:** Localizing the symphysis pubis and uterine fundus from surface geometry is the core unsolved problem. This may require a learned mapping between surface shape features and clinical landmarks.

6. **iPhone LiDAR is a promising deployment platform:** Built-in metric depth, adequate accuracy for body-scale measurements, and universal availability make it an attractive target for clinical deployment.

7. **A paired dataset is essential:** No public dataset exists with simultaneous belly images/scans AND clinician-measured SFH. Creating such a dataset would be the most impactful first step.
