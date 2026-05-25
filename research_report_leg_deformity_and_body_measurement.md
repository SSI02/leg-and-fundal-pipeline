# Comprehensive Research Report: Leg Deformity Detection, Pose Estimation, 3D Body Reconstruction, and Clinical Measurement from Images

**Date:** March 2026
**Scope:** State-of-the-art methods (2022-2025) across five domains

---

## TABLE OF CONTENTS

1. [Leg Deformity Detection from Images](#1-leg-deformity-detection-from-images)
2. [Human Pose Estimation SOTA](#2-human-pose-estimation-sota)
3. [3D Human Body Reconstruction for Clinical Measurement](#3-3d-human-body-reconstruction-for-clinical-measurement)
4. [Clinical Measurements from Images](#4-clinical-measurements-from-images)
5. [Key Anatomical Landmarks for Leg Deformity Assessment](#5-key-anatomical-landmarks-for-leg-deformity-assessment)
6. [Recommended Pipeline for a Leg Deformity Detection System](#6-recommended-pipeline)
7. [Key GitHub Repositories and Resources](#7-key-github-repositories-and-resources)

---

## 1. LEG DEFORMITY DETECTION FROM IMAGES

### 1.1 Overview of the Problem

Leg deformities (genu varum/bowlegs, genu valgum/knock-knees, genu recurvatum/hyperextended knee) are traditionally diagnosed using full-length standing radiographs (X-rays). However, recent research has explored non-radiographic, camera-based, and AI-driven methods for accessible screening.

### 1.2 Radiograph-Based AI Systems (Gold Standard Automation)

#### YARLA (YOLOv4 And Resnet Landmark regression Algorithm)
- **Paper:** Tack et al. (2021), Computer Methods and Programs in Biomedicine
- **Method:** YOLOv4 detects ROIs (hip, knee, ankle) in full-leg radiographs; ResNet regresses landmark coordinates per ROI
- **Accuracy:** Average landmark deviation <2.0 +/- 1.5 mm; HKA angle mismatch 0.09 +/- 0.67 degrees; weighted kappa 0.86 (almost perfect agreement)
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0169260721001553

#### HKA-Net (2024)
- **Paper:** Journal of Orthopaedic Surgery and Research, 2024
- **Method:** ResNet-50 architecture; processes lower limb radiographs to predict HKA angles without explicit landmark annotations
- **Accuracy:** Bias -0.025 degrees, SD 1.422 degrees; 126.7x faster than manual measurement (49.8 min reduced to 23.6 sec)
- **Link:** https://link.springer.com/article/10.1186/s13018-024-05265-y

#### Automated Lower Limb Deformity Assessment (2025)
- **Paper:** BMC Musculoskeletal Disorders, 2025
- **Method:** Image pyramid-based CNN with 7 Gaussian pyramid levels; detects 26 landmarks (13 per leg); iterative error feedback refinement
- **Accuracy:** Landmark error 0.79 +/- 0.57 mm; angle error 0.45 +/- 0.42 degrees; 97.4% landmarks within 3mm; 95.6% angles within 2 degrees
- **Measures:** MFMTA, MLDFA, MPTA, LDTA, JLCA
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12107965/

#### AI Software for Full-Leg Standing Radiographs (2024)
- **Paper:** Scientific Reports, 2024
- **Method:** Deep learning automated support system for full-length weight-bearing radiographs
- **Measures:** Mechanical tibiofemoral angle, mLDFA, MPTA, JLCA
- **Link:** https://www.nature.com/articles/s41598-024-57887-1

#### AI Varus Analysis (2023)
- **Paper:** Knee Surgery, Sports Traumatology, Arthroscopy, 2023
- **Method:** AI-based software for pre/post-operative analysis of varus alignment on long-leg radiographs
- **Accuracy:** No significant differences vs. manual reads; mean absolute differences <=0.5 degrees
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10719140/

### 1.3 Non-Radiographic Camera/Photograph-Based Methods

#### Adversarially Trained RTMPose for Genu Valgum Detection (2024)
- **Paper:** Computers in Biology and Medicine, Vol 183, Dec 2024
- **Method:** End-to-end GV prediction using RTMPose for body landmark extraction from photographs; adversarial training to bolster robustness against noisy landmarks
- **Dataset:** 1,519 Chinese adolescents; GV annotations from 3 medical professionals
- **Accuracy:** 75% accuracy (vs. 64.25% baseline)
- **Key Finding:** Addresses landmark inaccuracy through parallels between pose estimation biases and adversarial perturbations
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S001048252401299X

#### ChatGPT + RTMPose for Genu Valgum (2024)
- **Paper:** Biomedical Signal Processing and Control, 2024
- **Method:** Two-branch approach: (1) RTMPose identifies body landmarks, (2) ChatGPT generates supplementary semantic features from subject images
- **Accuracy:** 77.19% accuracy, 77.00% recall, AUC 83.04%
- **Innovation:** First to use LLM-generated features for orthopedic deformity detection
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S1746809424007341

#### MORA Vu - AI Posture Estimation Software (2025)
- **Paper:** PMC, 2025 (Prospective pilot study)
- **Method:** CNNs and multilayer perceptrons identify 24 anatomical reference points from photographs (iPad/smartphone at 3m distance); calculates Digital HKA (DHKA) angle
- **Accuracy:**
  - DHKA vs. radiographic HKA: r = 0.754 (p < 0.001)
  - Interrater reliability ICC: 0.90 for DHKA
  - Mean bias 0.89 degrees, limits of agreement -2.83 to 4.62 degrees
- **Key Feature:** No calibration needed; works with smartphone photos
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12155411/

#### Computer Vision App for Knee Valgus Angle (2023)
- **Paper:** Healthcare, 2023
- **Method:** Beta app using computer vision for knee valgus angle measurement in 42 elite handball athletes
- **Accuracy:**
  - Test-retest reliability ICC: 0.859-0.933 (excellent)
  - Inter-rater reliability ICC: 0.658
  - Concurrent validity vs. Kinovea: r = 0.931
  - Standard error: 1.69-3.50 degrees
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10177945/

#### OpenPose for HKA Angle in Walking Videos (2025)
- **Paper:** Scientific Reports, 2025
- **Method:** OpenPose video-based pose estimation for HKA angle in knee OA patients during dynamic walking
- **Accuracy:**
  - Test-retest reliability ICC: 1.000 (excellent)
  - Consistency with radiography ICC: 0.897 (good)
  - Fixed error: 0.131 degrees
  - Absolute error: 1.579 degrees
  - Correlation R-squared: 0.814
- **Link:** https://www.nature.com/articles/s41598-025-09627-2

### 1.4 Classification Systems for Varus/Valgus

| Condition | Measurement | Classification |
|-----------|------------|----------------|
| Normal alignment | HKA ~180 degrees (or 0 degrees deviation) | -- |
| Genu varum (bowlegs) | Intercondylar distance >= 2.5 cm | Stage I: 0-10 degrees; Stage II: 10-20 degrees; Stage III: >20 degrees |
| Genu valgum (knock-knees) | Intermalleolar distance >= 2.5 cm | Grade I: <10 degrees (passively correctable); Grade II: 10-20 degrees; Grade III: >20 degrees |
| Genu recurvatum | Hyperextension beyond 0 degrees | -- |

---

## 2. HUMAN POSE ESTIMATION SOTA

### 2.1 Model Comparison Table

| Model | Year | Architecture | COCO AP (test-dev) | Speed | Keypoints | Best Use Case |
|-------|------|-------------|-------------------|-------|-----------|---------------|
| **Sapiens-2B** | 2024 | ViT (MAE pretrained) | 61.1 AP (whole-body, Humans-5K) | Moderate | 308 (body+hands+feet+face) | Research, comprehensive analysis |
| **ViTPose++ (ViTAE-G)** | 2023 | Vision Transformer | 81.1 AP (body) | Moderate | 17 (COCO) / 133 (whole-body) | High-accuracy research |
| **RTMPose-l** | 2023 | CNN (SimCC head) | 67.0 AP (whole-body) | 130+ FPS (GPU) | 133 (whole-body) | Real-time deployment |
| **RTMPose-m** | 2023 | CNN (SimCC head) | 75.8 AP (body) | 90+ FPS (CPU), 430+ FPS (GPU) | 17 (body) | Production systems |
| **DWPose** | 2023 | Distilled RTMPose | 66.5 AP (whole-body) | Fast | 133 (whole-body) | Balanced accuracy/speed |
| **HRNet-W48** | 2019 | Multi-scale CNN | 75.5 AP (body) | Moderate | 17 (COCO) | Baseline, well-validated |
| **MediaPipe Pose** | 2020+ | BlazePose | ~75% (approx.) | Real-time (mobile) | 33 | Mobile apps, screening |
| **OpenPose** | 2018 | Bottom-up CNN | ~65 AP (body) | 15-25 FPS | 25 (body) / 135 (whole) | Clinical research (well-validated) |

### 2.2 Detailed Model Analysis

#### Sapiens (Meta, 2024) -- Current SOTA
- **Architecture:** Vision Transformer pretrained with MAE on 300M human images (1.2T tokens)
- **Scale:** 0.3B to 2B parameters
- **Keypoints:** 308 total (body, hands, feet, surface, face with 243 facial keypoints)
- **Performance:** Sapiens-2B outperforms DWPose-L by +7.1 AP on Humans-5K; outperforms ViTPose+-H by +7.9 AP
- **Tasks:** Pose, depth, normal estimation, segmentation (28 body part classes)
- **Link:** https://arxiv.org/abs/2408.12569

#### ViTPose / ViTPose++ (2022/2023)
- **Architecture:** Plain ViT backbone with simple decoder
- **Performance:** 80.9 AP on COCO test-dev (ViTPose-G, 1B params); 77.8% whole-body AP on COCO-WholeBody
- **Scalability:** 20M to 1B parameters
- **Publication:** NeurIPS 2022 (ViTPose), TPAMI 2023 (ViTPose++)
- **Repo:** https://github.com/ViTAE-Transformer/ViTPose

#### RTMPose (2023)
- **Architecture:** CSPNeXt backbone + SimCC head
- **Performance:** 75.8% AP (body), 67.0% AP (whole-body)
- **Speed:** 90+ FPS on CPU, 430+ FPS on GPU
- **Key Strength:** Best real-time performance; robust detection
- **Publication:** MMPose/open-mmlab
- **Repo:** https://github.com/open-mmlab/mmpose

#### DWPose (2023)
- **Method:** Two-stage pose distillation of RTMPose
- **Performance:** Boosts RTMPose-l whole-body AP from 64.8% to 66.5%, surpassing RTMPose-x teacher (65.3%)
- **Best For:** Efficient whole-body estimation

#### HRNet (2019)
- **Architecture:** Multi-resolution parallel branches with repeated information exchange
- **Performance:** 75.5 AP (W48); widely validated in clinical studies
- **Strength:** Robust, well-established baseline; extensive clinical literature

#### MediaPipe Pose (Google)
- **Keypoints:** 33 landmarks (17 COCO-equivalent + additional face, hands, feet points)
- **Lower Body:** Hips, knees, ankles, heels, foot index (toes)
- **Clinical Accuracy:**
  - Lower limb correlation with Vicon: 0.80 +/- 0.1 (Pearson's r)
  - Upper limb: 0.91 +/- 0.08
  - Legs accuracy in ergonomic assessment: 85.53% (controlled), 71.25% (real-world)
- **Strength:** Real-time on mobile devices; no GPU needed; extensive API
- **Limitation:** Lower accuracy for occluded joints; limited validation for clinical deformity screening

#### OpenPose (CMU)
- **Keypoints:** 25 body + 21 per hand + 70 face = 135+ total
- **Clinical Validation:**
  - HKA angle: ICC 0.897 vs. radiography; fixed error 0.131 degrees
  - Sagittal plane: Hip MAE 4.0 degrees, Knee MAE 5.6 degrees, Ankle MAE 7.4 degrees
  - Temporal gait parameters: MAE 0.02 s; step lengths: MAE 0.049 m
- **Strength:** Most clinically validated model; extensive gait analysis literature
- **Limitation:** Slower; aging architecture

### 2.3 Best for Medical/Clinical Use

**Recommendation by Priority:**

1. **For highest accuracy (non-real-time):** Sapiens-2B or ViTPose++ -- best keypoint localization
2. **For clinical validation evidence:** OpenPose -- most published clinical studies, validated for HKA angle measurement
3. **For real-time clinical screening:** RTMPose-l or MediaPipe Pose -- fast, deployable on mobile
4. **For balanced approach:** RTMPose with clinical fine-tuning (as demonstrated in the genu valgum detection papers)

**Critical Note:** Clinical accuracy of joint angle measurements across all models typically ranges from 1.5-5 degrees MAE compared to gold-standard motion capture. For HKA angles specifically, OpenPose achieves ~1.5 degrees absolute error compared to radiography.

---

## 3. 3D HUMAN BODY RECONSTRUCTION FOR CLINICAL MEASUREMENT

### 3.1 SMPL / SMPL-X Body Models

#### SMPL (Skinned Multi-Person Linear Model)
- **Parameters:** 72 pose params (24 joints x 3 rotations) + 10 shape params (PCA betas)
- **Vertices:** 6,890
- **Output:** Full 3D mesh with pose-dependent deformations
- **Key Property:** Shape parameters encode body proportions (height, weight, limb lengths)

#### SMPL-X (Expressive)
- **Extension:** Adds hands (30 joints) and face (3 jaw joints + expression params)
- **Vertices:** 10,475
- **Parameters:** 55 body joints + hand + face expression codes

### 3.2 Human Mesh Recovery Methods

#### HMR2.0 / 4DHumans (2023)
- **Architecture:** End-to-end transformer; predicts SMPL parameters from single image
- **Performance:** State-of-the-art on 3DPW; mAP 22.3 on AVA (14% better than second best)
- **Limitation:** Original training on 17 sparse 2D joints limits body shape accuracy
- **Repo:** https://github.com/shubham-goel/4D-Humans

#### CameraHMR (2024)
- **Advance:** Upgrades HMR2.0 with estimated camera parameters
- **Improvement:** More accurate pseudo-ground-truth through perspective-aware training
- **Training:** Uses DenseKP detector (138 dense surface keypoints from BEDLAM) for better body shapes

#### TokenHMR (CVPR 2024)
- **Innovation:** Reformulates body regression as token prediction; Threshold-Adaptive Loss Scaling (TALS)
- **Result:** Improved 3D accuracy on EMDB and 3DPW over state-of-the-art
- **Key Benefit:** Allows training on in-the-wild data while maintaining 3D accuracy
- **Repo:** https://github.com/saidwivedi/TokenHMR

#### SMPLer-X (2024)
- **Architecture:** ViT-Huge backbone; trained on 4.5M instances from diverse sources
- **Capability:** Whole-body (body + hands + face) mesh recovery using SMPL-X
- **Output:** Full expressive body mesh with shape parameters
- **Strength:** State-of-the-art for whole-body mesh; strong transferability to unseen environments

#### WHAM (CVPR 2024) -- Video-Based
- **Method:** World-grounded human motion recovery from video
- **Innovation:** Projects camera to global coordinates; uses foot-ground contact probability
- **Performance:** Outperforms all existing 3D human motion recovery methods on 3DPW, RICH, EMDB
- **Metrics:** MPJPE, PA-MPJPE, PVE, Acceleration error
- **Best For:** Video-based temporal reconstruction with metric scale
- **Link:** https://wham.is.tue.mpg.de/

#### TRAM (ECCV 2024) -- Video-Based
- **Method:** Two-stage: (1) Robustified SLAM for camera motion, (2) VIMO video transformer for kinematic body motion
- **Performance:** Reduces global motion errors by 60% from prior work
- **Innovation:** Uses scene background to derive motion scale
- **Repo:** https://github.com/yufu-wang/tram

### 3.3 Implicit Function Methods

#### PIFu / PIFuHD (Meta)
- **Method:** Pixel-aligned implicit function for 3D human digitization from single image
- **Architecture:** Multi-level (coarse + fine); marching cubes for mesh extraction
- **Strength:** High-fidelity geometry (cloth, hair, fine details)
- **Limitation:** Not SMPL-based; harder to extract standardized measurements; struggles with challenging poses in-the-wild
- **Repo:** https://github.com/facebookresearch/pifuhd

#### Recent Advances
- Self-supervised depth-guided PIFu (2024): IoU 89.03% (20% above PIFuHD on synthetic data)
- ECON, IntegratedPIFu: improved topology handling

### 3.4 Body Measurement Extraction Tools

#### SMPL-Anthropometry
- **Repo:** https://github.com/DavidBoja/SMPL-Anthropometry
- **Measurements (16 standard):**
  - **Lengths:** Height, arm length, inside leg height, shoulder breadth, shoulder-to-crotch
  - **Circumferences:** Head, neck, chest, waist, hip, wrist, bicep, forearm, thigh, calf, ankle
- **Method:** Lengths = distance between landmarks; Circumferences = plane cuts through mesh
- **Extensible:** Custom measurements can be defined

#### Pose-Independent Anthropometry (ECCV 2024 workshop)
- **Repo:** https://github.com/DavidBoja/pose-independent-anthropometry
- **Innovation:** Extracts anthropometric measurements from posed SMPL meshes or 3D scans

#### Focused Human Body Model (CVPR 2025)
- **Paper:** Chen et al., CVPR 2025
- **Method:** Bypass Network (CNN + ResNet) augments frozen SMPLer-X backbone; dynamic loss recalibration
- **Accuracy:** Average MAE 3.32 cm for chest (vs. 5.1 cm for YouFit commercial software)
- **Repo:** https://github.com/Eddie-cc/Focused-human-body-measurement

#### SMPL-X Dual-View Clinical Study
- **Result:** Average measurement error below 0.8 cm for waist and chest circumference (120 individuals)
- **Significance:** Confirms robustness for clinical applications

#### AnthroNet
- **Method:** High-resolution model conditioned on anthropometric measurements
- **Capability:** Bi-directional conversion between measurements and SMPL/SMPL-X parameters
- **Link:** https://unity-technologies.github.io/AnthroNet/

### 3.5 Accuracy Summary for Body Measurements from Images

| Method | Measurement Type | Accuracy (MAE) |
|--------|-----------------|----------------|
| SMPL-X dual-view | Waist/chest circumference | <0.8 cm |
| Focused Body Model (CVPR 2025) | Chest circumference | 3.32 cm |
| General SMPL anthropometry | Various measurements | 2.5-16.0 mm depending on measurement |
| Commercial software (YouFit) | Chest circumference | 5.1 cm |

---

## 4. CLINICAL MEASUREMENTS FROM IMAGES

### 4.1 Mechanical Axis Deviation (MAD)

**Definition:** The perpendicular distance from the knee joint center to the mechanical axis line (line from femoral head center to ankle center).

**Normal Value:** 4 +/- 2 mm medial to knee center

**Measurement Methods:**
1. **Gold Standard:** Full-length standing anteroposterior radiograph
2. **AI from radiographs:** YARLA, HKA-Net achieve <1 degree error
3. **From photographs:** MORA Vu (r = 0.754 correlation with radiographic HKA); OpenPose (1.58 degrees absolute error)

**How to Calculate from Pose Estimation:**
1. Detect hip joint center (femoral head proxy)
2. Detect knee joint center (tibial spine proxy)
3. Detect ankle joint center (tibial plafond/malleolar midpoint proxy)
4. Draw line from hip center to ankle center (mechanical axis)
5. Measure perpendicular distance from knee center to this line
6. Positive = medial deviation (varus); Negative = lateral deviation (valgus)

### 4.2 Tibiofemoral Angle (TFA)

**Types:**
- **Mechanical TFA (mTFA):** Angle between femoral mechanical axis (hip-to-knee) and tibial mechanical axis (knee-to-ankle). Normal = ~0 degrees / 180 degrees
- **Anatomical TFA (aTFA):** Angle between femoral and tibial anatomical (shaft) axes. Normal = 173-175 degrees (5-7 degrees valgus)

**From Pose Estimation:**
1. Compute vector from hip to knee (femoral mechanical axis)
2. Compute vector from knee to ankle (tibial mechanical axis)
3. Angle between these vectors = mechanical TFA
4. Deviation from 180 degrees indicates varus (angle > 180) or valgus (angle < 180)

**Achieved Accuracy:**
- AI on radiographs: 0.19-0.45 degrees MAE (BMC 2025 study)
- OpenPose on video: 0.131 degrees fixed error, 1.579 degrees absolute error vs. radiography
- MORA Vu on photos: r = 0.754 correlation, bias 0.89 degrees

### 4.3 Intercondylar Distance (ICD) / Intermalleolar Distance (IMD)

**Definitions:**
- **ICD:** Distance between medial surfaces of the two femoral condyles (knees touching). Measured with knees together and ankles apart. ICD >= 2.5 cm suggests genu varum.
- **IMD:** Distance between medial malleoli (ankles touching). Measured with ankles together and knees apart. IMD >= 2.5 cm suggests genu valgum.

**Clinical Significance (2024):** Each 1 cm reduction in intermalleolar distance results in 0.39 degrees deviation in knee joint line obliquity and 0.35 degrees deviation in ankle joint line obliquity.

**Measurement from Images:**
1. **YOLO + Edge Detection:** One approach uses YOLO for detection and Holistically-Nested Edge Detection for measuring knee/ankle gaps
2. **Pose Estimation + Calibration:** Detect bilateral knee/ankle keypoints; convert pixel distance to real distance using a reference object or known body dimension
3. **Stereo Camera:** Direct metric measurement possible with calibrated stereo setup

**Challenge:** Converting pixel distance to real-world distance requires either:
- A reference object of known size in the image
- Camera calibration (focal length, distance to subject)
- A known body dimension (e.g., head size, shoulder width) for scale estimation
- Metric depth estimation (Depth Pro, Metric3D)

### 4.4 Person Height Estimation from Images

#### Methods:

**1. Reference Object Method**
- Place known-size object in scene
- Compute pixel-to-cm ratio
- Measure person height in pixels and convert
- Accuracy: +/- 1-3 cm with proper setup

**2. Camera Calibration Method**
- Fix camera position, know distance to subject
- Convert pixel distances using known camera intrinsics
- Vanishing point/line methods can calibrate without explicit parameters

**3. Body Proportion Method**
- Use anthropometric ratios (e.g., head length is ~1/7.5 of height)
- No calibration needed but less precise
- Uses ratios between body segments

**4. Stereo Vision / Depth-Based**
- Deep learning stereo: MAE below 1.0 cm (best reported)
- Depth cameras (RealSense, Kinect): robust metric measurement
- Depth Pro (Apple, 2024): Zero-shot metric depth from single image in 0.3s; no camera intrinsics needed

**5. Monocular Metric Depth Estimation (2024-2025 SOTA)**
- **Depth Pro (Apple):** Metric scale without camera metadata; 2.25MP depth map in 0.3s
- **Metric3D v2:** Versatile geometric foundation model; directly enables size measurement from single image
- **UniDepth (CVPR 2024):** Universal metric depth; camera-agnostic
- **Depth Anything V2 (NeurIPS 2024):** Foundation model; supports 4K metric depth

**6. SMPL-Based Height**
- Fit SMPL model to person; extract height from mesh vertices
- SMPL-Anthropometry tool provides height measurement
- Accuracy depends on mesh fitting quality

### 4.5 Accuracy Summary for Clinical Measurements

| Measurement | Method | Accuracy |
|-------------|--------|----------|
| HKA angle | AI on radiographs | 0.09 +/- 0.67 degrees (YARLA) |
| HKA angle | OpenPose on video | 1.579 degrees absolute error |
| HKA angle | MORA Vu on photos | r=0.754, bias 0.89 degrees |
| Lower limb angles | CNN on X-ray (2025) | 0.45 +/- 0.42 degrees MAE |
| Landmark detection | CNN on X-ray (2025) | 0.79 +/- 0.57 mm MAE |
| Hip joint angle | Pose estimation (gait) | 2.1-4.0 degrees MAE |
| Knee joint angle | Pose estimation (gait) | 2.1-5.6 degrees MAE |
| Ankle joint angle | Pose estimation (gait) | 2.3-7.4 degrees MAE |
| Genu valgum detection | RTMPose + adversarial | 75% accuracy |
| Genu valgum detection | RTMPose + ChatGPT | 77.19% accuracy |
| Height estimation | Stereo vision + DL | <1.0 cm MAE |
| Body circumferences | SMPL-X dual-view | <0.8 cm MAE |

**Key Limitation:** Clinical vs. radiological measurement difference can be up to 10 degrees in 95% of cases, making any clinical (non-X-ray) measurement at best an estimate with a wide margin of error. Camera-based methods narrow this gap significantly with AI.

---

## 5. KEY ANATOMICAL LANDMARKS FOR LEG DEFORMITY ASSESSMENT

### 5.1 Primary Landmarks (6 Essential for HKA)

| Landmark | Location | Detection Method |
|----------|----------|-----------------|
| **Femoral head center** (bilateral) | Center of femoral head circle | Pose estimation: hip keypoint; Radiograph: circle fitting |
| **Tibial spine center** (bilateral) | Center of tibial eminence at knee | Pose estimation: knee keypoint; Radiograph: midpoint of tibial spines |
| **Ankle center** (bilateral) | Center of tibial plafond / mid-malleolar point | Pose estimation: ankle keypoint; Radiograph: midpoint of malleoli |

### 5.2 Extended Landmarks (for Complete Angular Assessment)

| Landmark | Used For |
|----------|----------|
| **Medial/lateral femoral condyles** | Joint line orientation, JLCA |
| **Medial/lateral tibial plateaus** | MPTA calculation |
| **Greater trochanter** | Anatomical axis reference |
| **Femoral shaft points** (10cm proximal to knee) | Anatomical femoral axis |
| **Tibial shaft points** (10cm distal to knee) | Anatomical tibial axis |
| **Medial/lateral malleoli** | Ankle width, IMD measurement |
| **Heel** | Foot alignment, recurvatum assessment |
| **First metatarsal head** | Foot progression angle |

### 5.3 Landmarks from Pose Estimation Models

#### MediaPipe Pose (33 landmarks) -- Lower Body Relevant:
- Left/Right hip (landmarks 23, 24)
- Left/Right knee (landmarks 25, 26)
- Left/Right ankle (landmarks 27, 28)
- Left/Right heel (landmarks 29, 30)
- Left/Right foot index (landmarks 31, 32)

#### COCO Format (17 keypoints) -- Lower Body:
- Left/Right hip (keypoints 11, 12)
- Left/Right knee (keypoints 13, 14)
- Left/Right ankle (keypoints 15, 16)

#### Sapiens (308 keypoints):
- Comprehensive coverage including body, hands, feet surface points, face
- Most detailed anatomical coverage of any current model

### 5.4 Angles Computed from Landmarks

| Angle | Landmarks Used | Normal Range | Clinical Significance |
|-------|---------------|--------------|----------------------|
| **HKA (hip-knee-ankle)** | Hip center, knee center, ankle center | ~180 degrees (0 degrees deviation) | Overall alignment; primary screening metric |
| **mTFA (mechanical tibiofemoral)** | Same as HKA | ~180 degrees | Same as HKA; standard clinical measure |
| **aTFA (anatomical tibiofemoral)** | Femoral shaft axis, tibial shaft axis | 173-175 degrees | Traditional alignment measure |
| **mLDFA (mech. lateral distal femoral)** | Femoral mech. axis, condylar tangent | 87 +/- 3 degrees | Femoral contribution to deformity |
| **MPTA (medial proximal tibial)** | Tibial mech. axis, tibial plateau tangent | 87 +/- 3 degrees | Tibial contribution to deformity |
| **LDTA (lateral distal tibial)** | Tibial mech. axis, distal tibial tangent | 86-92 degrees | Ankle-level deformity |
| **JLCA (joint line convergence)** | Femoral condyle line, tibial plateau line | 0-1 degrees medial convergence | Intra-articular deformity |
| **Q angle** | ASIS, patella center, tibial tuberosity | 12-18 degrees | Patellar tracking, valgus tendency |

### 5.5 How to Detect Landmarks

**From Standard Photographs/Video:**
1. Use pose estimation model (MediaPipe, RTMPose, ViTPose) to detect hip, knee, ankle keypoints
2. These approximate the femoral head center, tibial spine center, and ankle center
3. Compute angles directly from 2D keypoint coordinates

**From Radiographs (X-rays):**
1. Use YOLO/object detector to locate hip, knee, ankle ROIs
2. Use ResNet/CNN to regress precise landmark coordinates within each ROI
3. Iterative refinement with error feedback (image pyramid approach)

**From 3D Body Reconstruction:**
1. Fit SMPL/SMPL-X model to image
2. Extract 3D joint positions from fitted mesh
3. Compute angles in 3D space (more accurate than 2D)

---

## 6. RECOMMENDED PIPELINE FOR A LEG DEFORMITY DETECTION SYSTEM

### Option A: Photograph-Based Screening (No X-ray)

```
Input: Front-view standing photograph
  |
  v
[1. Person Detection] -- YOLO/RTDETR
  |
  v
[2. Pose Estimation] -- RTMPose-l (real-time) or ViTPose++ (accuracy)
  |  Output: 2D keypoints (hips, knees, ankles)
  v
[3. Optional: 3D Mesh Recovery] -- HMR2.0 / TokenHMR / WHAM (video)
  |  Output: SMPL params, 3D joint positions
  v
[4. Scale Estimation] -- Reference object / Depth Pro / known height
  |  Output: pixel-to-metric conversion
  v
[5. Angle Computation]
  |  - HKA angle (hip-knee-ankle)
  |  - Intercondylar / intermalleolar distance
  |  - Knee flexion/extension angle
  v
[6. Deformity Classification]
  |  - Varus: HKA > 180 degrees (or medial MAD)
  |  - Valgus: HKA < 180 degrees (or lateral MAD)
  |  - Recurvatum: knee hyperextension (sagittal view)
  v
[7. Report Generation]
   - Measured angles, distances, classification
   - Confidence intervals based on method accuracy
```

### Option B: Video-Based Gait Analysis

```
Input: Walking video (frontal/sagittal)
  |
  v
[1. Multi-frame Pose Estimation] -- RTMPose / OpenPose
  |
  v
[2. Temporal Smoothing & Tracking]
  |
  v
[3. Optional: WHAM / TRAM] -- World-grounded 3D mesh
  |
  v
[4. Dynamic HKA Angle] -- Per-frame + statistics
  |
  v
[5. Gait Parameters] -- Step length, cadence, joint angles
  |
  v
[6. Deformity + Gait Assessment]
```

### Option C: Comprehensive Body Measurement

```
Input: Multiple views (front + side)
  |
  v
[1. SMPLer-X / HMR2.0] -- 3D mesh reconstruction
  |
  v
[2. SMPL-Anthropometry] -- Extract measurements
  |  - Height, limb lengths, joint angles
  |  - Circumferences (thigh, calf)
  v
[3. Metric Depth Estimation] -- Depth Pro / Metric3D
  |  - Absolute scale recovery
  v
[4. Combined Clinical Report]
   - Body measurements
   - Alignment angles
   - Deformity classification
```

---

## 7. KEY GITHUB REPOSITORIES AND RESOURCES

### Pose Estimation
| Repository | Description | Link |
|-----------|-------------|------|
| MMPose (RTMPose, etc.) | Comprehensive pose estimation toolbox | https://github.com/open-mmlab/mmpose |
| ViTPose | Vision Transformer pose estimation | https://github.com/ViTAE-Transformer/ViTPose |
| MediaPipe | Google's on-device ML solutions | https://github.com/google-ai-edge/mediapipe |
| Sapiens | Meta's human vision foundation models | https://github.com/facebookresearch/sapiens |
| OpenPose | CMU's multi-person pose estimation | https://github.com/CMU-Perceptual-Computing-Lab/openpose |

### 3D Body Reconstruction
| Repository | Description | Link |
|-----------|-------------|------|
| 4D-Humans (HMR2.0) | Transformer-based human mesh recovery | https://github.com/shubham-goel/4D-Humans |
| TokenHMR | CVPR 2024 tokenized pose for HMR | https://github.com/saidwivedi/TokenHMR |
| WHAM | World-grounded humans with accurate 3D motion | https://github.com/yohanshin/WHAM |
| TRAM | Global trajectory and motion from video | https://github.com/yufu-wang/tram |
| PIFuHD | High-resolution 3D human digitization | https://github.com/facebookresearch/pifuhd |

### Body Measurement
| Repository | Description | Link |
|-----------|-------------|------|
| SMPL-Anthropometry | Measure SMPL body model | https://github.com/DavidBoja/SMPL-Anthropometry |
| Pose-independent-anthropometry | ECCV 2024 workshop | https://github.com/DavidBoja/pose-independent-anthropometry |
| Focused-human-body-measurement | CVPR 2025 | https://github.com/Eddie-cc/Focused-human-body-measurement |
| Human-Body-Measurements-CV | Single image body measurements | https://github.com/farazBhatti/Human-Body-Measurements-using-Computer-Vision |

### Depth Estimation
| Repository | Description | Link |
|-----------|-------------|------|
| Depth Pro (Apple) | Metric depth in <1 second | https://github.com/apple/ml-depth-pro |
| Metric3D | Zero-shot metric depth | https://github.com/YvanYin/Metric3D |
| UniDepth | Universal metric depth estimation | https://github.com/lpiccinelli-eth/UniDepth |
| Depth Anything V2 | Foundation model for depth | https://github.com/DepthAnything/Depth-Anything-V2 |

### Clinical / Knee-Specific
| Resource | Description | Link |
|----------|-------------|------|
| OpenPose Gait Analysis | 2D gait analysis with OpenPose | https://github.com/batking24/OpenPose-for-2D-Gait-Analysis |

---

## SOURCES

### Leg Deformity Detection
- [AI-based Varus Leg Alignment Analysis (PMC, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10719140/)
- [Fully Automated Knee Alignment from Long-Leg Radiographs (PubMed, 2023)](https://pubmed.ncbi.nlm.nih.gov/38091069/)
- [Automated Knee Joint Alignment Analysis (Scientific Reports, 2024)](https://www.nature.com/articles/s41598-024-57887-1)
- [Deep Learning for Full-Leg Standing Radiographs (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11606017/)
- [Adversarially Trained RTMPose for Genu Valgum (ScienceDirect, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S001048252401299X)
- [ChatGPT + Body Landmarks for Genu Valgum (ScienceDirect, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S1746809424007341)
- [MORA Vu AI Posture Estimation Validation (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12155411/)
- [OpenPose for HKA Angle in Walking Videos (Scientific Reports, 2025)](https://www.nature.com/articles/s41598-025-09627-2)
- [Computer Vision App for Knee Valgus Angle (PMC, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10177945/)
- [Automatic Lower Limb Deformity Assessment (BMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12107965/)
- [HKA-Net for Knee OA Assessment (Springer, 2024)](https://link.springer.com/article/10.1186/s13018-024-05265-y)
- [YARLA for Knee Alignment (ScienceDirect, 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0169260721001553)

### Pose Estimation
- [Sapiens: Foundation for Human Vision Models (arXiv, 2024)](https://arxiv.org/abs/2408.12569)
- [ViTPose++ (arXiv/TPAMI, 2023)](https://arxiv.org/abs/2212.04246)
- [RTMPose (Semantic Scholar, 2023)](https://www.semanticscholar.org/paper/RTMPose:-Real-Time-Multi-Person-Pose-Estimation-on-Jiang-Lu/7fc39b00981e017864ed01f9d5fdc27a1553e11a)
- [Comprehensive Analysis of Pose Estimation Models (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11566680/)
- [Clinical Gait Analysis Using Pose Estimation (medRxiv, 2023)](https://www.medrxiv.org/content/10.1101/2023.01.26.23285007v1.full)
- [MediaPipe Pose Documentation (GitHub)](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md)
- [MMPose Repository (GitHub)](https://github.com/open-mmlab/mmpose)

### 3D Reconstruction and Body Measurement
- [4D-Humans/HMR2.0 (GitHub)](https://github.com/shubham-goel/4D-Humans)
- [TokenHMR CVPR 2024 (GitHub)](https://github.com/saidwivedi/TokenHMR)
- [WHAM CVPR 2024](https://wham.is.tue.mpg.de/)
- [TRAM ECCV 2024 (GitHub)](https://github.com/yufu-wang/tram)
- [SMPL-Anthropometry (GitHub)](https://github.com/DavidBoja/SMPL-Anthropometry)
- [Focused Human Body Model CVPR 2025 (GitHub)](https://github.com/Eddie-cc/Focused-human-body-measurement)
- [PIFuHD (GitHub)](https://github.com/facebookresearch/pifuhd)
- [Leveraging Anthropometric Measurements for HMR (arXiv, 2024)](https://arxiv.org/html/2409.17671v1)
- [Reconstructing Humans with Biomechanically Accurate Skeleton (arXiv, 2025)](https://arxiv.org/html/2503.21751v1)

### Clinical Measurements and Depth
- [Radiological Assessment of Lower Limb Alignment (PMC, 2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8246117/)
- [Depth Pro (Apple, 2024)](https://github.com/apple/ml-depth-pro)
- [Metric3D (GitHub)](https://github.com/YvanYin/Metric3D)
- [UniDepth CVPR 2024 (GitHub)](https://github.com/lpiccinelli-eth/UniDepth)
- [AI-Based Gait Analysis Validity (PMC, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10747245/)
- [Human Height Estimation with AI (ScienceDirect, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0263224124010182)
- [Deep Learning for Lower Limb 3D Landmarks (Scientific Reports, 2024)](https://www.nature.com/articles/s41598-024-84387-z)
