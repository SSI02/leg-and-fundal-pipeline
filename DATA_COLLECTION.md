# Data Collection Protocol: Fundal Height Estimation

## Overview

This document describes the data needed to train and validate the fundal height
estimation pipeline. **No existing dataset covers this task** — you must collect
paired (belly images, clinician-measured SFH) data.

---

## What to Collect Per Patient Visit

### Required Data

| Item | Format | Details |
|------|--------|---------|
| **Belly images** | 3-6 JPG/PNG | Multiple views of exposed belly (see Capture Protocol below) |
| **Clinician-measured SFH** | Number (cm) | Tape measure from symphysis pubis to uterine fundus |
| **Gestational age** | Number (weeks) | From dating ultrasound (most accurate) or LMP |
| **Patient height** | Number (cm) | Measured with stadiometer |
| **Pre-pregnancy weight** | Number (kg) | Self-reported or from records |
| **Current weight** | Number (kg) | Measured at visit |
| **Parity** | Number | Number of previous deliveries (affects belly shape) |

### Recommended Additional Data

| Item | Why |
|------|-----|
| **BMI** | Body habitus affects belly shape-to-SFH relationship |
| **Fetal presentation** | Cephalic/breech affects fundus position |
| **Amniotic fluid index** | From ultrasound, correlates with belly volume |
| **Estimated fetal weight** | From ultrasound biometrics (BPD, HC, AC, FL) |
| **Fundal height measured by 2 clinicians** | For inter-observer variability baseline |
| **Ultrasound-measured uterine fundus height** | If available, gold standard comparison |

---

## Image Capture Protocol

### Patient Position
- Standing upright OR semi-reclined at 30° (standardize — pick ONE and use consistently)
- Belly exposed from just below breasts to upper thighs
- Arms at sides or hands clasped behind head (consistent across patients)
- Feet shoulder-width apart

### Required Views (minimum 3)

```
        Front (0°)          Left Oblique (45°)     Right Oblique (45°)
        ┌─────────┐         ┌─────────┐            ┌─────────┐
        │         │         │    /    │            │    \    │
        │  (o o)  │         │   /     │            │     \   │
        │   \_/   │         │  /      │            │      \  │
        │  |   |  │         │ /       │            │       \ │
        │  | O |  │         │/   O    │            │    O   \│
        │  |   |  │         │   /     │            │     \   │
        │  |   |  │         │  /      │            │      \  │
        └─────────┘         └─────────┘            └─────────┘
        REQUIRED            REQUIRED               REQUIRED
```

### Recommended Additional Views (for better 3D)

| View | Angle | Purpose |
|------|-------|---------|
| Left lateral | 90° | Full side profile of belly |
| Right lateral | 90° | Symmetry check |
| Slightly elevated | 30° above horizontal | Better view of fundus area |

### Camera Settings
- **Distance**: 0.5-1.5 meters from belly
- **Lighting**: Even, diffuse (no harsh shadows on belly)
- **Background**: Plain, uncluttered
- **Resolution**: Minimum 1920x1080 (higher is better)
- **Focus**: Sharp on belly surface

### Calibration (choose one)
1. **ArUco marker** (PREFERRED): Print a 5x5cm ArUco marker, tape to patient's hip/thigh
2. **Scale bar**: Place a ruler or tape measure in frame
3. **Patient height as reference**: Less accurate but requires no props

### Optional Anatomical Markers
For Mode A (direct measurement), a clinician places small colored sticker dots:
- **Symphysis pubis**: On the skin directly above the top of the pubic bone
- **Umbilicus**: Belly button (natural landmark, no sticker needed)
- **Fundus**: Clinician palpates the top of the uterus and marks with a sticker

These markers allow direct surface-path measurement of SFH without ML.

---

## Dataset Size Requirements

| Phase | Patients | Visits per Patient | Total Samples | Purpose |
|-------|----------|-------------------|---------------|---------|
| **Pilot** | 20-30 | 1 | 20-30 | Verify pipeline works, test feature extraction |
| **Training (minimum)** | 100-150 | 1-3 | 150-300 | Train regression model |
| **Training (ideal)** | 300-500 | 2-4 | 600-2000 | Robust model with longitudinal data |
| **Validation** | 30-50 | 1 | 30-50 | Held-out test set (never used for training) |

### Gestational Age Distribution
Aim for roughly uniform coverage across weeks 20-40:
- Weeks 20-24: ~20% of samples
- Weeks 24-28: ~20%
- Weeks 28-32: ~20%
- Weeks 32-36: ~20%
- Weeks 36-40: ~20%

### Diversity
Include variation in:
- BMI (underweight, normal, overweight, obese)
- Parity (nulliparous vs multiparous)
- Fetal presentation (cephalic, breech)
- Ethnicity (belly shape varies)
- Singleton vs multiple pregnancy

---

## Data Organization

```
fundal_height_dataset/
├── metadata.csv                    # Master spreadsheet (see below)
├── patient_001/
│   ├── visit_01/
│   │   ├── images/
│   │   │   ├── front.jpg
│   │   │   ├── left_oblique.jpg
│   │   │   ├── right_oblique.jpg
│   │   │   ├── left_lateral.jpg    # optional
│   │   │   └── right_lateral.jpg   # optional
│   │   └── metadata.json
│   ├── visit_02/
│   │   └── ...
│   └── ...
├── patient_002/
│   └── ...
└── ...
```

### metadata.csv Columns

```csv
patient_id,visit_id,date,gestational_age_weeks,sfh_cm,sfh_cm_observer2,height_cm,weight_kg,pre_pregnancy_weight_kg,bmi,parity,fetal_presentation,amniotic_fluid_index,estimated_fetal_weight_g,notes
P001,V01,2026-04-15,28,27.5,28.0,165,72,58,26.4,1,cephalic,14.2,1150,
P001,V02,2026-05-13,32,31.0,31.5,165,75,58,27.5,1,cephalic,13.8,1850,
P002,V01,2026-04-16,24,23.0,,170,68,62,23.5,0,cephalic,,,first pregnancy
```

### Per-Visit metadata.json

```json
{
    "patient_id": "P001",
    "visit_id": "V01",
    "date": "2026-04-15",
    "gestational_age_weeks": 28,
    "gestational_age_days": 196,
    "sfh_cm": 27.5,
    "sfh_cm_observer2": 28.0,
    "patient_height_cm": 165,
    "patient_weight_kg": 72,
    "pre_pregnancy_weight_kg": 58,
    "bmi": 26.4,
    "parity": 1,
    "gravidity": 2,
    "fetal_presentation": "cephalic",
    "amniotic_fluid_index": 14.2,
    "estimated_fetal_weight_g": 1150,
    "patient_position": "standing",
    "aruco_marker_size_cm": 5.0,
    "anatomical_markers_placed": false,
    "camera_distance_m": 1.0,
    "notes": ""
}
```

---

## Ethics & IRB

### Required Before Data Collection
1. **IRB/Ethics approval** for imaging pregnant women
2. **Informed consent** from each patient covering:
   - Photography of exposed belly
   - Storage and use of images for research
   - De-identification of all data
3. **De-identification**: No faces in belly images; strip all EXIF metadata; use coded patient IDs

### Privacy Considerations
- Store raw images on encrypted, access-controlled storage
- Strip EXIF GPS/location data
- Ensure belly images don't include face or identifying features
- Follow local data protection regulations (HIPAA, GDPR, etc.)

---

## Quality Control Checklist

For each capture session, verify:

- [ ] All required views captured (front, left oblique, right oblique)
- [ ] Belly fully visible (sub-mammary to pubic area)
- [ ] No motion blur
- [ ] Even lighting, no harsh shadows
- [ ] ArUco marker (or reference) visible in at least one image
- [ ] SFH measured and recorded (by trained clinician)
- [ ] Gestational age confirmed (from dating ultrasound)
- [ ] Patient metadata complete
- [ ] Images de-identified (no face visible, EXIF stripped)

---

## What This Data Enables

### With 20-30 patients (pilot)
- Verify 3D reconstruction works on belly images
- Test surface feature extraction
- Establish baseline correlation between features and SFH

### With 100-150 patients (minimum training set)
- Train a gradient boosting or MLP regression model: features → SFH
- Expected accuracy: ±2-3 cm (comparable to manual measurement variability)

### With 300+ patients (robust model)
- Train a CNN or transformer that takes images directly → SFH
- Add longitudinal tracking (growth velocity)
- Predict fetal weight, GDM risk (following GWU Fit3D study approach)
- Expected accuracy: ±1.5-2 cm

### With 500+ patients (research publication)
- Validate against ultrasound measurements
- Multi-site validation
- Subgroup analysis (BMI, parity, ethnicity)
- Publishable results
