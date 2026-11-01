# RetinaScan AI

**Explainable Deep Learning for OCT-Based Retinal Disease Classification**

An end-to-end deep learning research application that classifies Optical Coherence Tomography (OCT) retinal scans into four categories — **CNV**, **DME**, **DRUSEN**, and **NORMAL** — with real-time Grad-CAM visual explanations, feature-based severity assessment, and patient symptom questionnaire integration.

> **Disclaimer:** This system is intended for research and decision-support purposes only. It is not a substitute for professional clinical diagnosis by a qualified ophthalmologist.

---

## Key Highlights

| Metric | Value |
|---|---|
| **Architecture** | ConvNeXt-Tiny (Liu et al., 2022) |
| **Parameters** | 28.5 million |
| **Dataset** | Kermany OCT2017 (84,484 images) |
| **Test Accuracy** | **99.90%** (967/968 correct) |
| **Explainability** | Grad-CAM heatmaps (`model.features[-1]`) |
| **Input Validation** | 5-signal OOD pipeline (HSV + RGB + Laplacian + Edge + Feature-space) |
| **Interfaces** | Flask web platform + Gradio standalone UI |

---

## Project Structure

```
RetinaScan_AI/
├── server.py                              # Flask web server (main platform)
├── app.py                                 # Gradio standalone UI
├── best_convnext_tiny_oct2017.pth         # Trained model checkpoint (~334 MB, Git LFS)
├── requirements.txt                       # Python dependencies
├── final_metrics.json                     # Model evaluation metrics
├── training_history.json                  # Training epoch history
│
├── backend/
│   ├── __init__.py
│   ├── inference.py                       # Model loading, preprocessing, Grad-CAM, analysis
│   ├── ood_detector.py                    # Out-of-distribution / non-OCT image detection
│   ├── history.py                         # Analysis history (JSON-based storage)
│   ├── report_generator.py                # PDF report generation (reportlab)
│   ├── severity.py                        # Feature-based disease severity assessment
│   └── questionnaire.py                   # Patient symptom questionnaire & consistency
│
├── templates/                             # Flask HTML templates
│   ├── base.html                          # Shared layout (sidebar, navigation)
│   ├── login.html                         # Authentication page
│   ├── dashboard.html                     # Overview dashboard
│   ├── analysis.html                      # New OCT analysis (upload)
│   ├── result.html                        # Analysis result with Grad-CAM + severity + questionnaire
│   ├── history.html                       # Analysis history table
│   ├── analytics.html                     # Usage analytics and charts
│   ├── model_info.html                    # Model architecture and performance details
│   ├── settings.html                      # Application settings
│   └── profile.html                       # User profile
│
├── static/                                # CSS, JavaScript, icons
├── demo_image/                            # Sample OCT images (one per class)
│   ├── CNV-1016042-1.jpeg
│   ├── DME-119840-1.jpeg
│   ├── DRUSEN-1246453-1.jpeg
│   └── NORMAL-1073137-1.jpeg
│
├── data/
│   └── history.json                       # Runtime analysis history (gitignored)
│
├── screenshots/                           # Auto-generated report screenshots
├── gradcam_individual/                    # Pre-generated Grad-CAM samples
├── accuracy_curve.png                     # Training accuracy curve
├── loss_curve.png                         # Training loss curve
├── confusion_matrix.png                   # Test confusion matrix
├── gradcam_comparison.png                 # Grad-CAM comparison grid
└── RETINASCAN_AI_CODE_MODULES.md          # Code module reference
```

---

## Quick Start

### Prerequisites
- Python 3.10 or later
- ~1.5 GB disk space (model checkpoint + dependencies)
- Git LFS (for cloning the model checkpoint)

### 1. Clone the Repository

```bash
git clone https://github.com/murali18-a/RetinaScan_AI.git
cd RetinaScan_AI
```

If Git LFS is not installed, install it first:
```bash
git lfs install
git lfs pull
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

**Flask Web Platform** (recommended):
```bash
python server.py
```
Open **http://127.0.0.1:7860** in your browser.

Demo credentials:
- Email: `researcher@retinascan.ai`
- Password: `retinascan2024`

**Gradio Standalone UI** (alternative):
```bash
python app.py
```
The Gradio interface will auto-open at **http://127.0.0.1:7860**.

---

## Classification Pipeline

1. **Input Validation** — Five-signal out-of-distribution check rejects non-OCT images:
   - HSV saturation (OCT scans are grayscale, S < 25)
   - RGB channel similarity (R ≈ G ≈ B for grayscale)
   - Laplacian variance (rejects screenshots/documents with sharp text)
   - Edge orientation (OCT scans have horizontal retinal layers)
   - Feature-space similarity + confidence entropy analysis
2. **Preprocessing** — Convert to grayscale → resize to 224×224 → ImageNet normalization
3. **Inference** — ConvNeXt-Tiny forward pass → softmax probabilities
4. **Grad-CAM** — Hook `model.features[-1]` → weighted activation map → JET heatmap overlay
5. **Report** — Prediction with confidence, clinical context, and attention region

### Disease Classes

| Class | Full Name | Clinical Significance |
|---|---|---|
| **CNV** | Choroidal Neovascularization | Abnormal blood vessel growth; wet AMD indicator |
| **DME** | Diabetic Macular Edema | Fluid accumulation from diabetes complications |
| **DRUSEN** | Drusen Deposits | Early AMD indicator; sub-RPE deposits |
| **NORMAL** | Healthy Retina | No pathological features detected |

---

## New Features

### Feature-Based Severity Assessment

After classification, a clinician or reviewer can assess disease-specific OCT features through an interactive checklist on the result page. The system computes a prototype severity category based on the number of observed features.

**Disease-specific features:**

| Disease | Features |
|---|---|
| **CNV** | PED, intraretinal fluid, subretinal fluid, subretinal hyperreflective material, outer retinal changes |
| **DME** | Intraretinal cysts, retinal thickening, subretinal fluid, DRIL, ellipsoid zone disruption |
| **DRUSEN** | RPE elevations, drusen size, drusen distribution, drusen confluence, reticular pseudodrusen |
| **NORMAL** | Retinal layer organization, foveal contour, absence of disease features |

**Severity categories** (research prototypes, NOT clinically validated):
- **Low feature burden** — 1–2 features observed
- **Moderate feature burden** — 3 features observed
- **High feature burden** — 4–5 features observed
- **Unable to assess** — No features checked

### Patient Symptom Questionnaire

An 8-question symptom questionnaire collects patient-reported visual symptoms:
- Blurred vision, distorted/wavy vision, dark/blank spots
- Difficulty reading, reduced central vision, sudden vision changes
- Duration of symptoms, affected eyes

The system compares reported symptoms against expected symptom profiles for each disease class and provides a **consistency summary**. If symptoms and the OCT prediction appear inconsistent, a message recommends professional review. The questionnaire never overrides the model prediction.

**Urgent referral:** If sudden significant vision changes are reported, the system flags the case for immediate professional medical assessment.

### PDF Report Generation

Downloadable PDF reports include: scan metadata, input validation results, model prediction with probabilities, Grad-CAM images, severity assessment (if completed), questionnaire results (if completed), and disclaimer.

---

## API Endpoints

All API endpoints require authentication (session-based).

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Upload and analyze an OCT image |
| `POST` | `/api/analyze-example` | Analyze a built-in demo image |
| `GET` | `/api/history` | List all analysis records |
| `DELETE` | `/api/history/<scan_id>` | Delete an analysis record |
| `POST` | `/api/history/clear` | Clear all history |
| `GET` | `/api/statistics` | Get aggregate statistics |
| `GET` | `/api/report/<scan_id>` | Download PDF report |
| `GET` | `/api/severity/features/<class>` | Get feature checklist for a class |
| `POST` | `/api/severity/assess` | Compute severity from checked features |
| `GET` | `/api/questionnaire/questions` | Get symptom questionnaire |
| `POST` | `/api/questionnaire/evaluate` | Evaluate symptom consistency |

---

## Model Details

### Architecture
- **Base:** ConvNeXt-Tiny (pretrained on ImageNet-1K)
- **Head:** Linear(768, 4) replacing the original 1000-class classifier
- **Training:** Fine-tuned on Kermany OCT2017 dataset for 10 epochs
- **Best Validation Accuracy:** 97.77%
- **Test Accuracy:** 99.90% on the 968-image held-out test set

### Grad-CAM Implementation
- **Target layer:** `model.features[-1]` (last convolutional block)
- **Hook type:** `register_full_backward_hook` (PyTorch 2.x compatible)
- **Overlay:** JET colormap, 55% original / 45% heatmap blend

### Input Validation Pipeline

| Signal | Threshold | Purpose |
|---|---|---|
| HSV Saturation | Mean S < 25 | Reject color images |
| RGB Channel Diff | Δ < 20 | Reject non-grayscale |
| Laplacian Variance | Var < 3500 | Reject screenshots/text |
| Edge Orientation | H-ratio > 0.35 | Require horizontal layers |
| Feature Similarity | Cosine distance to reference | Reject out-of-domain |
| Confidence + Entropy | conf < 70% AND entropy > 75% max | Flag uncertain predictions |

---

## Limitations

- The model was trained on the Kermany OCT2017 dataset and may not generalize to OCT images from different scanners, protocols, or patient populations.
- The severity assessment module uses feature counts as a proxy and does not measure OCT biomarkers (e.g., retinal thickness). It is a research prototype.
- The symptom questionnaire provides a consistency check only and cannot independently diagnose disease.
- Grad-CAM highlights regions of model attention and should not be interpreted as anatomical annotation.
- The system does not replace clinical judgment. All results should be reviewed by a qualified ophthalmologist.

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
gradio>=6.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
flask>=3.0.0
reportlab>=4.0.0
```

---

## Team

- **Murali A** — Model training, Grad-CAM integration, severity assessment
- **Niranjan T** — Data preprocessing, evaluation
- **Praveen K** — UI development, input validation

**Guide:** Dr. C. Santhosh Kumar
**Institution:** Sona College of Technology

---

## References

1. Kermany, D. S., et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 172(5), 1122–1131.
2. Liu, Z., et al. (2022). "A ConvNet for the 2020s." *CVPR 2022*.
3. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV 2017*.

---

## License

This project is developed as an academic final-year project. All rights reserved.
