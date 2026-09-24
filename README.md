# 🔬 RetinaScan AI

**OCT Retinal Disease Classifier with Explainable AI (Grad-CAM)**

An end-to-end deep learning prototype that classifies Optical Coherence Tomography (OCT) retinal scans into four categories — **CNV**, **DME**, **DRUSEN**, and **NORMAL** — with real-time Grad-CAM visual explanations. Built as an academic mini-project for faculty review.

---

## 🎯 Key Highlights

| Metric | Value |
|---|---|
| **Architecture** | ConvNeXt-Tiny (Liu et al., 2022) |
| **Parameters** | 28.5 million |
| **Dataset** | Kermany OCT2017 (84,484 images) |
| **Test Accuracy** | **99.90%** (967/968 correct) |
| **Explainability** | Grad-CAM heatmaps (`model.features[-1]`) |
| **Input Validation** | 5-signal pipeline (HSV + RGB + Laplacian + Edge + Entropy) |
| **UI Framework** | Gradio 6.x (premium dark-mode design) |

---

## 🏗️ Project Structure

```
RetinaScan_AI/
├── app.py                             # Main UI application (Gradio)
├── best_convnext_tiny_oct2017.pth     # Trained model checkpoint (~334 MB)
├── demo_image/                        # Sample OCT images (1 per class)
│   ├── CNV-1016042-1.jpeg
│   ├── DME-119840-1.jpeg
│   ├── DRUSEN-1246453-1.jpeg
│   └── NORMAL-1073137-1.jpeg
├── screenshots/                       # Report screenshots (auto-generated)
├── _test_predictions.py               # Automated verification tests
├── _capture_screenshots.py            # Screenshot automation (Selenium)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- ~1 GB disk space (model checkpoint + dependencies)

### 1. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

The web UI will auto-open at **http://127.0.0.1:7860**

---

## 🧪 How It Works

### Classification Pipeline
1. **Input Validation** — Five-signal check rejects non-OCT images:
   - HSV saturation check (OCT scans are grayscale, S < 25)
   - RGB channel similarity check (R ≈ G ≈ B for grayscale)
   - Laplacian variance check (rejects screenshots/documents with sharp text)
   - Edge orientation analysis (OCT scans have horizontal retinal layers)
   - Model confidence + Shannon entropy analysis (flags uncertain predictions)
2. **Preprocessing** — Convert to grayscale → resize to 224×224 → ImageNet normalization
3. **Inference** — ConvNeXt-Tiny forward pass → softmax probabilities
4. **Grad-CAM** — Hook `model.features[-1]` → weighted activation map → heatmap overlay
5. **Clinical Report** — Formatted diagnosis with confidence, clinical significance, and attention region interpretation

### Classes
| Class | Full Name | Clinical Significance |
|---|---|---|
| **CNV** | Choroidal Neovascularization | Abnormal blood vessel growth; wet AMD indicator |
| **DME** | Diabetic Macular Edema | Fluid accumulation from diabetes complications |
| **DRUSEN** | Drusen Deposits | Early AMD indicator; sub-RPE deposits |
| **NORMAL** | Healthy Retina | No pathological features detected |

---

## ✅ Verification

Run the automated test suite to verify end-to-end functionality:

```bash
python _test_predictions.py
```

Expected output:
```
============================================================
RUNTIME VERIFICATION TESTS
============================================================

--- Test Suite 1: Prediction Accuracy ---
  [PASS] CNV: predicted=CNV (100.00%), overlay=YES, why_text=OK
  [PASS] DME: predicted=DME (100.00%), overlay=YES, why_text=OK
  [PASS] DRUSEN: predicted=DRUSEN (98.71%), overlay=YES, why_text=OK
  [PASS] NORMAL: predicted=NORMAL (99.99%), overlay=YES, why_text=OK

--- Test Suite 2: Status Banner Check ---
  [PASS] CNV: banner=VALID
  [PASS] DME: banner=VALID
  [PASS] DRUSEN: banner=VALID
  [PASS] NORMAL: banner=VALID

--- Test Suite 3: Non-OCT Input Rejection ---
  [PASS] Non-OCT: rejected=YES, no_prediction=YES

--- Test Suite 4: Input Validation Function ---
  [PASS] CNV: valid=True
  [PASS] DME: valid=True
  [PASS] DRUSEN: valid=True
  [PASS] NORMAL: valid=True
  [PASS] Non-OCT: valid=False, hard_reject=True

============================================================
ALL TESTS PASSED SUCCESSFULLY!
============================================================
```

---

## 🔧 Technical Details

### Model Architecture
- **Base:** ConvNeXt-Tiny (pretrained on ImageNet-1K)
- **Head:** Linear(768, 4) replacing the original 1000-class classifier
- **Training:** Fine-tuned on Kermany OCT2017 dataset
- **Accuracy:** 99.90% on the 968-image held-out test set

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
| Confidence + Entropy | conf < 70% AND entropy > 75% max | Flag uncertain predictions |

---

## 👥 Team

- **Murali A** — Model training, Grad-CAM integration
- **Niranjan T** — Data preprocessing, evaluation
- **Praveen K** — UI development, input validation

**Guide:** Dr. C. Santhosh Kumar  
**Institution:** Sona College of Technology

---

## 📚 References

1. Kermany, D. S., et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 172(5), 1122–1131.
2. Liu, Z., et al. (2022). "A ConvNet for the 2020s." *CVPR 2022*.
3. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." *ICCV 2017*.
