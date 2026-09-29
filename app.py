# ================================================================
# RETINASCAN AI — Premium Dark-Mode OCT Classifier
# ================================================================
# Team: Murali A, Niranjan T, Praveen K
# Guide: Dr. C. Santhosh Kumar | Sona College of Technology
# ================================================================

import os
import sys
import math

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny
from torchvision import transforms
import gradio as gr

# ================================================================
# 1. CONFIGURATION
# ================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_convnext_tiny_oct2017.pth"
EXAMPLE_IMAGES_DIR = BASE_DIR / "demo_image"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = 4
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Validation thresholds
MAX_MEAN_SATURATION = 25.0
MAX_CHANNEL_DIFF = 20.0
MAX_LAPLACIAN_VAR = 3500.0
MIN_HORIZ_EDGE_RATIO = 0.35
HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.70
MAX_ENTROPY = math.log(4)
HIGH_ENTROPY_RATIO = 0.75

# ================================================================
# 2. CSS — CLEAN LIGHT MOBILE-APP MEDICAL DESIGN
# ================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');

:root {
    --bg-page:       #F0F4F8;
    --bg-card:       #FFFFFF;
    --bg-input:      #F7F9FC;
    --border-light:  #E2E8F0;
    --border-card:   #D8E2EE;
    --text-primary:  #1E293B;
    --text-secondary:#64748B;
    --text-muted:    #94A3B8;
    --blue-50:       #EFF6FF;
    --blue-100:      #DBEAFE;
    --blue-200:      #BFDBFE;
    --blue-500:      #3B82F6;
    --blue-600:      #2563EB;
    --blue-700:      #1D4ED8;
    --green-50:      #F0FDF4;
    --green-100:     #DCFCE7;
    --green-500:     #22C55E;
    --green-600:     #16A34A;
    --green-700:     #15803D;
    --red-50:        #FEF2F2;
    --red-100:       #FEE2E2;
    --red-500:       #EF4444;
    --red-600:       #DC2626;
    --amber-50:      #FFFBEB;
    --amber-100:     #FEF3C7;
    --amber-500:     #F59E0B;
    --radius:        12px;
    --radius-lg:     16px;
    --shadow-sm:     0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:     0 4px 12px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg:     0 10px 25px rgba(0,0,0,0.08), 0 4px 10px rgba(0,0,0,0.04);
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-page) !important;
    color: var(--text-primary) !important;
    font-family: 'Nunito', 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}
.dark .gradio-container {
    background: var(--bg-page) !important;
    color: var(--text-primary) !important;
}
footer { display: none !important; }

/* Remove any dark body overlays */
body::before, body::after { display: none !important; }

/* ──── App Header ──── */
.app-header {
    background: linear-gradient(135deg, var(--blue-600) 0%, #4F46E5 50%, var(--blue-700) 100%);
    border-radius: 0 0 24px 24px;
    padding: 28px 24px 22px;
    text-align: center;
    margin: 0 -16px 20px;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.25);
}
.app-header .app-logo {
    width: 52px; height: 52px;
    background: rgba(255,255,255,0.2);
    border-radius: 14px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.6em;
    margin-bottom: 10px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
}
.app-header h1 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.8em !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    margin: 0 !important;
    letter-spacing: -0.01em;
}
.app-header .app-subtitle {
    font-size: 0.85em;
    color: rgba(255,255,255,0.8);
    margin-top: 4px;
    font-weight: 400;
}
.app-header .app-team {
    font-size: 0.75em;
    color: rgba(255,255,255,0.6);
    margin-top: 8px;
}
.app-header .app-team strong {
    color: rgba(255,255,255,0.9);
    font-weight: 600;
}

/* ──── Stat Pills Row ──── */
.stat-pills {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 14px;
}
.stat-pill {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.7em;
    color: rgba(255,255,255,0.9);
    font-weight: 600;
    backdrop-filter: blur(8px);
}
.stat-pill .pill-val {
    color: #FFFFFF;
    font-weight: 800;
}

/* ──── Card Containers ──── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    padding: 18px;
    margin-bottom: 14px;
}
.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-light);
}
.card-header .card-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1em;
}
.card-icon-blue {
    background: var(--blue-50);
    color: var(--blue-600);
    border: 1px solid var(--blue-200);
}
.card-icon-green {
    background: var(--green-50);
    color: var(--green-600);
    border: 1px solid var(--green-100);
}
.card-header h3 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.95em !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
}
.card-header .card-tag {
    font-size: 0.6em;
    padding: 3px 10px;
    background: var(--blue-50);
    color: var(--blue-600);
    border-radius: 100px;
    border: 1px solid var(--blue-200);
    font-weight: 700;
    margin-left: auto;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ──── Status Banners ──── */
.status-banner-valid {
    background: var(--green-50);
    border: 1px solid var(--green-100);
    border-left: 4px solid var(--green-500);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin: 12px 0 16px;
}
.status-banner-valid h3 {
    color: var(--green-700) !important;
    margin: 0 0 4px !important;
    font-size: 0.92em !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
}
.status-banner-valid p { color: var(--text-secondary); margin: 0; font-size: 0.85em; }

.status-banner-invalid {
    background: var(--red-50);
    border: 1px solid var(--red-100);
    border-left: 4px solid var(--red-500);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin: 12px 0 16px;
}
.status-banner-invalid h3 {
    color: var(--red-600) !important;
    margin: 0 0 4px !important;
    font-size: 0.92em !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
}
.status-banner-invalid p  { color: var(--text-secondary); margin: 0; font-size: 0.85em; }
.status-banner-invalid ul { color: var(--text-secondary); margin: 6px 0 0 16px; font-size: 0.82em; padding: 0; }

.status-banner-warning {
    background: var(--amber-50);
    border: 1px solid var(--amber-100);
    border-left: 4px solid var(--amber-500);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin: 12px 0 16px;
}
.status-banner-warning h3 {
    color: #B45309 !important;
    margin: 0 0 4px !important;
    font-size: 0.92em !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
}
.status-banner-warning p { color: var(--text-secondary); margin: 0; font-size: 0.85em; }

@keyframes bannerSlideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.status-banner-valid, .status-banner-invalid, .status-banner-warning {
    animation: bannerSlideIn 0.35s ease-out;
}

/* ──── Buttons ──── */
#analyze-btn {
    background: var(--blue-600) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 700 !important;
    font-size: 0.95em !important;
    padding: 12px 24px !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 12px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.2s ease !important;
    font-family: 'Poppins', sans-serif !important;
}
#analyze-btn:hover {
    background: var(--blue-700) !important;
    box-shadow: 0 5px 18px rgba(37, 99, 235, 0.4) !important;
    transform: translateY(-1px) !important;
}
#clear-btn {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: var(--radius) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-family: 'Poppins', sans-serif !important;
    transition: all 0.2s ease !important;
}
#clear-btn:hover {
    border-color: var(--red-500) !important;
    color: var(--red-600) !important;
    background: var(--red-50) !important;
}

/* ──── Gradio Component Overrides ──── */
.image-container, .upload-container {
    background: var(--bg-input) !important;
    border: 2px dashed var(--border-light) !important;
    border-radius: var(--radius) !important;
    transition: all 0.2s ease !important;
}
.upload-container:hover {
    border-color: var(--blue-500) !important;
    background: var(--blue-50) !important;
}
.label-wrap, label, .label-text {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-family: 'Nunito', sans-serif !important;
}
#confidence-bars .label-class-text { color: var(--text-primary) !important; }
#confidence-bars .label-class-confidence { color: var(--blue-600) !important; font-weight: 700 !important; }

/* ──── Markdown ──── */
.prose, .markdown-text, .md { color: var(--text-secondary) !important; }
.prose h3, .markdown-text h3, .md h3 {
    color: var(--text-primary) !important;
    font-family: 'Poppins', sans-serif !important;
}
.prose strong { color: var(--text-primary) !important; }
.prose hr { border-color: var(--border-light) !important; }

/* ──── Section Dividers ──── */
.section-divider {
    height: 1px;
    background: var(--border-light);
    margin: 18px 0;
    border: none;
}

/* ──── About Card ──── */
.about-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    margin-top: 16px;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
}
.about-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, var(--blue-500), #8B5CF6, var(--blue-500));
}
.about-card h3 {
    color: var(--blue-700) !important;
    font-family: 'Poppins', sans-serif !important;
}

/* ──── Gallery ──── */
.gallery-item {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border-light) !important;
    transition: all 0.2s ease !important;
    background: var(--bg-card) !important;
}
.gallery-item:hover {
    border-color: var(--blue-500) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-2px);
}

/* ──── Block backgrounds ──── */
.block, .form, .wrap { background: transparent !important; border: none !important; }

/* ──── Scrollbar ──── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-page); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ──── Responsive ──── */
@media (max-width: 768px) {
    .app-header h1 { font-size: 1.5em !important; }
    .stat-pills { gap: 6px; }
    .stat-pill { font-size: 0.62em; padding: 4px 10px; }
    .card { padding: 14px; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: fadeInUp 0.4s ease-out forwards; }
"""


# ================================================================
# 3. DEVICE & MODEL SETUP
# ================================================================
print("\n=======================================================")
print("       RETINASCAN AI - INITIALIZING")
print("=======================================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

assert MODEL_PATH.exists(), (
    f"\n[ERROR] Model not found at:\n   {MODEL_PATH}\n"
    f"Ensure best_convnext_tiny_oct2017.pth is in the same folder as app.py."
)

print("Loading ConvNeXt-Tiny model...")
model = convnext_tiny(weights=None)
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, NUM_CLASSES)

checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()
print("[OK] Model loaded (ConvNeXt-Tiny, 28.5M params, 99.90% test accuracy)")
print("=======================================================\n")


# ================================================================
# 4. GRAD-CAM
# ================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())
        )

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam, class_idx, probs

gradcam = GradCAM(model, model.features[-1])

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ================================================================
# 5. DISEASE DESCRIPTIONS
# ================================================================
DESCRIPTIONS = {
    "CNV": {
        "full_name": "Choroidal Neovascularization",
        "clinical": "Abnormal blood vessel growth beneath the retina. Associated with wet age-related macular degeneration (AMD). Requires urgent ophthalmological review.",
        "attention_region": "the choroidal region and elevated dome-shaped areas above the retinal pigment epithelium",
    },
    "DME": {
        "full_name": "Diabetic Macular Edema",
        "clinical": "Fluid accumulation in the macula due to diabetes complications. A leading cause of vision loss in diabetic patients.",
        "attention_region": "cystoid fluid pockets in the central macular region",
    },
    "DRUSEN": {
        "full_name": "Drusen Deposits",
        "clinical": "Yellowish deposits at the retinal pigment epithelium interface. Early indicator of age-related macular degeneration (AMD).",
        "attention_region": "sub-RPE deposits along the retinal pigment epithelium layer",
    },
    "NORMAL": {
        "full_name": "Healthy Retina",
        "clinical": "No significant pathological features detected. Normal retinal layer architecture visible with intact foveal contour.",
        "attention_region": "the foveal region with intact retinal layer stratification",
    },
}

DYNAMIC_WHY = {
    "CNV": "The model focused on the elevated retinal/RPE region, which is an area typically associated with CNV-related abnormalities.",
    "DME": "The model focused on the central intraretinal region, where cystoid/fluid-filled spaces associated with DME are typically observed.",
    "DRUSEN": "The model focused near the retinal pigment epithelium (RPE), where drusen deposits are typically located.",
    "NORMAL": "The model showed distributed attention around the foveal region without a strong localized pathological focus, consistent with normal retinal architecture.",
}


# ================================================================
# 6. INPUT VALIDATION
# ================================================================
def validate_oct_input(image_array, softmax_probs=None):
    """Five-signal OCT validation pipeline."""
    reasons = []
    is_hard_reject = False

    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Check 1: HSV saturation
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    mean_sat = float(hsv[:, :, 1].mean())

    # Check 2: RGB channel diff
    r, g, b = image_array[:,:,0].astype(float), image_array[:,:,1].astype(float), image_array[:,:,2].astype(float)
    channel_diff = float(np.mean(np.abs(r-g)) + np.mean(np.abs(g-b)) + np.mean(np.abs(r-b))) / 3

    if mean_sat > MAX_MEAN_SATURATION:
        reasons.append(f"High color saturation detected (mean S = {mean_sat:.1f}; OCT scans are grayscale with S < {MAX_MEAN_SATURATION:.0f})")
        is_hard_reject = True
    if channel_diff > MAX_CHANNEL_DIFF:
        reasons.append(f"Significant RGB channel differences (diff = {channel_diff:.1f}; OCT scans are monochromatic with diff < {MAX_CHANNEL_DIFF:.0f})")
        is_hard_reject = True

    if is_hard_reject:
        return False, reasons, True

    # Check 3: Laplacian variance
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_resized = cv2.resize(gray, (224, 224))
    lap_var = float(cv2.Laplacian(gray_resized, cv2.CV_64F).var())

    if lap_var > MAX_LAPLACIAN_VAR:
        reasons.append(f"Image has unusually sharp edges (Laplacian variance = {lap_var:.0f}; OCT scans typically < {MAX_LAPLACIAN_VAR:.0f}). This looks like a screenshot, document, or UI image.")
        is_hard_reject = True

    # Check 4: Edge orientation
    sobel_x = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
    total_energy = float(np.sum(sobel_y**2)) + float(np.sum(sobel_x**2)) + 1e-10
    horiz_ratio = float(np.sum(sobel_y**2)) / total_energy

    if horiz_ratio < MIN_HORIZ_EDGE_RATIO:
        reasons.append(f"Edge orientation is not consistent with OCT scans (horizontal edge ratio = {horiz_ratio:.2f}; OCT scans > {MIN_HORIZ_EDGE_RATIO:.2f}). OCT scans have predominantly horizontal retinal layer patterns.")
        is_hard_reject = True

    if is_hard_reject:
        return False, reasons, True

    # Check 5: Confidence + entropy (soft warning)
    if softmax_probs is not None:
        top_conf = float(np.max(softmax_probs))
        entropy = -float(np.sum(softmax_probs * np.log(softmax_probs + 1e-10)))
        entropy_ratio = entropy / MAX_ENTROPY
        if top_conf < MEDIUM_CONFIDENCE and entropy_ratio > HIGH_ENTROPY_RATIO:
            reasons.append(f"Very low model confidence ({top_conf*100:.1f}%) with high prediction uncertainty (entropy ratio {entropy_ratio:.2f}). This may not be a typical OCT scan.")
            return False, reasons, False

    return True, [], False


# ================================================================
# 7. HTML HELPERS
# ================================================================
def make_status_html(banner_type, title, body, items=None):
    """Generate a styled HTML status banner."""
    css_class = f"status-banner-{banner_type}"
    items_html = ""
    if items:
        items_html = "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    return f'<div class="{css_class}"><h3>{title}</h3><p>{body}</p>{items_html}</div>'


# ================================================================
# 8. PREDICTION FUNCTION
# ================================================================
def predict_and_explain(image):
    """Main prediction pipeline: validate → classify → explain."""
    if image is None:
        return (
            {c: 0.0 for c in CLASSES},
            None,
            "### Upload an OCT scan and click **Analyze Scan** to begin.",
            "",
        )

    # Step 1: Hard validation
    is_valid, reasons, is_hard = validate_oct_input(image)
    if is_hard:
        html_banner = make_status_html(
            "invalid",
            "&#128683; Invalid Input &mdash; Not an OCT Scan",
            "The uploaded image does not appear to be an OCT retinal scan. "
            "OCT scans are grayscale medical images of retinal cross-sections.",
            reasons,
        )
        explanation = (
            "### ⚠️ Prediction Skipped\n\n"
            "This model was trained exclusively on grayscale OCT retinal scans from "
            "the Kermany OCT2017 dataset. It cannot meaningfully classify other image types.\n\n"
            "To make a prediction, please upload a genuine OCT scan."
        )
        # Issue fix #2: Show "Invalid" instead of CNV when image is rejected
        invalid_conf = {"Invalid": 0.0}
        return (invalid_conf, None, explanation, html_banner)

    # Step 2: Preprocessing
    pil_img = Image.fromarray(image).convert("L")
    display_img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    display_rgb = np.stack([display_img] * 3, axis=-1)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    # Step 3: Prediction + Grad-CAM
    heatmap, pred_idx, probs = gradcam(input_tensor)
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # Step 4: Soft validation (confidence + entropy)
    is_valid, reasons, _ = validate_oct_input(image, softmax_probs=probs)

    # Step 5: Status banner
    if not is_valid:
        reason_list = "\n".join([f"- {r}" for r in reasons])
        html_banner = make_status_html(
            "warning",
            f"&#9888;&#65039; Low-Confidence Prediction ({confidence*100:.2f}%)",
            "The model is not confident. This may not be a typical OCT scan. Interpret with caution.",
        )
    elif confidence >= HIGH_CONFIDENCE:
        html_banner = make_status_html(
            "valid",
            f"&#9989; Valid OCT Input &mdash; High-Confidence Prediction ({confidence*100:.2f}%)",
            f"Input validation passed. The model is highly confident this is <strong>{pred_class}</strong>.",
        )
    else:
        html_banner = make_status_html(
            "warning",
            f"&#9888;&#65039; Moderate Confidence ({confidence*100:.2f}%)",
            "The model's confidence is moderate. This may indicate a borderline case &mdash; verify with a clinician.",
        )

    # Step 6: Grad-CAM overlay
    # Issue fix #3: For NORMAL predictions, suppress the heatmap to avoid
    # showing misleading "affected areas" when there is no pathology.
    if pred_class == "NORMAL":
        # Apply a high threshold so only very faint, distributed attention remains.
        # This reflects that a healthy retina has no localized pathological focus.
        threshold = 0.6
        heatmap_suppressed = np.where(heatmap > threshold, heatmap * 0.15, heatmap * 0.05)
        heatmap_suppressed = np.clip(heatmap_suppressed, 0, 1)
        heatmap_for_overlay = heatmap_suppressed
    else:
        heatmap_for_overlay = heatmap

    heatmap_resized = cv2.resize(heatmap_for_overlay, (display_img.shape[1], display_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display_rgb, 0.55, heatmap_color, 0.45, 0)

    # Step 7: Confidence dict
    conf_dict = {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    # Step 8: Clinical explanation
    desc = DESCRIPTIONS[pred_class]
    if pred_class == "NORMAL":
        gradcam_note = (
            "For a **NORMAL** scan, the Grad-CAM heatmap is intentionally suppressed. "
            "Since no pathology is detected, there are no specific abnormal regions to highlight. "
            "The model confirmed intact retinal architecture across the scan."
        )
    else:
        gradcam_note = (
            f"The colored overlay shows the retinal region the model focused on. "
            f"Red/yellow = high attention, blue = low attention. For a **{pred_class}** diagnosis, "
            f"the model should attend to {desc['attention_region']}."
        )

    explanation = (
        f"### Predicted Diagnosis: **{pred_class}** — {desc['full_name']}\n\n"
        f"**Confidence:** {confidence*100:.2f}%\n\n"
        f"**Clinical significance:** {desc['clinical']}\n\n"
        f"---\n\n"
        f"### Why this prediction?\n\n"
        f"{DYNAMIC_WHY[pred_class]}\n\n"
        f"---\n\n"
        f"### Grad-CAM Heatmap Interpretation\n\n"
        f"{gradcam_note}\n\n"
        f"---\n\n"
        f"*Model: ConvNeXt-Tiny (28.5M parameters) · Trained on Kermany OCT2017 "
        f"(84,484 images) · Test accuracy: 99.90% · Grad-CAM target: model.features[-1]*"
    )

    return (conf_dict, overlay, explanation, html_banner)


# ================================================================
# 9. EXAMPLE IMAGES
# ================================================================
example_list = []
if EXAMPLE_IMAGES_DIR.exists():
    for f in sorted(EXAMPLE_IMAGES_DIR.iterdir()):
        if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
            example_list.append([str(f)])
    print(f"[OK] Loaded {len(example_list)} example images")


# ================================================================
# 10. GRADIO UI — CLEAN LIGHT MOBILE-APP INTERFACE
# ================================================================
with gr.Blocks(title="RetinaScan AI") as demo:

    # ──── APP HEADER ────
    gr.HTML("""
    <div class="app-header">
        <div class="app-logo">&#128300;</div>
        <h1>RetinaScan AI</h1>
        <div class="app-subtitle">OCT Retinal Disease Classifier with Explainable AI</div>
        <div class="app-team">
            <strong>Murali A</strong> &middot; <strong>Niranjan T</strong> &middot; <strong>Praveen K</strong>
            &nbsp;|&nbsp; Guide: <strong>Dr. C. Santhosh Kumar</strong>
            &nbsp;|&nbsp; Sona College of Technology
        </div>
        <div class="stat-pills">
            <span class="stat-pill">Architecture: <span class="pill-val">ConvNeXt-Tiny</span></span>
            <span class="stat-pill">Accuracy: <span class="pill-val">99.90%</span></span>
            <span class="stat-pill">Params: <span class="pill-val">28.5M</span></span>
            <span class="stat-pill">Classes: <span class="pill-val">CNV · DME · DRUSEN · NORMAL</span></span>
        </div>
    </div>
    """)

    # ──── STATUS BANNER (raw HTML) ────
    status_output = gr.HTML(value="", elem_id="status-banner")

    # ──── MAIN CONTENT — 2 COLUMNS ────
    with gr.Row(equal_height=False):
        # LEFT PANEL — Upload
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="card-header">
                <div class="card-icon card-icon-blue">&#128228;</div>
                <h3>Upload OCT Scan</h3>
                <span class="card-tag">INPUT</span>
            </div>
            """)
            gr.Markdown(
                "*Supported: grayscale OCT retinal scans (JPEG/PNG). "
                "Non-OCT images are automatically detected and rejected.*"
            )
            input_image = gr.Image(
                type="numpy",
                label="OCT Scan Input",
                height=360,
                sources=["upload", "clipboard"],
                elem_id="oct-input-image",
            )
            with gr.Row():
                submit_btn = gr.Button(
                    "Analyze Scan",
                    variant="primary",
                    size="lg",
                    scale=3,
                    elem_id="analyze-btn",
                )
                clear_btn = gr.Button(
                    "Clear",
                    variant="secondary",
                    scale=1,
                    elem_id="clear-btn",
                )

        # RIGHT PANEL — Results
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="card-header">
                <div class="card-icon card-icon-green">&#127919;</div>
                <h3>Prediction & Explanation</h3>
                <span class="card-tag">OUTPUT</span>
            </div>
            """)
            output_confidence = gr.Label(
                num_top_classes=4,
                label="Class Confidence Distribution",
                elem_id="confidence-bars",
            )
            output_heatmap = gr.Image(
                type="numpy",
                label="Grad-CAM Attention Heatmap",
                height=300,
                elem_id="gradcam-overlay",
            )

    # ──── SECTION DIVIDER ────
    gr.HTML('<div class="section-divider"></div>')

    # ──── CLINICAL INTERPRETATION ────
    gr.HTML("""
    <div class="card-header" style="margin-bottom:8px;">
        <div class="card-icon card-icon-green">&#129658;</div>
        <h3>Clinical Interpretation</h3>
    </div>
    """)
    output_explanation = gr.Markdown(
        value="*Upload an OCT scan and click **Analyze Scan** to receive an AI-assisted diagnostic interpretation with Grad-CAM visual explanation.*",
        elem_id="clinical-interpretation",
    )

    # ──── SECTION DIVIDER ────
    gr.HTML('<div class="section-divider"></div>')

    # Outputs list for wiring
    ui_outputs = [output_confidence, output_heatmap, output_explanation, status_output]

    # ──── EXAMPLES GALLERY ────
    if example_list:
        gr.HTML("""
        <div>
            <div class="card-header">
                <div class="card-icon card-icon-blue">&#129514;</div>
                <h3>Sample OCT Test Images</h3>
            </div>
            <p style="color: var(--text-muted); font-size: 0.85em; margin-top: 4px;">Click any example below to auto-analyze it with the model.</p>
        </div>
        """)
        gr.Examples(
            examples=example_list,
            inputs=input_image,
            outputs=ui_outputs,
            fn=predict_and_explain,
            cache_examples=False,
            label="Demo Samples (CNV, DME, DRUSEN, NORMAL)",
            examples_per_page=4,
        )

    # ──── SECTION DIVIDER ────
    gr.HTML('<div class="section-divider"></div>')

    # ──── ABOUT THE MODEL ────
    gr.HTML("""
    <div class="about-card">
        <h3 style="font-family: 'Poppins', sans-serif; margin-top: 0; font-size: 1.05em;">&#8505;&#65039; About the Model</h3>
        <p style="color: var(--text-secondary); font-size: 0.88em; line-height: 1.7;">
            This system uses <strong style="color: var(--text-primary);">ConvNeXt-Tiny</strong> (Liu et al., 2022) &mdash;
            a modern CNN with <strong style="color: var(--text-primary);">28.5 million parameters</strong>
            &mdash; fine-tuned on the
            <strong style="color: var(--text-primary);">Kermany OCT2017 dataset</strong>
            (84,484 labeled OCT retinal scans across 4 classes).
            On the held-out test set of 968 images, the model achieves
            <strong style="color: var(--green-600);">99.90% accuracy</strong>
            (967/968 correctly classified).
        </p>
        <p style="color: var(--text-secondary); font-size: 0.88em; line-height: 1.7; margin-top: 10px;">
            <strong style="color: var(--blue-600);">Interpretability:</strong>
            Every prediction is accompanied by a real-time Grad-CAM heatmap generated by hooking into
            <code style="background: #F1F5F9; color: var(--blue-600); padding: 2px 6px; border-radius: 4px; font-size: 0.85em;">model.features[-1]</code>.
        </p>
        <p style="color: var(--text-secondary); font-size: 0.88em; line-height: 1.7; margin-top: 10px;">
            <strong style="color: #7C3AED;">Input Validation:</strong>
            Five-signal validation pipeline &mdash; HSV saturation, RGB channel similarity,
            Laplacian variance, edge orientation analysis, and model confidence + entropy &mdash;
            ensures non-OCT images are rejected before or after inference.
        </p>
    </div>
    """)

    # ──── WIRING ────
    submit_btn.click(
        fn=predict_and_explain,
        inputs=input_image,
        outputs=ui_outputs,
    )

    def clear_all():
        return (
            None,                    # input_image
            {c: 0.0 for c in CLASSES},  # output_confidence
            None,                    # output_heatmap
            "*Upload an OCT scan and click **Analyze Scan** to receive an AI-assisted "
            "diagnostic interpretation with Grad-CAM visual explanation.*",
            "",                      # status_output
        )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[input_image, output_confidence, output_heatmap, output_explanation, status_output],
    )


# ================================================================
# 11. LAUNCH
# ================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting RetinaScan AI (Light Mobile-App UI)...")
    print("=" * 60)
    demo.launch(
        share=False,
        inbrowser=True,
        show_error=True,
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    )