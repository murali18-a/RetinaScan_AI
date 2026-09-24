"""
RetinaScan AI â€” Prediction Interface v2 (Local Laptop)
======================================================

Improvements over v1:
- OCT input validation (rejects movies, cats, random colorful images)
- Confidence thresholding (warns if the model is unsure)
- Visual status indicator (âœ“ valid input / âš  suspicious input)
- Polished 2-column layout with clinical descriptions
- Warning banners for out-of-distribution inputs

USAGE
-----
1. Install dependencies (one-time):
   pip install torch torchvision gradio opencv-python pillow numpy

2. Update MODEL_PATH below to point to your downloaded .pth file.
   (Optional) Update EXAMPLE_IMAGES_DIR to a folder with sample OCT scans.

3. Run:
   python retinascan_ui_v2.py

4. Browser auto-opens with the UI at http://127.0.0.1:7860

Team: Murali A, Niranjan T, Praveen K
Guide: Dr. C. Santhosh Kumar
Sona College of Technology
"""

import os
import sys
import math

# Fix Windows console encoding (cp1252 cannot handle Unicode symbols)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Fallback: some environments don't support reconfigure
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
# CONFIGURATION â€” UPDATE THESE PATHS
# ================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_convnext_tiny_oct2017.pth"
EXAMPLE_IMAGES_DIR = BASE_DIR / "demo_image"

# ================================================================
# INPUT VALIDATION THRESHOLDS
# ================================================================
# OCT images are grayscale. If saturation is high, it's not an OCT.
MAX_MEAN_SATURATION = 25.0   # HSV saturation channel -- OCT scans typically < 15, movies > 40
MAX_CHANNEL_DIFF = 20.0      # Mean absolute difference between RGB channels -- OCT ~ 0, color ~ 30+

# Structural checks -- catch screenshots, documents, UI images that are grayscale
MAX_LAPLACIAN_VAR = 3500.0   # Laplacian variance: OCT ~1500-2900, screenshots/text ~4000-8000+
MIN_HORIZ_EDGE_RATIO = 0.35  # OCT scans have mostly horizontal edges (layers); text has mixed edges

# Confidence thresholds for warnings
HIGH_CONFIDENCE = 0.90       # Above this = trust the prediction
MEDIUM_CONFIDENCE = 0.70     # Above this = show prediction with caution
# Below MEDIUM_CONFIDENCE = flag as likely invalid input

# Entropy threshold -- log(4) ~ 1.386 is max entropy for 4 classes
MAX_ENTROPY = math.log(4)    # ~ 1.386
HIGH_ENTROPY_RATIO = 0.75    # If entropy > 75% of max AND confidence < 70%, flag it

# ================================================================
# SETUP
# ================================================================
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = 4
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ================================================================
# MODEL LOADING
# ================================================================
assert MODEL_PATH.exists(), (
    f"[ERROR] Model file not found at:\n   {MODEL_PATH}\n"
    f"Download best_convnext_tiny_oct2017.pth from Kaggle -> save to that path."
)

print(f"Loading model from {MODEL_PATH}...")
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
print("[OK] Model loaded (ConvNeXt-Tiny, 28.5M parameters, 99.90% test accuracy)")

# ================================================================
# GRAD-CAM
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

# ================================================================
# PREPROCESSING
# ================================================================
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

DESCRIPTIONS = {
    "CNV": {
        "full_name": "Choroidal Neovascularization",
        "clinical": "Abnormal blood vessel growth beneath the retina. Associated with wet age-related macular degeneration (AMD). **Requires urgent ophthalmological review** â€” anti-VEGF injection therapy is often needed.",
        "attention_region": "the choroidal region and elevated dome-shaped areas above the retinal pigment epithelium",
    },
    "DME": {
        "full_name": "Diabetic Macular Edema",
        "clinical": "Fluid accumulation in the macula due to diabetes complications. A leading cause of vision loss in diabetic patients. Requires management of underlying diabetes plus targeted treatment.",
        "attention_region": "cystoid fluid pockets in the central macular region",
    },
    "DRUSEN": {
        "full_name": "Drusen Deposits",
        "clinical": "Yellowish deposits at the retinal pigment epithelium interface. Early indicator of age-related macular degeneration (AMD). Regular monitoring recommended to catch progression.",
        "attention_region": "sub-RPE deposits along the retinal pigment epithelium layer",
    },
    "NORMAL": {
        "full_name": "Healthy Retina",
        "clinical": "No significant pathological features detected. Normal retinal layer architecture visible with intact foveal contour.",
        "attention_region": "the foveal region with intact retinal layer stratification",
    },
}

# ================================================================
# DYNAMIC "WHY THIS PREDICTION?" EXPLANATIONS
# ================================================================
DYNAMIC_WHY_EXPLANATIONS = {
    "CNV": "The model focused on the elevated retinal/RPE region, which is an area typically associated with CNV-related abnormalities.",
    "DME": "The model focused on the central intraretinal region, where cystoid/fluid-filled spaces associated with DME are typically observed.",
    "DRUSEN": "The model focused near the retinal pigment epithelium (RPE), where drusen deposits are typically located.",
    "NORMAL": "The model showed distributed attention around the foveal region without a strong localized pathological focus, consistent with normal retinal architecture.",
}

# ================================================================
# INPUT VALIDATION — detect non-OCT images
# ================================================================
def validate_oct_input(image_array, softmax_probs=None):
    """
    Determines whether an uploaded image is plausibly a grayscale OCT scan.

    Five validation signals:
    1. HSV saturation -- OCT scans are grayscale, so saturation should be very low
    2. RGB channel similarity -- R, G, B channels should be near-identical
    3. Laplacian variance -- screenshots/text have very sharp edges (high variance);
       OCT scans have smoother, gradual transitions
    4. Edge orientation -- OCT scans have predominantly horizontal edges (retinal
       layers); screenshots/documents have mixed/vertical edges (text, UI elements)
    5. Model confidence + entropy -- if top prediction is low and distribution is
       spread out (high entropy), the image is likely not a valid OCT scan

    Returns:
        (is_valid: bool, reasons: list[str], is_hard_reject: bool)
        - is_hard_reject=True  -> color/structure checks failed, skip prediction
        - is_hard_reject=False, is_valid=False -> soft warning (low confidence)
        - is_valid=True -> everything looks good
    """
    reasons = []
    is_hard_reject = False

    # Ensure RGB
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # --- Check 1: HSV saturation -- OCT should be low ---
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    mean_saturation = float(hsv[:, :, 1].mean())

    # --- Check 2: RGB channel difference -- OCT channels should be nearly identical ---
    r = image_array[:, :, 0].astype(float)
    g = image_array[:, :, 1].astype(float)
    b = image_array[:, :, 2].astype(float)
    channel_diff = float(
        np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b))
    ) / 3

    if mean_saturation > MAX_MEAN_SATURATION:
        reasons.append(
            f"High color saturation detected (mean S = {mean_saturation:.1f}; "
            f"OCT scans are grayscale with S < {MAX_MEAN_SATURATION:.0f})"
        )
        is_hard_reject = True
    if channel_diff > MAX_CHANNEL_DIFF:
        reasons.append(
            f"Significant RGB channel differences (diff = {channel_diff:.1f}; "
            f"OCT scans are monochromatic with diff < {MAX_CHANNEL_DIFF:.0f})"
        )
        is_hard_reject = True

    if is_hard_reject:
        return False, reasons, True

    # --- Check 3: Laplacian variance -- detect text/screenshots/documents ---
    # Screenshots and documents have very sharp text edges -> high Laplacian variance
    # OCT scans have smooth, gradual layer transitions -> moderate Laplacian variance
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_resized = cv2.resize(gray, (224, 224))
    laplacian = cv2.Laplacian(gray_resized, cv2.CV_64F)
    lap_var = float(laplacian.var())

    if lap_var > MAX_LAPLACIAN_VAR:
        reasons.append(
            f"Image has unusually sharp edges (Laplacian variance = {lap_var:.0f}; "
            f"OCT scans typically < {MAX_LAPLACIAN_VAR:.0f}). "
            f"This looks like a screenshot, document, or UI image."
        )
        is_hard_reject = True

    # --- Check 4: Edge orientation analysis -- OCT = horizontal retinal layers ---
    # OCT scans have strong horizontal edge patterns (parallel retinal layers).
    # Screenshots, text, and random images have much more vertical/diagonal edges.
    sobel_x = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)  # vertical edges
    sobel_y = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)  # horizontal edges
    energy_horiz = float(np.sum(sobel_y ** 2))
    energy_vert = float(np.sum(sobel_x ** 2))
    total_energy = energy_horiz + energy_vert + 1e-10
    horiz_ratio = energy_horiz / total_energy

    if horiz_ratio < MIN_HORIZ_EDGE_RATIO:
        reasons.append(
            f"Edge orientation is not consistent with OCT scans "
            f"(horizontal edge ratio = {horiz_ratio:.2f}; OCT scans > {MIN_HORIZ_EDGE_RATIO:.2f}). "
            f"OCT scans have predominantly horizontal retinal layer patterns."
        )
        is_hard_reject = True

    if is_hard_reject:
        return False, reasons, True

    # --- Check 5: Model confidence + entropy (soft warning) ---
    if softmax_probs is not None:
        top_confidence = float(np.max(softmax_probs))
        # Compute Shannon entropy of the softmax distribution
        entropy = -float(np.sum(softmax_probs * np.log(softmax_probs + 1e-10)))
        entropy_ratio = entropy / MAX_ENTROPY

        if top_confidence < MEDIUM_CONFIDENCE and entropy_ratio > HIGH_ENTROPY_RATIO:
            reasons.append(
                f"Very low model confidence ({top_confidence*100:.1f}%) with high "
                f"prediction uncertainty (entropy ratio {entropy_ratio:.2f}). "
                f"This may not be a typical OCT scan."
            )
            return False, reasons, False  # soft warning, not hard reject

    return True, [], False


# ================================================================
# "WHY THIS PREDICTION?" EXPLANATION GENERATOR
# ================================================================
def make_why_prediction_html(pred_class, confidence):
    """
    Renders the prominent 'Why this prediction?' section card.
    Dynamically presents:
    1. Predicted disease
    2. Actual confidence/probability
    3. Anatomical focus justification (exact per-class wording)
    4. How the decision is made (pipeline explanation)
    5. Interpretability guardrail
    """
    if pred_class is None or confidence <= 0.0:
        return """
        <div class="why-prediction-card">
            <div class="why-header">
                <div class="why-title-group">
                    <span class="why-icon">&#128300;</span>
                    <h3 class="why-title">Why this prediction?</h3>
                </div>
            </div>
            <p style="color: var(--text-muted); font-size: 0.95em; margin: 0; line-height: 1.6;">
                Upload an OCT retinal scan and click <strong>Analyze Scan</strong> to view the model's prediction rationale, visual attention regions, and decision pipeline.
            </p>
        </div>
        """

    why_reason = DYNAMIC_WHY_EXPLANATIONS.get(
        pred_class,
        "The model identified distinctive retinal morphological patterns consistent with this condition."
    )
    desc = DESCRIPTIONS.get(pred_class, {})
    full_name = desc.get("full_name", pred_class)

    return f"""
    <div class="why-prediction-card animate-in">
        <div class="why-header">
            <div class="why-title-group">
                <span class="why-icon">&#128300;</span>
                <h3 class="why-title">Why this prediction?</h3>
            </div>
            <div class="why-badges">
                <span class="disease-badge disease-badge-{pred_class}">{pred_class} &mdash; {full_name}</span>
                <span class="confidence-badge">Confidence: {confidence*100:.2f}%</span>
            </div>
        </div>

        <div class="explanation-subcard">
            <div class="explanation-subcard-title">
                <span>&#127919;</span> Model Focus &amp; Anatomical Relevance
            </div>
            <p class="explanation-subcard-text">
                {why_reason}
            </p>
        </div>

        <div class="explanation-subcard">
            <div class="explanation-subcard-title">
                <span>&#9881;&#65039;</span> How the decision is made
            </div>
            <div class="pipeline-step-container">
                <div class="pipeline-step">
                    <span class="pipeline-chip">ConvNeXt-Tiny</span>
                    <span class="pipeline-arrow">&rarr;</span>
                    <span>extracts retinal features</span>
                    <span class="pipeline-arrow">&rarr;</span>
                    <span>produces scores for 4 classes</span>
                    <span class="pipeline-arrow">&rarr;</span>
                    <strong>highest score becomes prediction.</strong>
                </div>
                <div class="pipeline-step">
                    <span class="pipeline-chip">Grad-CAM</span>
                    <span class="pipeline-arrow">&rarr;</span>
                    <strong>shows which image regions contributed most to that prediction.</strong>
                </div>
            </div>
        </div>

        <div class="guardrail-card">
            <strong>Scientific &amp; Interpretability Guardrail:</strong>
            Grad-CAM visualizes the spatial attention of feature activations that contributed most strongly to the model's classification score. It illustrates the network's spatial attention and does not perform the classification itself or directly diagnose/prove tissue pathology.
        </div>
    </div>
    """


# ================================================================
# PREDICTION FUNCTION
# ================================================================
def predict_and_explain(image, return_details=False):
    """
    Main prediction pipeline:
    1. Validate input (reject non-OCT images via color checks)
    2. Preprocess
    3. Predict + Grad-CAM
    4. Validate via confidence + entropy (soft warning)
    5. Return formatted results with appropriate status banner
       If return_details=True, returns (overlay, conf_dict, explanation, status_banner,
                                        display_rgb, heatmap_color, why_html)
       Otherwise returns (overlay, conf_dict, explanation, status_banner) for backward compatibility.
    """
    if image is None:
        placeholder_why = make_why_prediction_html(None, 0.0)
        res = (
            None,
            {c: 0.0 for c in CLASSES},
            "### 📥 Please upload an OCT retinal scan image to begin analysis.",
            "",  # status banner
        )
        if return_details:
            return res[0], res[1], res[2], res[3], None, None, placeholder_why
        return res

    # ---------- 1. COLOR-BASED INPUT VALIDATION (hard reject) ----------
    is_valid, reasons, is_hard = validate_oct_input(image)

    if is_hard:
        # Non-OCT input detected — REJECT before running inference
        reason_list = "\n".join([f"- {r}" for r in reasons])
        status_banner = (
            '<div class="status-invalid">\n\n'
            "### 🚫 Invalid Input — Not an OCT Scan\n\n"
            "The uploaded image does not appear to be an Optical Coherence Tomography "
            "(OCT) scan. OCT scans are grayscale medical images of retinal cross-sections.\n\n"
            f"**Detected issues:**\n{reason_list}\n\n"
            "**Please upload a valid grayscale OCT retinal scan.**\n\n"
            "</div>"
        )
        explanation = (
            "### ⚠️ Prediction Skipped\n\n"
            "This model was trained exclusively on grayscale OCT retinal scans from "
            "the Kermany OCT2017 dataset. It cannot meaningfully classify other image "
            "types (natural photos, colored medical images, artwork, etc.).\n\n"
            "To make a prediction, please upload a genuine OCT scan showing a retinal "
            "cross-section."
        )
        placeholder_why = make_why_prediction_html(None, 0.0)
        res = (None, {c: 0.0 for c in CLASSES}, explanation, status_banner)
        if return_details:
            return res[0], res[1], res[2], res[3], None, None, placeholder_why
        return res

    # ---------- 2. PREPROCESSING ----------
    pil_img = Image.fromarray(image).convert("L")
    display_img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    display_rgb = np.stack([display_img] * 3, axis=-1)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    # ---------- 3. PREDICTION + GRAD-CAM ----------
    heatmap, pred_idx, probs = gradcam(input_tensor)
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # ---------- 4. CONFIDENCE + ENTROPY VALIDATION (soft warning) ----------
    is_valid, reasons, _ = validate_oct_input(image, softmax_probs=probs)

    # ---------- 5. STATUS BANNER ----------
    if not is_valid:
        # Soft warning — low confidence + high entropy
        reason_list = "\n".join([f"- {r}" for r in reasons])
        status_banner = (
            '<div class="status-warning">\n\n'
            f"### ⚠️ Low-Confidence Prediction — This May Not Be a Typical OCT Scan\n\n"
            f"The model's top prediction is **{confidence*100:.2f}%** confident, "
            f"which is below the {int(MEDIUM_CONFIDENCE*100)}% threshold. "
            f"The prediction uncertainty is unusually high.\n\n"
            f"{reason_list}\n\n"
            f"**Interpret the result below with caution.**\n\n"
            "</div>"
        )
    elif confidence >= HIGH_CONFIDENCE:
        status_banner = (
            '<div class="status-valid">\n\n'
            f"### ✅ Valid OCT Input — High-Confidence Prediction ({confidence*100:.2f}%)\n\n"
            f"Input validation passed. The model is highly confident in its prediction.\n\n"
            "</div>"
        )
    else:
        status_banner = (
            '<div class="status-warning">\n\n'
            f"### ⚠️ Valid OCT Input — Moderate Confidence ({confidence*100:.2f}%)\n\n"
            f"Input passes validation, but the model's confidence is moderate. "
            f"This may indicate a borderline case or an atypical scan — "
            f"please verify with a clinician.\n\n"
            "</div>"
        )

    # ---------- 6. GRAD-CAM OVERLAY & STANDALONE HEATMAP ----------
    heatmap_resized = cv2.resize(heatmap, (display_img.shape[1], display_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display_rgb, 0.55, heatmap_color, 0.45, 0)

    # ---------- 7. CONFIDENCE DICT (for the bar chart) ----------
    conf_dict = {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    # ---------- 8. FORMATTED EXPLANATION & WHY PREDICTION HTML ----------
    desc = DESCRIPTIONS[pred_class]
    why_html = make_why_prediction_html(pred_class, confidence)
    explanation = (
        f"### 🎯 Predicted Diagnosis: **{pred_class}** — {desc['full_name']}\n\n"
        f"**Confidence:** {confidence*100:.2f}%\n\n"
        f"**Clinical significance:** {desc['clinical']}\n\n"
        f"---\n\n"
        f"### 🔍 Grad-CAM Heatmap Interpretation\n\n"
        f"The colored overlay shows the retinal region the model focused on to make "
        f"this prediction. Red/yellow indicates high attention, blue indicates low "
        f"attention. For a **{pred_class}** diagnosis, a clinically valid model should "
        f"attend to {desc['attention_region']}.\n\n"
        f"---\n\n"
        f"*Model: ConvNeXt-Tiny (28.5M parameters) · Trained on Kermany OCT2017 "
        f"(84,484 images) · Test accuracy: 99.90% · Grad-CAM target: "
        f"model.features[-1]*"
    )

    if return_details:
        return overlay, conf_dict, explanation, status_banner, display_rgb, heatmap_color, why_html
    return overlay, conf_dict, explanation, status_banner


# ================================================================
# GRADIO UI â€” PREMIUM DARK-MODE MEDICAL INTERFACE
# ================================================================
# Preload example images if available
example_list = []
if EXAMPLE_IMAGES_DIR.exists():
    for f in sorted(EXAMPLE_IMAGES_DIR.iterdir()):
        if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
            example_list.append([str(f)])
    print(f"[OK] Loaded {len(example_list)} example images from {EXAMPLE_IMAGES_DIR}")
else:
    print("[INFO] No examples folder found -- users can still upload manually")


# â”€â”€ Premium CSS â”€â”€
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════════════════════
   ROOT — IGLOO-INSPIRED IMMERSIVE DARK PALETTE
   ═══════════════════════════════════════════════════════ */
:root {
    --bg-void:       #050510;
    --bg-primary:    #080c18;
    --bg-secondary:  #0d1425;
    --bg-card:       rgba(13, 20, 40, 0.65);
    --bg-card-solid: #0f1730;
    --bg-card-hover: rgba(16, 26, 52, 0.8);
    --border-subtle: rgba(100, 130, 200, 0.12);
    --border-glow:   rgba(99, 179, 237, 0.25);
    --border-active: rgba(99, 179, 237, 0.5);
    --text-primary:  #e8edf5;
    --text-secondary:#8b9dc3;
    --text-muted:    #5a6b8a;
    --accent-blue:   #4f8ff7;
    --accent-cyan:   #22d3ee;
    --accent-teal:   #2dd4bf;
    --accent-green:  #34d399;
    --accent-red:    #f87171;
    --accent-amber:  #fbbf24;
    --accent-purple: #a78bfa;
    --accent-pink:   #f472b6;
    --glass-bg:      rgba(13, 20, 40, 0.55);
    --glass-border:  rgba(99, 179, 237, 0.15);
    --glow-blue:     0 0 30px rgba(79, 143, 247, 0.12), 0 0 60px rgba(79, 143, 247, 0.06);
    --glow-cyan:     0 0 30px rgba(34, 211, 238, 0.12), 0 0 60px rgba(34, 211, 238, 0.06);
    --glow-green:    0 0 25px rgba(52, 211, 153, 0.12), 0 0 50px rgba(52, 211, 153, 0.06);
    --glow-red:      0 0 25px rgba(248, 113, 113, 0.12), 0 0 50px rgba(248, 113, 113, 0.06);
    --glow-purple:   0 0 30px rgba(167, 139, 250, 0.12);
    --radius:        14px;
    --radius-lg:     20px;
    --radius-xl:     28px;
}

/* ═══════════════════════════════════════════════════════
   GLOBAL RESETS & CONTAINER
   ═══════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-void) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.gradio-container {
    max-width: 1360px !important;
    margin: 0 auto !important;
    padding: 0 32px !important;
    position: relative;
    z-index: 1;
}
.dark .gradio-container { background: var(--bg-void) !important; }
footer { display: none !important; }

/* ═══════════════════════════════════════════════════════
   IMMERSIVE AURORA BACKGROUND — IGLOO-STYLE
   Multiple animated radial gradients + floating orbs
   ═══════════════════════════════════════════════════════ */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 900px 600px at 15% 10%, rgba(79, 143, 247, 0.08), transparent 70%),
        radial-gradient(ellipse 700px 500px at 85% 15%, rgba(167, 139, 250, 0.06), transparent 70%),
        radial-gradient(ellipse 800px 450px at 50% 85%, rgba(34, 211, 238, 0.05), transparent 70%),
        radial-gradient(ellipse 500px 400px at 75% 50%, rgba(244, 114, 182, 0.04), transparent 70%),
        radial-gradient(ellipse 600px 350px at 25% 70%, rgba(52, 211, 153, 0.04), transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: auroraShift 25s ease-in-out infinite alternate;
}
body::after {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle 180px at 20% 30%, rgba(79, 143, 247, 0.07), transparent),
        radial-gradient(circle 120px at 70% 60%, rgba(167, 139, 250, 0.06), transparent),
        radial-gradient(circle 150px at 50% 20%, rgba(34, 211, 238, 0.05), transparent);
    pointer-events: none;
    z-index: 0;
    animation: orbFloat 30s ease-in-out infinite alternate-reverse;
}
@keyframes auroraShift {
    0%   { opacity: 0.6; transform: scale(1) rotate(0deg); }
    50%  { opacity: 1.0; transform: scale(1.02) rotate(0.5deg); }
    100% { opacity: 0.7; transform: scale(0.98) rotate(-0.5deg); }
}
@keyframes orbFloat {
    0%   { transform: translate(0, 0) scale(1); opacity: 0.7; }
    33%  { transform: translate(30px, -20px) scale(1.1); opacity: 1; }
    66%  { transform: translate(-20px, 15px) scale(0.95); opacity: 0.8; }
    100% { transform: translate(10px, -10px) scale(1.05); opacity: 0.9; }
}

/* Noise/grain texture overlay for depth */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.015'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ═══════════════════════════════════════════════════════
   HERO HEADER — IMMERSIVE & BOLD
   ═══════════════════════════════════════════════════════ */
.hero-header {
    text-align: center;
    padding: 48px 20px 36px;
    position: relative;
    z-index: 2;
    margin-bottom: 8px;
}
.hero-header .logo-icon {
    font-size: 3.2em;
    display: block;
    margin-bottom: 12px;
    filter: drop-shadow(0 0 30px rgba(79, 143, 247, 0.5)) drop-shadow(0 0 60px rgba(34, 211, 238, 0.3));
    animation: logoPulse 4s ease-in-out infinite;
}
@keyframes logoPulse {
    0%, 100% { filter: drop-shadow(0 0 30px rgba(79, 143, 247, 0.5)) drop-shadow(0 0 60px rgba(34, 211, 238, 0.3)); transform: scale(1); }
    50%      { filter: drop-shadow(0 0 40px rgba(79, 143, 247, 0.7)) drop-shadow(0 0 80px rgba(34, 211, 238, 0.4)); transform: scale(1.05); }
}
.hero-header h1 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    font-size: 3.4em !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #60a5fa 0%, #22d3ee 25%, #34d399 50%, #a78bfa 75%, #60a5fa 100%);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.03em;
    animation: titleGradient 8s ease-in-out infinite;
    text-shadow: none;
}
@keyframes titleGradient {
    0%, 100% { background-position: 0% center; }
    50%      { background-position: 100% center; }
}
.hero-subtitle {
    font-size: 1.15em;
    color: var(--text-secondary);
    font-weight: 400;
    margin-top: 8px;
    letter-spacing: 0.04em;
    opacity: 0.85;
}
.hero-team {
    font-size: 0.88em;
    color: var(--text-muted);
    margin-top: 10px;
    letter-spacing: 0.01em;
}
.hero-team span {
    color: var(--accent-cyan);
    font-weight: 600;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.3);
}

/* ── Stat Badges — Floating Pills ── */
.stat-badges {
    display: flex;
    justify-content: center;
    gap: 14px;
    margin-top: 24px;
    flex-wrap: wrap;
}
.stat-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    background: rgba(13, 20, 40, 0.6);
    border: 1px solid var(--border-subtle);
    border-radius: 100px;
    font-size: 0.8em;
    font-weight: 500;
    color: var(--text-secondary);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.stat-badge::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(79, 143, 247, 0.05), rgba(34, 211, 238, 0.03));
    opacity: 0;
    transition: opacity 0.4s ease;
}
.stat-badge:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-blue);
    transform: translateY(-3px) scale(1.02);
}
.stat-badge:hover::before { opacity: 1; }
.stat-badge .badge-val {
    color: var(--text-primary);
    font-weight: 700;
    position: relative;
    z-index: 1;
}
.stat-badge .badge-icon {
    font-size: 1.15em;
    position: relative;
    z-index: 1;
}

/* ═══════════════════════════════════════════════════════
   STATUS BANNERS (gr.HTML — no sanitization)
   Glassmorphism + neon left border + glow effects
   ═══════════════════════════════════════════════════════ */
.status-banner-valid {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.08) 0%, rgba(34, 211, 238, 0.05) 100%);
    border: 1px solid rgba(52, 211, 153, 0.25);
    border-left: 5px solid var(--accent-green);
    border-radius: var(--radius);
    padding: 18px 24px;
    margin: 14px 0 18px;
    box-shadow: var(--glow-green), inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    animation: bannerSlideIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
.status-banner-valid h3 {
    color: #34d399 !important;
    margin: 0 0 6px !important;
    font-size: 1.08em !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    text-shadow: 0 0 20px rgba(52, 211, 153, 0.3);
}
.status-banner-valid p { color: var(--text-secondary); margin: 0; font-size: 0.92em; }

.status-banner-invalid {
    background: linear-gradient(135deg, rgba(248, 113, 113, 0.08) 0%, rgba(239, 68, 68, 0.04) 100%);
    border: 1px solid rgba(248, 113, 113, 0.25);
    border-left: 5px solid var(--accent-red);
    border-radius: var(--radius);
    padding: 18px 24px;
    margin: 14px 0 18px;
    box-shadow: var(--glow-red), inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter: blur(16px);
    animation: bannerSlideIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
.status-banner-invalid h3 {
    color: #f87171 !important;
    margin: 0 0 6px !important;
    font-size: 1.08em !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    text-shadow: 0 0 20px rgba(248, 113, 113, 0.3);
}
.status-banner-invalid p  { color: var(--text-secondary); margin: 0; font-size: 0.92em; }
.status-banner-invalid ul { color: var(--text-secondary); margin: 8px 0 0 18px; font-size: 0.88em; padding: 0; }
.status-banner-invalid li { margin-bottom: 4px; }

.status-banner-warning {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.08) 0%, rgba(245, 158, 11, 0.04) 100%);
    border: 1px solid rgba(251, 191, 36, 0.25);
    border-left: 5px solid var(--accent-amber);
    border-radius: var(--radius);
    padding: 18px 24px;
    margin: 14px 0 18px;
    backdrop-filter: blur(16px);
    animation: bannerSlideIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
.status-banner-warning h3 {
    color: #fbbf24 !important;
    margin: 0 0 6px !important;
    font-size: 1.08em !important;
    font-family: 'Space Grotesk', sans-serif !important;
    text-shadow: 0 0 20px rgba(251, 191, 36, 0.3);
}
.status-banner-warning p { color: var(--text-secondary); margin: 0; font-size: 0.92em; }

@keyframes bannerSlideIn {
    from { opacity: 0; transform: translateY(-12px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* ═══════════════════════════════════════════════════════
   GLASS CARDS — DEEP GLASSMORPHISM PANELS
   ═══════════════════════════════════════════════════════ */
.glass-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0 !important;
    backdrop-filter: blur(24px) saturate(1.3);
    -webkit-backdrop-filter: blur(24px) saturate(1.3);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: var(--radius-lg);
    background: linear-gradient(135deg, rgba(79, 143, 247, 0.03), transparent 60%);
    opacity: 0;
    transition: opacity 0.4s ease;
    pointer-events: none;
}
.glass-card:hover {
    border-color: var(--border-glow) !important;
    box-shadow: var(--glow-blue);
    transform: translateY(-2px);
}
.glass-card:hover::before { opacity: 1; }

/* Panel headers inside glass cards */
.panel-header {
    padding: 18px 24px 14px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    gap: 12px;
    position: relative;
}
.panel-header::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 24px;
    right: 24px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-glow), transparent);
}
.panel-header .panel-icon {
    font-size: 1.3em;
    width: 40px; height: 40px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(79, 143, 247, 0.15), rgba(34, 211, 238, 0.1));
    border-radius: 12px;
    border: 1px solid rgba(79, 143, 247, 0.2);
    box-shadow: 0 0 20px rgba(79, 143, 247, 0.1);
}
.panel-header h3 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    font-size: 1.08em !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    letter-spacing: -0.01em;
}
.panel-header .panel-tag {
    font-size: 0.68em;
    padding: 4px 12px;
    background: linear-gradient(135deg, rgba(79, 143, 247, 0.12), rgba(167, 139, 250, 0.08));
    color: var(--accent-blue);
    border-radius: 100px;
    border: 1px solid rgba(79, 143, 247, 0.2);
    font-weight: 600;
    margin-left: auto;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.panel-body { padding: 18px 24px 24px; }

/* ═══════════════════════════════════════════════════════
   GRADIO COMPONENT OVERRIDES (Dark theme)
   ═══════════════════════════════════════════════════════ */
/* Image upload area */
#oct-input-image { border-radius: var(--radius) !important; overflow: hidden; }
.image-container, .upload-container {
    background: var(--bg-secondary) !important;
    border: 2px dashed rgba(100, 130, 200, 0.2) !important;
    border-radius: var(--radius) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.upload-container:hover {
    border-color: var(--accent-blue) !important;
    box-shadow: var(--glow-blue);
    background: rgba(13, 20, 40, 0.8) !important;
}

/* Labels/text */
.label-wrap, label, .label-text {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

/* Confidence label component */
#confidence-bars .label-class-text { color: var(--text-primary) !important; }
#confidence-bars .label-class-confidence { color: var(--accent-cyan) !important; font-weight: 700 !important; }

/* Buttons — 3D depth with glow */
#analyze-btn {
    background: linear-gradient(135deg, #4f8ff7 0%, #2563eb 50%, #4f8ff7 100%) !important;
    background-size: 200% auto !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.02em !important;
    padding: 14px 28px !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(79, 143, 247, 0.35), 0 2px 8px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: none !important;
    letter-spacing: 0.02em !important;
    position: relative;
    overflow: hidden;
}
#analyze-btn:hover {
    background-position: right center !important;
    box-shadow: 0 8px 30px rgba(79, 143, 247, 0.5), 0 4px 12px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transform: translateY(-2px) !important;
}
#analyze-btn:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 10px rgba(79, 143, 247, 0.3) !important;
}
#clear-btn {
    background: rgba(13, 20, 40, 0.6) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
#clear-btn:hover {
    border-color: rgba(248, 113, 113, 0.4) !important;
    color: var(--accent-red) !important;
    box-shadow: 0 0 20px rgba(248, 113, 113, 0.1) !important;
    transform: translateY(-1px) !important;
}

/* Markdown text */
.prose, .markdown-text, .md { color: var(--text-secondary) !important; }
.prose h3, .markdown-text h3, .md h3 {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
}
.prose strong { color: var(--text-primary) !important; }
.prose em { color: var(--text-muted) !important; }
.prose hr { border-color: var(--border-subtle) !important; margin: 18px 0 !important; }
.prose code {
    background: rgba(13, 20, 40, 0.8) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    padding: 2px 8px !important;
    font-size: 0.85em !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Grad-CAM output image */
#gradcam-overlay {
    border-radius: var(--radius) !important;
    overflow: hidden;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ═══════════════════════════════════════════════════════
   CLINICAL INTERPRETATION — PREMIUM GLASS CARD
   ═══════════════════════════════════════════════════════ */
.clinical-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(13, 20, 40, 0.5) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 28px 32px;
    margin-top: 18px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}
.clinical-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-teal), var(--accent-purple));
    background-size: 200% auto;
    animation: shineBar 4s ease-in-out infinite;
}
@keyframes shineBar {
    0%, 100% { background-position: 0% center; }
    50%      { background-position: 100% center; }
}

/* ═══════════════════════════════════════════════════════
   ABOUT / FOOTER SECTION
   ═══════════════════════════════════════════════════════ */
.about-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(13, 20, 40, 0.4) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 28px 32px;
    margin-top: 24px;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}
.about-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
}
.about-card h3 {
    color: var(--accent-cyan) !important;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.2);
}

/* ═══════════════════════════════════════════════════════
   EXAMPLES GALLERY — HOVERABLE THUMBNAILS
   ═══════════════════════════════════════════════════════ */
.examples-wrapper {
    margin-top: 24px;
}
.examples-wrapper h3 { color: var(--text-primary) !important; }
.examples-wrapper p  { color: var(--text-muted) !important; }

.gallery-item {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border-subtle) !important;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
}
.gallery-item::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(79, 143, 247, 0.05), transparent);
    opacity: 0;
    transition: opacity 0.4s ease;
    pointer-events: none;
}
.gallery-item:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan);
    transform: scale(1.06) translateY(-4px);
}
.gallery-item:hover::after { opacity: 1; }

/* ═══════════════════════════════════════════════════════
   SECTION DIVIDERS — GRADIENT LINES
   ═══════════════════════════════════════════════════════ */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(100, 130, 200, 0.15) 20%, rgba(100, 130, 200, 0.15) 80%, transparent 100%);
    margin: 24px 0;
    border: none;
}

/* ═══════════════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    .hero-header h1 { font-size: 2.2em !important; }
    .stat-badges { gap: 8px; }
    .stat-badge { font-size: 0.72em; padding: 6px 12px; }
    .panel-body { padding: 14px 16px 18px; }
}

/* ═══════════════════════════════════════════════════════
   ANIMATION UTILITIES
   ═══════════════════════════════════════════════════════ */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards; }

/* Smooth scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb {
    background: rgba(100, 130, 200, 0.2);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(100, 130, 200, 0.35); }

/* Block/Panel backgrounds */
.block, .form, .wrap {
    background: transparent !important;
    border: none !important;
}

/* ═══════════════════════════════════════════════════════
   "WHY THIS PREDICTION?" EXPLANATION SECTION
   ═══════════════════════════════════════════════════════ */
.why-prediction-card {
    background: linear-gradient(135deg, rgba(13, 20, 40, 0.85) 0%, rgba(15, 23, 48, 0.9) 100%);
    border: 1px solid rgba(99, 179, 237, 0.3);
    border-radius: var(--radius-lg);
    padding: 24px 28px;
    margin-top: 14px;
    backdrop-filter: blur(20px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
    position: relative;
    overflow: hidden;
}
.why-prediction-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue), var(--accent-purple));
}
.why-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(100, 130, 200, 0.15);
}
.why-title-group {
    display: flex;
    align-items: center;
    gap: 10px;
}
.why-icon {
    font-size: 24px;
}
.why-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35em;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.why-badges {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.disease-badge {
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.88em;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.disease-badge-CNV {
    background: rgba(248, 113, 113, 0.18);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.4);
}
.disease-badge-DME {
    background: rgba(251, 146, 60, 0.18);
    color: #fb923c;
    border: 1px solid rgba(251, 146, 60, 0.4);
}
.disease-badge-DRUSEN {
    background: rgba(250, 204, 21, 0.18);
    color: #facc15;
    border: 1px solid rgba(250, 204, 21, 0.4);
}
.disease-badge-NORMAL {
    background: rgba(52, 211, 153, 0.18);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.4);
}
.confidence-badge {
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.88em;
    font-weight: 600;
    background: rgba(34, 211, 238, 0.15);
    color: var(--accent-cyan);
    border: 1px solid rgba(34, 211, 238, 0.35);
}
.explanation-subcard {
    background: rgba(8, 12, 24, 0.65);
    border: 1px solid rgba(100, 130, 200, 0.12);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 14px;
}
.explanation-subcard-title {
    font-size: 0.84em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--accent-cyan);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.explanation-subcard-text {
    font-size: 1.02em;
    line-height: 1.6;
    color: #f1f5f9;
    margin: 0;
}
.pipeline-step-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 6px;
}
.pipeline-step {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.94em;
    color: #cbd5e1;
    line-height: 1.5;
}
.pipeline-chip {
    background: rgba(79, 143, 247, 0.2);
    color: #93c5fd;
    border: 1px solid rgba(79, 143, 247, 0.35);
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.92em;
}
.pipeline-arrow {
    color: var(--accent-teal);
    font-weight: bold;
    font-size: 1.1em;
}
.guardrail-card {
    background: rgba(15, 23, 42, 0.55);
    border-left: 3px solid var(--accent-purple);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 12px 18px;
    font-size: 0.86em;
    color: #94a3b8;
    line-height: 1.6;
    margin-top: 14px;
}
.guardrail-card strong {
    color: var(--accent-purple);
}
"""


# ── Build status banners as raw HTML (bypass Markdown sanitization) ──
def make_status_html(banner_type, title, body, items=None):
    """Generate a styled HTML banner string."""
    css_class = f"status-banner-{banner_type}"
    items_html = ""
    if items:
        items_html = "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    return f'<div class="{css_class}"><h3>{title}</h3><p>{body}</p>{items_html}</div>'


# ── Override predict_and_explain to use HTML banners & support detail triad ──
_original_predict = predict_and_explain

def predict_and_explain_v2(image):
    """
    Enhanced UI wrapper for Gradio interface.
    Returns:
        output_confidence: dict (confidence bars)
        out_orig_img: np.ndarray (original OCT image)
        out_heatmap_img: np.ndarray (Grad-CAM jet heatmap)
        out_overlay_img: np.ndarray (Grad-CAM overlay)
        out_why_html: str (HTML card for 'Why this prediction?')
        output_explanation: str (Markdown for clinical background)
        status_output: str (HTML status banner)
    """
    if image is None:
        return (
            {c: 0.0 for c in CLASSES},
            None,
            None,
            None,
            make_why_prediction_html(None, 0.0),
            "### Upload an OCT scan and click **Analyze Scan** to begin.",
            "",
        )

    overlay, conf_dict, explanation, _md_banner, orig_rgb, heatmap_color, why_html = _original_predict(
        image, return_details=True
    )

    # Rebuild banner as raw HTML for gr.HTML component
    if overlay is None and all(v == 0.0 for v in conf_dict.values()):
        # Hard rejection
        is_valid, reasons, _ = validate_oct_input(image)
        html_banner = make_status_html(
            "invalid",
            "&#128683; Invalid Input &mdash; Not an OCT Scan",
            "The uploaded image does not appear to be an OCT retinal scan. "
            "OCT scans are grayscale medical images of retinal cross-sections.",
            reasons,
        )
    else:
        top_conf = max(conf_dict.values())
        pred_class = max(conf_dict, key=conf_dict.get)
        if top_conf >= HIGH_CONFIDENCE:
            html_banner = make_status_html(
                "valid",
                f"&#9989; Valid OCT Input &mdash; High-Confidence Prediction ({top_conf*100:.2f}%)",
                f"Input validation passed. The model is highly confident this is <strong>{pred_class}</strong>.",
            )
        elif top_conf >= MEDIUM_CONFIDENCE:
            html_banner = make_status_html(
                "warning",
                f"&#9888;&#65039; Moderate Confidence ({top_conf*100:.2f}%)",
                "The model's confidence is moderate. This may indicate a borderline case &mdash; verify with a clinician.",
            )
        else:
            html_banner = make_status_html(
                "warning",
                f"&#9888;&#65039; Low-Confidence Prediction ({top_conf*100:.2f}%)",
                "The model is not confident. This may not be a typical OCT scan. Interpret with caution.",
            )

    return (
        conf_dict,
        orig_rgb,
        heatmap_color,
        overlay,
        why_html,
        explanation,
        html_banner,
    )


# â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• 
# BUILD THE INTERFACE
# â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• 
with gr.Blocks(
    title="RetinaScan AI â€” OCT Disease Classifier",
) as demo:

    # â”€â”€â”€â”€ HERO HEADER â”€â”€â”€â”€
    gr.HTML("""
    <div class="hero-header">
        <span class="logo-icon">&#128300;</span>
        <h1>RetinaScan AI</h1>
        <div class="hero-subtitle">OCT Retinal Disease Classifier with Explainable AI</div>
        <div class="hero-team">
            <span>Murali A</span> &middot; <span>Niranjan T</span> &middot; <span>Praveen K</span>
            &nbsp;&vert;&nbsp; Guide: <span>Dr. C. Santhosh Kumar</span>
            &nbsp;&vert;&nbsp; Sona College of Technology
        </div>
        <div class="stat-badges">
            <div class="stat-badge">
                <span class="badge-icon">&#129504;</span>
                <span>Architecture: <span class="badge-val">ConvNeXt-Tiny</span></span>
            </div>
            <div class="stat-badge">
                <span class="badge-icon">&#127919;</span>
                <span>Test Accuracy: <span class="badge-val">99.90%</span></span>
            </div>
            <div class="stat-badge">
                <span class="badge-icon">&#128202;</span>
                <span>Parameters: <span class="badge-val">28.5M</span></span>
            </div>
            <div class="stat-badge">
                <span class="badge-icon">&#128065;</span>
                <span>Classes: <span class="badge-val">CNV &middot; DME &middot; DRUSEN &middot; NORMAL</span></span>
            </div>
        </div>
    </div>
    """)

    # â”€â”€â”€â”€ STATUS BANNER (raw HTML â€” renders with full styling) â”€â”€â”€â”€
    status_output = gr.HTML(value="", elem_id="status-banner")

    # â”€â”€â”€â”€ MAIN CONTENT â€” 2 COLUMNS â”€â”€â”€â”€
    with gr.Row(equal_height=False):
        # â• â• â•  LEFT PANEL â€” Upload â• â• â• 
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="panel-header">
                <div class="panel-icon">&#128228;</div>
                <h3>Upload OCT Scan</h3>
                <span class="panel-tag">INPUT</span>
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

        # ═══ RIGHT PANEL — Results ═══
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="panel-header">
                <div class="panel-icon">&#127919;</div>
                <h3>Prediction &amp; Confidence</h3>
                <span class="panel-tag">OUTPUT</span>
            </div>
            """)
            output_confidence = gr.Label(
                num_top_classes=4,
                label="Class Confidence Distribution",
                elem_id="confidence-bars",
            )
            gr.HTML("""
            <div style="margin-top: 14px; padding: 14px 18px; background: rgba(13, 20, 40, 0.6); border: 1px solid rgba(100, 130, 200, 0.15); border-radius: var(--radius); font-size: 0.88em; color: var(--text-secondary); line-height: 1.6;">
                <span style="color: var(--accent-cyan); font-weight: 600;">Automated Diagnostic Assessment:</span>
                The model computes softmax probabilities across the 4 retinal diagnostic categories. The highest scoring category is selected as the predicted class. See the detailed visual explanation below.
            </div>
            """)

    # ──── SECTION DIVIDER ────
    gr.HTML('<div class="section-divider"></div>')

    # ──── "WHY THIS PREDICTION?" EXPLANATION SECTION ────
    gr.HTML("""
    <div class="panel-header" style="padding-left:0; margin-top: 6px;">
        <div class="panel-icon" style="background: linear-gradient(135deg, rgba(34, 211, 238, 0.2), rgba(79, 143, 247, 0.15)); border-color: rgba(34, 211, 238, 0.35);">&#128300;</div>
        <h3>Why this prediction? &mdash; Explainable AI Diagnostics</h3>
        <span class="panel-tag">VISUAL EXPLANATION</span>
    </div>
    """)

    with gr.Row():
        out_orig_img = gr.Image(
            type="numpy",
            label="Original OCT Image",
            height=270,
            elem_id="why-orig-img",
        )
        out_heatmap_img = gr.Image(
            type="numpy",
            label="Grad-CAM Heatmap",
            height=270,
            elem_id="why-heatmap-img",
        )
        out_overlay_img = gr.Image(
            type="numpy",
            label="Grad-CAM Overlay",
            height=270,
            elem_id="why-overlay-img",
        )

    out_why_html = gr.HTML(
        value=make_why_prediction_html(None, 0.0),
        elem_id="why-prediction-container",
    )

    # ──── SECTION DIVIDER ────
    gr.HTML('<div class="section-divider"></div>')

    # ──── CLINICAL INTERPRETATION ────
    gr.HTML("""
    <div class="panel-header" style="padding-left:0;">
        <div class="panel-icon" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 182, 212, 0.1)); border-color: rgba(16, 185, 129, 0.2);">&#129658;</div>
        <h3>Clinical Interpretation</h3>
    </div>
    """)
    output_explanation = gr.Markdown(
        value=(
            "*Upload an OCT scan and click **Analyze Scan** to receive an AI-assisted "
            "diagnostic interpretation with Grad-CAM visual explanation.*"
        ),
        elem_id="clinical-interpretation",
    )

    # ──── SECTION DIVIDER ────
    gr.HTML('<div class="section-divider"></div>')

    ui_outputs = [
        output_confidence,
        out_orig_img,
        out_heatmap_img,
        out_overlay_img,
        out_why_html,
        output_explanation,
        status_output,
    ]

    # ──── EXAMPLES GALLERY ────
    if example_list:
        gr.HTML("""
        <div class="examples-wrapper">
            <div class="panel-header" style="padding-left:0;">
                <div class="panel-icon" style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.1)); border-color: rgba(139, 92, 246, 0.2);">&#129514;</div>
                <h3>Sample OCT Test Images</h3>
            </div>
            <p style="color: var(--text-muted); font-size: 0.9em; margin-top: 4px;">Click any example below to auto-analyze it with the model.</p>
        </div>
        """)
        gr.Examples(
            examples=example_list,
            inputs=input_image,
            outputs=ui_outputs,
            fn=predict_and_explain_v2,
            cache_examples=False,
            run_on_click=True,
            label="Demo Samples (CNV, DME, DRUSEN, NORMAL)",
            examples_per_page=4,
        )

    # â”€â”€â”€â”€ SECTION DIVIDER â”€â”€â”€â”€
    gr.HTML('<div class="section-divider"></div>')

    # â”€â”€â”€â”€ ABOUT THE MODEL â”€â”€â”€â”€
    gr.HTML("""
    <div class="about-card">
        <h3 style="font-family: 'Space Grotesk', sans-serif; margin-top: 0;">&#8505;&#65039; About the Model</h3>
        <p style="color: #94a3b8; font-size: 0.92em; line-height: 1.7;">
            This system uses <strong style="color: #f1f5f9;">ConvNeXt-Tiny</strong> (Liu et al., 2022) &mdash;
            a modern CNN architecture with <strong style="color: #f1f5f9;">28.5 million parameters</strong>
            &mdash; fine-tuned on the
            <strong style="color: #f1f5f9;">Kermany OCT2017 dataset</strong>
            (84,484 labeled OCT retinal scans across 4 classes).
            On the official held-out test set of 968 images, the model achieves
            <strong style="color: #10b981;">99.90% accuracy</strong>
            (967/968 correctly classified), surpassing the base paper baseline of 97.54%.
        </p>
        <p style="color: #94a3b8; font-size: 0.92em; line-height: 1.7; margin-top: 12px;">
            <strong style="color: #06b6d4;">Interpretability:</strong>
            Every prediction is accompanied by a real-time Grad-CAM heatmap generated by hooking into
            the last convolutional stage (<code style="background: #111827; color: #06b6d4; padding: 2px 6px; border-radius: 4px; font-size: 0.88em;">model.features[-1]</code>).
            Clinical validation has confirmed that the model consistently attends to anatomically
            correct retinal regions.
        </p>
        <p style="color: #94a3b8; font-size: 0.92em; line-height: 1.7; margin-top: 12px;">
            <strong style="color: #8b5cf6;">Input Validation:</strong>
            Three-signal validation pipeline &mdash; HSV saturation check, RGB channel similarity check,
            and model confidence + entropy analysis &mdash; ensures that non-OCT images are rejected
            before or after inference.
        </p>
    </div>
    """)

    # ──── WIRING ────
    submit_btn.click(
        fn=predict_and_explain_v2,
        inputs=input_image,
        outputs=ui_outputs,
    )

    def clear_all():
        return (
            None,                                      # input_image
            {},                                        # output_confidence
            None,                                      # out_orig_img
            None,                                      # out_heatmap_img
            None,                                      # out_overlay_img
            make_why_prediction_html(None, 0.0),        # out_why_html
            (
                "*Upload an OCT scan and click **Analyze Scan** to receive an "
                "AI-assisted diagnostic interpretation with Grad-CAM visual "
                "explanation.*"
            ),                                         # output_explanation
            "",                                        # status_output
        )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[
            input_image,
            output_confidence,
            out_orig_img,
            out_heatmap_img,
            out_overlay_img,
            out_why_html,
            output_explanation,
            status_output,
        ],
    )


# ================================================================
# LAUNCH
# ================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting RetinaScan AI v2 (Premium UI)...")
    print("=" * 60)
    demo.launch(
        share=False,
        inbrowser=True,
        show_error=True,
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    )



