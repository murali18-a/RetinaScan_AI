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
# 2. PREMIUM CSS — IGLOO-INSPIRED IMMERSIVE DARK DESIGN
# ================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-void:       #050510;
    --bg-primary:    #080c18;
    --bg-secondary:  #0d1425;
    --bg-card:       rgba(13, 20, 40, 0.65);
    --bg-card-solid: #0f1730;
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

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-void) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
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

/* Aurora Background */
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

/* Hero Header */
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
}
.hero-team {
    font-size: 0.88em;
    color: var(--text-muted);
    margin-top: 10px;
}
.hero-team span {
    color: var(--accent-cyan);
    font-weight: 600;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.3);
}

/* Stat Badges */
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
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.stat-badge:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-blue);
    transform: translateY(-3px) scale(1.02);
}
.stat-badge .badge-val {
    color: var(--text-primary);
    font-weight: 700;
    position: relative;
    z-index: 1;
}

/* Status Banners */
.status-banner-valid {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.08), rgba(34, 211, 238, 0.05));
    border: 1px solid rgba(52, 211, 153, 0.25);
    border-left: 5px solid var(--accent-green);
    border-radius: var(--radius);
    padding: 18px 24px;
    margin: 14px 0 18px;
    box-shadow: var(--glow-green);
    backdrop-filter: blur(16px);
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
    background: linear-gradient(135deg, rgba(248, 113, 113, 0.08), rgba(239, 68, 68, 0.04));
    border: 1px solid rgba(248, 113, 113, 0.25);
    border-left: 5px solid var(--accent-red);
    border-radius: var(--radius);
    padding: 18px 24px;
    margin: 14px 0 18px;
    box-shadow: var(--glow-red);
    backdrop-filter: blur(16px);
    animation: bannerSlideIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
.status-banner-invalid h3 {
    color: #f87171 !important;
    margin: 0 0 6px !important;
    font-size: 1.08em !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
}
.status-banner-invalid p  { color: var(--text-secondary); margin: 0; font-size: 0.92em; }
.status-banner-invalid ul { color: var(--text-secondary); margin: 8px 0 0 18px; font-size: 0.88em; padding: 0; }

.status-banner-warning {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.08), rgba(245, 158, 11, 0.04));
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
}
.status-banner-warning p { color: var(--text-secondary); margin: 0; font-size: 0.92em; }

@keyframes bannerSlideIn {
    from { opacity: 0; transform: translateY(-12px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* Panel Headers */
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

/* Buttons */
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
}
#analyze-btn:hover {
    background-position: right center !important;
    box-shadow: 0 8px 30px rgba(79, 143, 247, 0.5), 0 4px 12px rgba(0,0,0,0.4) !important;
    transform: translateY(-2px) !important;
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
    transform: translateY(-1px) !important;
}

/* Gradio component overrides */
.image-container, .upload-container {
    background: var(--bg-secondary) !important;
    border: 2px dashed rgba(100, 130, 200, 0.2) !important;
    border-radius: var(--radius) !important;
    transition: all 0.4s ease !important;
}
.upload-container:hover {
    border-color: var(--accent-blue) !important;
    box-shadow: var(--glow-blue);
}
.label-wrap, label, .label-text { color: var(--text-secondary) !important; font-weight: 500 !important; }
#confidence-bars .label-class-text { color: var(--text-primary) !important; }
#confidence-bars .label-class-confidence { color: var(--accent-cyan) !important; font-weight: 700 !important; }

/* Markdown */
.prose, .markdown-text, .md { color: var(--text-secondary) !important; }
.prose h3, .markdown-text h3, .md h3 {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
}
.prose strong { color: var(--text-primary) !important; }
.prose hr { border-color: var(--border-subtle) !important; }

/* Section dividers */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(100, 130, 200, 0.15) 20%, rgba(100, 130, 200, 0.15) 80%, transparent 100%);
    margin: 24px 0;
    border: none;
}

/* About card */
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
.about-card h3 { color: var(--accent-cyan) !important; text-shadow: 0 0 20px rgba(34, 211, 238, 0.2); }

/* Gallery */
.gallery-item {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border-subtle) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.gallery-item:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan);
    transform: scale(1.06) translateY(-4px);
}

/* Block backgrounds */
.block, .form, .wrap { background: transparent !important; border: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: rgba(100, 130, 200, 0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(100, 130, 200, 0.35); }

/* Responsive */
@media (max-width: 768px) {
    .hero-header h1 { font-size: 2.2em !important; }
    .stat-badges { gap: 8px; }
    .stat-badge { font-size: 0.72em; padding: 6px 12px; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards; }
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
        return ({c: 0.0 for c in CLASSES}, None, explanation, html_banner)

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
    heatmap_resized = cv2.resize(heatmap, (display_img.shape[1], display_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display_rgb, 0.55, heatmap_color, 0.45, 0)

    # Step 7: Confidence dict
    conf_dict = {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    # Step 8: Clinical explanation
    desc = DESCRIPTIONS[pred_class]
    explanation = (
        f"### 🎯 Predicted Diagnosis: **{pred_class}** — {desc['full_name']}\n\n"
        f"**Confidence:** {confidence*100:.2f}%\n\n"
        f"**Clinical significance:** {desc['clinical']}\n\n"
        f"---\n\n"
        f"### 🔍 Why this prediction?\n\n"
        f"{DYNAMIC_WHY[pred_class]}\n\n"
        f"---\n\n"
        f"### 🔥 Grad-CAM Heatmap Interpretation\n\n"
        f"The colored overlay shows the retinal region the model focused on. "
        f"Red/yellow = high attention, blue = low attention. For a **{pred_class}** diagnosis, "
        f"the model should attend to {desc['attention_region']}.\n\n"
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
# 10. GRADIO UI — PREMIUM DARK-MODE INTERFACE
# ================================================================
with gr.Blocks(title="RetinaScan AI") as demo:

    # ──── HERO HEADER ────
    gr.HTML("""
    <div class="hero-header">
        <span class="logo-icon">&#128300;</span>
        <h1>RetinaScan AI</h1>
        <div class="hero-subtitle">OCT Retinal Disease Classifier with Explainable AI</div>
        <div class="hero-team">
            <span>Murali A</span> &middot; <span>Niranjan T</span> &middot; <span>Praveen K</span>
            &nbsp;|&nbsp; Guide: <span>Dr. C. Santhosh Kumar</span>
            &nbsp;|&nbsp; Sona College of Technology
        </div>
        <div class="stat-badges">
            <div class="stat-badge">
                <span>&#129504;</span>
                <span>Architecture: <span class="badge-val">ConvNeXt-Tiny</span></span>
            </div>
            <div class="stat-badge">
                <span>&#127919;</span>
                <span>Test Accuracy: <span class="badge-val">99.90%</span></span>
            </div>
            <div class="stat-badge">
                <span>&#128202;</span>
                <span>Parameters: <span class="badge-val">28.5M</span></span>
            </div>
            <div class="stat-badge">
                <span>&#128065;</span>
                <span>Classes: <span class="badge-val">CNV &middot; DME &middot; DRUSEN &middot; NORMAL</span></span>
            </div>
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

        # RIGHT PANEL — Results
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="panel-header">
                <div class="panel-icon">&#127919;</div>
                <h3>Prediction &amp; Explanation</h3>
                <span class="panel-tag">OUTPUT</span>
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
    <div class="panel-header" style="padding-left:0;">
        <div class="panel-icon" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 182, 212, 0.1)); border-color: rgba(16, 185, 129, 0.2);">&#129658;</div>
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
        <h3 style="font-family: 'Space Grotesk', sans-serif; margin-top: 0;">&#8505;&#65039; About the Model</h3>
        <p style="color: #94a3b8; font-size: 0.92em; line-height: 1.7;">
            This system uses <strong style="color: #f1f5f9;">ConvNeXt-Tiny</strong> (Liu et al., 2022) &mdash;
            a modern CNN with <strong style="color: #f1f5f9;">28.5 million parameters</strong>
            &mdash; fine-tuned on the
            <strong style="color: #f1f5f9;">Kermany OCT2017 dataset</strong>
            (84,484 labeled OCT retinal scans across 4 classes).
            On the held-out test set of 968 images, the model achieves
            <strong style="color: #10b981;">99.90% accuracy</strong>
            (967/968 correctly classified).
        </p>
        <p style="color: #94a3b8; font-size: 0.92em; line-height: 1.7; margin-top: 12px;">
            <strong style="color: #06b6d4;">Interpretability:</strong>
            Every prediction is accompanied by a real-time Grad-CAM heatmap generated by hooking into
            <code style="background: #111827; color: #06b6d4; padding: 2px 6px; border-radius: 4px; font-size: 0.88em;">model.features[-1]</code>.
        </p>
        <p style="color: #94a3b8; font-size: 0.92em; line-height: 1.7; margin-top: 12px;">
            <strong style="color: #8b5cf6;">Input Validation:</strong>
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
    print("Starting RetinaScan AI (Premium Dark UI)...")
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