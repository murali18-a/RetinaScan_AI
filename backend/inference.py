# ================================================================
# RETINASCAN AI — Inference Backend
# ================================================================
# Extracted from app.py — preserves ALL existing ML logic exactly.
# This module is imported by server.py (Flask) without modifying
# any model behavior, validation thresholds, or Grad-CAM logic.
# ================================================================

import os
import sys
import math
import io
import base64
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny
from torchvision import transforms

# Fix Windows console encoding
if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ================================================================
# CONFIGURATION (unchanged from app.py)
# ================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best_convnext_tiny_oct2017.pth"
EXAMPLE_IMAGES_DIR = BASE_DIR / "demo_image"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = 4
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Validation thresholds (unchanged)
MAX_MEAN_SATURATION = 25.0
MAX_CHANNEL_DIFF = 20.0
MAX_LAPLACIAN_VAR = 3500.0
MIN_HORIZ_EDGE_RATIO = 0.35
HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.70
MAX_ENTROPY = math.log(4)
HIGH_ENTROPY_RATIO = 0.75


# ================================================================
# DISEASE DESCRIPTIONS (unchanged from app.py)
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
# GRAD-CAM (unchanged from app.py)
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


# ================================================================
# INPUT VALIDATION (unchanged from app.py)
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
        reasons.append(f"High color saturation (mean S = {mean_sat:.1f})")
        is_hard_reject = True
    if channel_diff > MAX_CHANNEL_DIFF:
        reasons.append(f"Significant RGB channel differences (diff = {channel_diff:.1f})")
        is_hard_reject = True

    if is_hard_reject:
        return False, reasons, True

    # Check 3: Laplacian variance
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_resized = cv2.resize(gray, (224, 224))
    lap_var = float(cv2.Laplacian(gray_resized, cv2.CV_64F).var())

    if lap_var > MAX_LAPLACIAN_VAR:
        reasons.append(f"Unusually sharp edges (Laplacian variance = {lap_var:.0f})")
        is_hard_reject = True

    # Check 4: Edge orientation
    sobel_x = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
    total_energy = float(np.sum(sobel_y**2)) + float(np.sum(sobel_x**2)) + 1e-10
    horiz_ratio = float(np.sum(sobel_y**2)) / total_energy

    if horiz_ratio < MIN_HORIZ_EDGE_RATIO:
        reasons.append(f"Edge orientation inconsistent with OCT (horizontal ratio = {horiz_ratio:.2f})")
        is_hard_reject = True

    if is_hard_reject:
        return False, reasons, True

    # Check 5: Confidence + entropy (soft warning)
    if softmax_probs is not None:
        top_conf = float(np.max(softmax_probs))
        entropy = -float(np.sum(softmax_probs * np.log(softmax_probs + 1e-10)))
        entropy_ratio = entropy / MAX_ENTROPY
        if top_conf < MEDIUM_CONFIDENCE and entropy_ratio > HIGH_ENTROPY_RATIO:
            reasons.append(f"Very low model confidence ({top_conf*100:.1f}%)")
            return False, reasons, False

    return True, [], False


# ================================================================
# MODEL SINGLETON
# ================================================================
_model = None
_gradcam = None
_preprocess = None
_device = None


def _load_model():
    """Load model once (singleton pattern)."""
    global _model, _gradcam, _preprocess, _device

    if _model is not None:
        return

    print("\n=======================================================")
    print("       RETINASCAN AI - INITIALIZING")
    print("=======================================================")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {_device}")

    assert MODEL_PATH.exists(), (
        f"\n[ERROR] Model not found at:\n   {MODEL_PATH}\n"
        f"Ensure best_convnext_tiny_oct2017.pth is in the project root."
    )

    print("Loading ConvNeXt-Tiny model...")
    _model = convnext_tiny(weights=None)
    num_features = _model.classifier[2].in_features
    _model.classifier[2] = nn.Linear(num_features, NUM_CLASSES)

    checkpoint = torch.load(str(MODEL_PATH), map_location=_device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        _model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        _model.load_state_dict(checkpoint["state_dict"])
    else:
        _model.load_state_dict(checkpoint)

    _model = _model.to(_device)
    _model.eval()

    _gradcam = GradCAM(_model, _model.features[-1])

    _preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print("[OK] Model loaded (ConvNeXt-Tiny, 28.5M params, 99.90% test accuracy)")
    print("=======================================================\n")


def get_model():
    _load_model()
    return _model, _gradcam, _preprocess, _device


# ================================================================
# NUMPY ARRAY → BASE64 PNG
# ================================================================
def numpy_to_base64(img_array):
    """Convert a numpy image array to a base64-encoded PNG string."""
    if img_array is None:
        return None
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ================================================================
# MAIN ANALYSIS FUNCTION
# ================================================================
def analyze_image(image_array):
    """
    Full analysis pipeline: validate → classify → Grad-CAM → return results dict.

    Args:
        image_array: numpy array (H, W, C) in RGB format

    Returns:
        dict with all analysis results
    """
    model, gradcam_fn, preprocess_fn, device = get_model()

    scan_id = f"RS-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().isoformat()

    # Step 1: Hard validation
    is_valid, reasons, is_hard = validate_oct_input(image_array)
    if is_hard:
        return {
            "scan_id": scan_id,
            "timestamp": timestamp,
            "valid": False,
            "hard_reject": True,
            "rejection_reasons": reasons,
            "prediction": None,
            "confidence": 0.0,
            "probabilities": {},
            "original_b64": numpy_to_base64(image_array),
            "heatmap_b64": None,
            "overlay_b64": None,
        }

    # Step 2: Preprocessing
    pil_img = Image.fromarray(image_array).convert("L")
    display_img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    display_rgb = np.stack([display_img] * 3, axis=-1)
    input_tensor = preprocess_fn(pil_img).unsqueeze(0).to(device)

    # Step 3: Prediction + Grad-CAM
    heatmap, pred_idx, probs = gradcam_fn(input_tensor)
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # Step 4: Soft validation
    is_valid_soft, soft_reasons, _ = validate_oct_input(image_array, softmax_probs=probs)

    # Step 5: Grad-CAM heatmap suppression for NORMAL
    if pred_class == "NORMAL":
        threshold = 0.6
        heatmap_suppressed = np.where(heatmap > threshold, heatmap * 0.15, heatmap * 0.05)
        heatmap_for_overlay = np.clip(heatmap_suppressed, 0, 1)
    else:
        heatmap_for_overlay = heatmap

    # Step 6: Generate visualization images
    heatmap_resized = cv2.resize(heatmap_for_overlay, (display_img.shape[1], display_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display_rgb, 0.55, heatmap_color, 0.45, 0)

    # Build probabilities dict
    probabilities = {CLASSES[i]: round(float(probs[i]) * 100, 2) for i in range(NUM_CLASSES)}

    # Determine status
    if not is_valid_soft:
        status = "low_confidence"
        status_reasons = soft_reasons
    elif confidence >= HIGH_CONFIDENCE:
        status = "high_confidence"
        status_reasons = []
    else:
        status = "moderate_confidence"
        status_reasons = []

    desc = DESCRIPTIONS[pred_class]

    return {
        "scan_id": scan_id,
        "timestamp": timestamp,
        "valid": True,
        "hard_reject": False,
        "rejection_reasons": [],
        "prediction": pred_class,
        "prediction_full": desc["full_name"],
        "confidence": round(confidence * 100, 2),
        "probabilities": probabilities,
        "status": status,
        "status_reasons": status_reasons,
        "clinical_description": desc["clinical"],
        "attention_region": desc["attention_region"],
        "interpretation": DYNAMIC_WHY[pred_class],
        "original_b64": numpy_to_base64(display_rgb),
        "heatmap_b64": numpy_to_base64(heatmap_color),
        "overlay_b64": numpy_to_base64(overlay),
    }


def get_example_images():
    """Return list of example image paths."""
    examples = []
    if EXAMPLE_IMAGES_DIR.exists():
        for f in sorted(EXAMPLE_IMAGES_DIR.iterdir()):
            if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
                examples.append({
                    "filename": f.name,
                    "path": str(f),
                    "class": f.name.split("-")[0],
                })
    return examples
