"""
RetinaScan AI — Local Prediction UI (Laptop Backup)
====================================================

Standalone Python script that launches the same Gradio interface locally.
Use this if Kaggle/internet is unavailable during the demo.

USAGE
-----
1. Install dependencies (one-time):
   pip install torch torchvision gradio opencv-python pillow numpy

2. Update MODEL_PATH below to point to your downloaded .pth file.

3. Run:
   python retinascan_ui_local.py

4. Open the URL that appears (usually http://127.0.0.1:7860) in a browser.

Team: Murali A, Niranjan T, Praveen K
Guide: Dr. C. Santhosh Kumar
Sona College of Technology — Department of Information Technology
"""

import os
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
# CONFIGURATION — UPDATE THIS
# ================================================================
# On Windows, use forward slashes or raw strings, e.g.:
#   MODEL_PATH = r"C:\Users\murul\OneDrive\Desktop\best_convnext_tiny_oct2017.pth"
#   MODEL_PATH = "C:/Users/murul/OneDrive/Desktop/best_convnext_tiny_oct2017.pth"

MODEL_PATH = r"C:\Users\murul\Downloads\RetinaScan_AI\best_convnext_tiny_oct2017.pth"

# Optional — folder with example OCT images to preload in the UI
# Leave as None if you don't have example images ready
EXAMPLE_IMAGES_DIR = r"C:\Users\murul\OneDrive\Desktop\demo_images"

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
if device.type == "cpu":
    print("⚠  Running on CPU — inference will be ~2-3 seconds per image (still fine for demo)")

# ================================================================
# MODEL LOADING
# ================================================================
assert os.path.exists(MODEL_PATH), (
    f"❌ Model file not found at:\n   {MODEL_PATH}\n"
    f"Download best_convnext_tiny_oct2017.pth from Kaggle → Output tab → save to that path."
)

print(f"Loading model from {MODEL_PATH}...")
model = convnext_tiny(weights=None)
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, NUM_CLASSES)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()
print("✓ Model loaded (ConvNeXt-Tiny, 28.5M parameters, 99.90% test accuracy)")

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
    "CNV": "**Choroidal Neovascularization** — abnormal blood vessel growth beneath the retina, associated with wet age-related macular degeneration. Requires urgent ophthalmological review.",
    "DME": "**Diabetic Macular Edema** — fluid buildup in the macula due to diabetes complications, a leading cause of vision loss in diabetic patients.",
    "DRUSEN": "**Drusen deposits** — yellowish deposits at the retinal pigment epithelium, an early indicator of age-related macular degeneration.",
    "NORMAL": "**Healthy retina** — no significant pathological features detected. Normal retinal architecture visible.",
}

# ================================================================
# PREDICTION FUNCTION
# ================================================================
def predict_and_explain(image):
    if image is None:
        return None, {c: 0.0 for c in CLASSES}, "Please upload an OCT retinal scan image."

    pil_img = Image.fromarray(image).convert("L")
    display_img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    heatmap, pred_idx, probs = gradcam(input_tensor)
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx] * 100

    heatmap_resized = cv2.resize(heatmap, (display_img.shape[1], display_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    display_rgb = np.stack([display_img] * 3, axis=-1)
    overlay = cv2.addWeighted(display_rgb, 0.55, heatmap_color, 0.45, 0)

    conf_dict = {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    explanation = (
        f"### Predicted: **{pred_class}** — {confidence:.2f}% confidence\n\n"
        f"{DESCRIPTIONS[pred_class]}\n\n"
        f"---\n"
        f"**How to read the heatmap:** The colored overlay on the right shows the retinal region "
        f"the model focused on to make this prediction. Red/yellow = high attention, blue = low attention. "
        f"For a valid diagnosis, attention should be centered on clinically relevant anatomy.\n\n"
        f"*Model: ConvNeXt-Tiny (28.5M params) | Trained on Kermany OCT2017 (84,484 images) | "
        f"Test accuracy: 99.90%*"
    )
    return overlay, conf_dict, explanation

# ================================================================
# GRADIO INTERFACE
# ================================================================
# Preload example images if available
example_list = []
if EXAMPLE_IMAGES_DIR and os.path.exists(EXAMPLE_IMAGES_DIR):
    for f in sorted(os.listdir(EXAMPLE_IMAGES_DIR)):
        if f.lower().endswith((".jpeg", ".jpg", ".png")):
            example_list.append([os.path.join(EXAMPLE_IMAGES_DIR, f)])
    print(f"✓ Loaded {len(example_list)} example images")

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    title="RetinaScan AI"
) as demo:

    gr.Markdown(
        """
        # 🔬 RetinaScan AI — OCT Retinal Disease Classifier

        **Automated deep learning system for classifying OCT retinal scans into CNV, DME, Drusen, or Normal — with visual explanation of the model's reasoning via Grad-CAM heatmaps.**

        *Mini Project | Team: Murali A, Niranjan T, Praveen K | Guide: Dr. C. Santhosh Kumar | Sona College of Technology — Department of Information Technology*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Step 1: Upload an OCT Scan")
            input_image = gr.Image(type="numpy", label="OCT Retinal Scan", height=340)
            submit_btn = gr.Button("🔍 Analyze Scan", variant="primary", size="lg")
            clear_btn = gr.Button("Clear", size="sm")

        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Step 2: Prediction with Explanation")
            output_confidence = gr.Label(num_top_classes=4, label="Confidence per Class")
            output_heatmap = gr.Image(type="numpy",
                                      label="Grad-CAM Attention Heatmap Overlay",
                                      height=280)

    output_explanation = gr.Markdown()

    if example_list:
        gr.Markdown("### 🧪 Or try one of these test examples (click to auto-run)")
        gr.Examples(examples=example_list, inputs=input_image)

    gr.Markdown(
        """
        ---
        **Model:** ConvNeXt-Tiny (28.5M parameters) fine-tuned on Kermany OCT2017 dataset (84,484 images)
        **Test Accuracy:** 99.90% (967/968 correctly classified on held-out test set)
        **Base Paper:** Ajmal et al. (2026), *Experimental Eye Research* — this work exceeds the base paper baseline (97.54% VGG19) with ConvNeXt-Tiny (2022) and clinically validated Grad-CAM interpretability.
        """
    )

    submit_btn.click(predict_and_explain, inputs=input_image,
                     outputs=[output_heatmap, output_confidence, output_explanation])
    clear_btn.click(lambda: (None, None, {}, ""),
                    outputs=[input_image, output_heatmap, output_confidence, output_explanation])

if __name__ == "__main__":
    # share=False → local only (no internet needed once packages installed)
    # Change to share=True if you want a public URL from your laptop too
    demo.launch(share=False, inbrowser=True)
