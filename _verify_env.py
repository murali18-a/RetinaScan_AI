import gradio
print(f"gradio={gradio.__version__}")

import torch
ckpt = torch.load("best_convnext_tiny_oct2017.pth", map_location="cpu", weights_only=False)
print(f"Checkpoint type: {type(ckpt)}")
if isinstance(ckpt, dict):
    print(f"Keys: {list(ckpt.keys())}")
else:
    print("Raw state_dict (OrderedDict)")
print("Checkpoint loaded OK")
