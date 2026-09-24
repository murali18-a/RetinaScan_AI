import gradio as gr

# Quick smoke test: verify Blocks accepts theme and css via kwargs
try:
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo", neutral_hue="slate"),
        title="Test",
        css="footer {display: none !important;}",
    ) as demo:
        gr.Markdown("# Test")
        btn = gr.Button("Click")
    print("Blocks + theme + css: OK")
except Exception as e:
    print(f"Blocks error: {e}")

# Check if gr.Label exists
try:
    with gr.Blocks() as demo2:
        lbl = gr.Label(num_top_classes=4, label="Test")
    print("gr.Label: OK")
except Exception as e:
    print(f"Label error: {e}")

# Check if gr.Examples exists and its signature
import inspect
sig = inspect.signature(gr.Examples.__init__)
print(f"gr.Examples params: {list(sig.parameters.keys())}")
