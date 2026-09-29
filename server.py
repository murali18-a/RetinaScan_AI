# ================================================================
# RETINASCAN AI — Flask Web Server
# ================================================================
# Professional medical imaging research platform.
# Wraps the existing ML backend (backend/inference.py) with a
# multi-page Flask web application.
# ================================================================

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from functools import wraps

import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session, send_from_directory,
)

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.inference import analyze_image, get_example_images, CLASSES, DESCRIPTIONS, BASE_DIR
from backend.history import (
    load_history, save_analysis, get_analysis_by_id,
    delete_analysis, clear_history, get_statistics,
)

# ================================================================
# FLASK APP
# ================================================================
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = "retinascan-ai-dev-key-change-in-production"

# ================================================================
# AUTH (simple demo session-based)
# ================================================================
DEMO_USER = {
    "email": "researcher@retinascan.ai",
    "password": "retinascan2024",
    "name": "Dr. Research User",
    "role": "Research Analyst",
    "institution": "Sona College of Technology",
}


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ================================================================
# ROUTES — PAGES
# ================================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        if email == DEMO_USER["email"] and password == DEMO_USER["password"]:
            session["logged_in"] = True
            session["user"] = DEMO_USER
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid email or password")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    stats = get_statistics()
    recent = load_history()[:8]
    return render_template("dashboard.html",
                           page="dashboard",
                           stats=stats,
                           recent=recent,
                           user=session.get("user", DEMO_USER))


@app.route("/analysis")
@login_required
def analysis():
    examples = get_example_images()
    return render_template("analysis.html",
                           page="analysis",
                           examples=examples,
                           user=session.get("user", DEMO_USER))


@app.route("/result/<scan_id>")
@login_required
def result(scan_id):
    record = get_analysis_by_id(scan_id)
    if not record:
        return redirect(url_for("history"))
    desc = DESCRIPTIONS.get(record.get("prediction", ""), {})
    return render_template("result.html",
                           page="analysis",
                           record=record,
                           desc=desc,
                           user=session.get("user", DEMO_USER))


@app.route("/history")
@login_required
def history():
    records = load_history()
    return render_template("history.html",
                           page="history",
                           records=records,
                           user=session.get("user", DEMO_USER))


@app.route("/analytics")
@login_required
def analytics():
    stats = get_statistics()
    # Load model evaluation metrics
    metrics_file = BASE_DIR / "final_metrics.json"
    training_file = BASE_DIR / "training_history.json"
    model_metrics = {}
    training_history = {}
    if metrics_file.exists():
        model_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    if training_file.exists():
        training_history = json.loads(training_file.read_text(encoding="utf-8"))
    return render_template("analytics.html",
                           page="analytics",
                           stats=stats,
                           model_metrics=model_metrics,
                           training_history=training_history,
                           user=session.get("user", DEMO_USER))


@app.route("/model-info")
@login_required
def model_info():
    metrics_file = BASE_DIR / "final_metrics.json"
    training_file = BASE_DIR / "training_history.json"
    model_metrics = {}
    training_history = {}
    if metrics_file.exists():
        model_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    if training_file.exists():
        training_history = json.loads(training_file.read_text(encoding="utf-8"))
    return render_template("model_info.html",
                           page="model_info",
                           model_metrics=model_metrics,
                           training_history=training_history,
                           user=session.get("user", DEMO_USER))


@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html",
                           page="settings",
                           user=session.get("user", DEMO_USER))


@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html",
                           page="profile",
                           user=session.get("user", DEMO_USER))


# ================================================================
# API ROUTES
# ================================================================
@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    """Accept an uploaded image and return analysis results."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        img_array = np.array(img)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    # Run inference
    result = analyze_image(img_array)

    # Save to history if valid
    if result.get("valid"):
        save_analysis(result)
    else:
        # Save rejected analyses too (without images to save space)
        save_result = result.copy()
        save_result["original_b64"] = None
        save_result["heatmap_b64"] = None
        save_result["overlay_b64"] = None
        save_analysis(save_result)

    return jsonify(result)


@app.route("/api/analyze-example", methods=["POST"])
@login_required
def api_analyze_example():
    """Analyze one of the built-in example images."""
    data = request.get_json()
    filename = data.get("filename", "")

    example_path = BASE_DIR / "demo_image" / filename
    if not example_path.exists():
        return jsonify({"error": "Example image not found"}), 404

    try:
        img = Image.open(str(example_path)).convert("RGB")
        img_array = np.array(img)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    result = analyze_image(img_array)
    if result.get("valid"):
        save_analysis(result)

    return jsonify(result)


@app.route("/api/history")
@login_required
def api_history():
    return jsonify(load_history())


@app.route("/api/history/<scan_id>", methods=["DELETE"])
@login_required
def api_delete_analysis(scan_id):
    delete_analysis(scan_id)
    return jsonify({"success": True})


@app.route("/api/history/clear", methods=["POST"])
@login_required
def api_clear_history():
    clear_history()
    return jsonify({"success": True})


@app.route("/api/statistics")
@login_required
def api_statistics():
    return jsonify(get_statistics())


# Serve demo images directly
@app.route("/demo-image/<path:filename>")
def serve_demo_image(filename):
    return send_from_directory(str(BASE_DIR / "demo_image"), filename)


# Serve project assets (confusion matrix, curves, etc.)
@app.route("/project-asset/<path:filename>")
def serve_project_asset(filename):
    return send_from_directory(str(BASE_DIR), filename)


# ================================================================
# LAUNCH
# ================================================================
if __name__ == "__main__":
    # Pre-load model at startup
    from backend.inference import get_model
    print("\n" + "=" * 60)
    print("  RETINASCAN AI — Professional Medical Platform")
    print("=" * 60)
    get_model()
    print("\nStarting Flask server...")
    print("  URL: http://127.0.0.1:7860")
    print("  Demo login: researcher@retinascan.ai / retinascan2024")
    print("=" * 60 + "\n")
    app.run(
        host="127.0.0.1",
        port=7860,
        debug=False,
    )
