# ================================================================
# RETINASCAN AI — Analysis History (JSON-based local storage)
# ================================================================

import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_FILE = DATA_DIR / "history.json"


def _ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def load_history():
    """Load all analysis history records."""
    _ensure_data_dir()
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_analysis(result):
    """Save an analysis result to history."""
    _ensure_data_dir()
    history = load_history()

    record = {
        "scan_id": result.get("scan_id", ""),
        "timestamp": result.get("timestamp", datetime.now().isoformat()),
        "prediction": result.get("prediction"),
        "prediction_full": result.get("prediction_full", ""),
        "confidence": result.get("confidence", 0.0),
        "valid": result.get("valid", False),
        "status": result.get("status", ""),
        "probabilities": result.get("probabilities", {}),
        "clinical_description": result.get("clinical_description", ""),
        "attention_region": result.get("attention_region", ""),
        "interpretation": result.get("interpretation", ""),
        "original_b64": result.get("original_b64"),
        "heatmap_b64": result.get("heatmap_b64"),
        "overlay_b64": result.get("overlay_b64"),
    }

    history.insert(0, record)

    # Keep max 200 records
    history = history[:200]

    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return record


def get_analysis_by_id(scan_id):
    """Retrieve a single analysis by scan_id."""
    history = load_history()
    for record in history:
        if record.get("scan_id") == scan_id:
            return record
    return None


def delete_analysis(scan_id):
    """Delete an analysis by scan_id."""
    history = load_history()
    history = [r for r in history if r.get("scan_id") != scan_id]
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return True


def clear_history():
    """Clear all history."""
    _ensure_data_dir()
    HISTORY_FILE.write_text("[]", encoding="utf-8")
    return True


def get_statistics():
    """Compute usage statistics from history."""
    history = load_history()
    valid_analyses = [r for r in history if r.get("valid")]

    total = len(history)
    completed = len(valid_analyses)
    rejected = total - completed

    # Class distribution
    class_counts = {"CNV": 0, "DME": 0, "DRUSEN": 0, "NORMAL": 0}
    confidences = []

    for r in valid_analyses:
        pred = r.get("prediction")
        if pred in class_counts:
            class_counts[pred] += 1
        conf = r.get("confidence", 0)
        if conf > 0:
            confidences.append(conf)

    avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

    return {
        "total_scans": total,
        "completed": completed,
        "rejected": rejected,
        "avg_confidence": avg_confidence,
        "class_distribution": class_counts,
    }
