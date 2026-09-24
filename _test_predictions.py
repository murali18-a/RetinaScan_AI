"""Quick end-to-end test: load model, run predictions on all 4 classes + non-OCT rejection."""
import os, sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

print("Loading app module (this loads the model)...")
import app

print("\n" + "="*60)
print("RUNTIME VERIFICATION TESTS")
print("="*60)

demo_dir = app.EXAMPLE_IMAGES_DIR
test_files = {
    "CNV": "CNV-1016042-1.jpeg",
    "DME": "DME-119840-1.jpeg",
    "DRUSEN": "DRUSEN-1246453-1.jpeg",
    "NORMAL": "NORMAL-1073137-1.jpeg",
}

all_pass = True

print("\n--- Test Suite 1: Prediction Accuracy ---")
for expected_class, fname in test_files.items():
    fpath = os.path.join(str(demo_dir), fname)
    img = np.array(Image.open(fpath).convert("RGB"))
    conf_dict, overlay, explanation, html_banner = app.predict_and_explain(img)

    predicted = max(conf_dict, key=conf_dict.get)
    confidence = conf_dict[predicted]
    has_overlay = overlay is not None
    has_why = "Why this prediction" in explanation or app.DYNAMIC_WHY[expected_class][:30] in explanation

    correct = (predicted == expected_class) and has_overlay and has_why
    status = "PASS" if correct else "FAIL"
    if not correct:
        all_pass = False
    print(f"  [{status}] {expected_class}: predicted={predicted} ({confidence*100:.2f}%), "
          f"overlay={'YES' if has_overlay else 'NO'}, why_text={'OK' if has_why else 'FAIL'}")

print("\n--- Test Suite 2: Status Banner Check ---")
for expected_class, fname in test_files.items():
    fpath = os.path.join(str(demo_dir), fname)
    img = np.array(Image.open(fpath).convert("RGB"))
    conf_dict, overlay, explanation, html_banner = app.predict_and_explain(img)

    has_valid_banner = "Valid OCT Input" in html_banner
    status = "PASS" if has_valid_banner else "FAIL"
    if not has_valid_banner:
        all_pass = False
    print(f"  [{status}] {expected_class}: banner={'VALID' if has_valid_banner else 'OTHER'}")

print("\n--- Test Suite 3: Non-OCT Input Rejection ---")
nonoct_path = os.path.join(os.path.dirname(__file__), "test_nonoct.jpg")
if os.path.exists(nonoct_path):
    img = np.array(Image.open(nonoct_path).convert("RGB"))
    conf_dict, overlay, explanation, html_banner = app.predict_and_explain(img)
    rejected = "Invalid Input" in html_banner
    no_pred = overlay is None and all(v == 0.0 for v in conf_dict.values())

    correct_rejection = rejected and no_pred
    status = "PASS" if correct_rejection else "FAIL"
    if not correct_rejection:
        all_pass = False
    print(f"  [{status}] Non-OCT: rejected={'YES' if rejected else 'NO'}, "
          f"no_prediction={'YES' if no_pred else 'NO'}")
else:
    print(f"  [SKIP] test_nonoct.jpg not found")

print("\n--- Test Suite 4: Input Validation Function ---")
for expected_class, fname in test_files.items():
    fpath = os.path.join(str(demo_dir), fname)
    img = np.array(Image.open(fpath).convert("RGB"))
    is_valid, reasons, is_hard = app.validate_oct_input(img)
    status = "PASS" if is_valid else "FAIL"
    if not is_valid:
        all_pass = False
    print(f"  [{status}] {expected_class}: valid={is_valid}"
          + (f", reasons={reasons}" if reasons else ""))

if os.path.exists(nonoct_path):
    img = np.array(Image.open(nonoct_path).convert("RGB"))
    is_valid, reasons, is_hard = app.validate_oct_input(img)
    status = "PASS" if not is_valid else "FAIL"
    if is_valid:
        all_pass = False
    print(f"  [{status}] Non-OCT: valid={is_valid}, hard_reject={is_hard}")

print("\n" + "="*60)
if all_pass:
    print("ALL TESTS PASSED SUCCESSFULLY!")
else:
    print("SOME TESTS FAILED - review above")
print("="*60)
