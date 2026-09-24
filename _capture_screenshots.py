"""
Automated screenshot capture for RetinaScan AI project report.
Uses Selenium to interact with the Gradio UI and capture:
  1. Landing page
  2. Successful prediction (CNV)
  3. Non-OCT rejection
"""
import os
import sys
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_DIR = Path(__file__).resolve().parent
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# Test images
CNV_IMG = BASE_DIR / "demo_image" / "CNV-1016042-1.jpeg"
NONOCT_IMG = Path(r"C:\Users\murul\.gemini\antigravity-ide\brain\0cee9ab1-69c4-4237-8d7a-5c2552f4ab25\test_nonoct_1789059655448.png")

URL = "http://127.0.0.1:7860"


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--force-device-scale-factor=1")
    # Don't use headless so we get full rendering
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1920, 1080)
    return driver


def wait_for_gradio(driver, timeout=30):
    """Wait until Gradio app is fully loaded."""
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#analyze-btn"))
    )
    time.sleep(2)  # Extra time for CSS/fonts


def find_file_input(driver):
    """Find the hidden file input in Gradio's image upload component."""
    return driver.find_element(By.CSS_SELECTOR, "#oct-input-image input[type='file']")


def click_analyze(driver):
    """Click the Analyze Scan button."""
    btn = driver.find_element(By.CSS_SELECTOR, "#analyze-btn")
    btn.click()


def click_clear(driver):
    """Click the Clear button."""
    btn = driver.find_element(By.CSS_SELECTOR, "#clear-btn")
    btn.click()


def wait_for_prediction(driver, timeout=60):
    """Wait for prediction to complete (status banner appears)."""
    WebDriverWait(driver, timeout).until(
        lambda d: d.find_element(By.CSS_SELECTOR, "#status-banner").get_attribute("innerHTML").strip() != ""
    )
    time.sleep(1)  # Let animations finish


def save_full_page_screenshot(driver, name):
    """Save a full-page screenshot."""
    filepath = SCREENSHOTS_DIR / f"{name}.png"
    
    # Get full page height
    total_height = driver.execute_script("return document.body.scrollHeight")
    viewport_height = driver.execute_script("return window.innerHeight")
    
    # Resize window to capture full page
    driver.set_window_size(1920, total_height + 100)
    time.sleep(1)
    
    driver.save_screenshot(str(filepath))
    print(f"  [SAVED] {filepath}")
    
    # Restore window size
    driver.set_window_size(1920, 1080)
    time.sleep(0.5)
    
    return filepath


def main():
    print("=" * 60)
    print("RetinaScan AI — Automated Screenshot Capture")
    print("=" * 60)
    
    driver = setup_driver()
    
    try:
        # ─── SCREENSHOT 1: Landing Page ───
        print("\n[1/3] Capturing Landing Page...")
        driver.get(URL)
        wait_for_gradio(driver)
        time.sleep(3)  # Let animations and fonts load
        save_full_page_screenshot(driver, "01_landing_page")
        
        # ─── SCREENSHOT 2: Successful CNV Prediction ───
        print("\n[2/3] Capturing Successful Prediction (CNV)...")
        driver.get(URL)  # Fresh load
        wait_for_gradio(driver)
        
        # Upload CNV image via file input
        file_input = find_file_input(driver)
        file_input.send_keys(str(CNV_IMG))
        time.sleep(2)
        
        # Click Analyze
        click_analyze(driver)
        print("  Waiting for inference (this takes ~15-30s on CPU)...")
        wait_for_prediction(driver, timeout=120)
        time.sleep(2)
        
        save_full_page_screenshot(driver, "02_successful_prediction_cnv")
        
        # ─── SCREENSHOT 3: Non-OCT Rejection ───
        print("\n[3/3] Capturing Non-OCT Rejection...")
        driver.get(URL)  # Fresh load
        wait_for_gradio(driver)
        
        # Upload non-OCT image
        file_input = find_file_input(driver)
        file_input.send_keys(str(NONOCT_IMG))
        time.sleep(2)
        
        # Click Analyze
        click_analyze(driver)
        print("  Waiting for rejection...")
        wait_for_prediction(driver, timeout=30)
        time.sleep(2)
        
        save_full_page_screenshot(driver, "03_nonoct_rejection")
        
        print("\n" + "=" * 60)
        print("ALL 3 SCREENSHOTS CAPTURED SUCCESSFULLY")
        print(f"Saved to: {SCREENSHOTS_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
