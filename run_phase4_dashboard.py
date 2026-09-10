"""
===============================================================================
CardioAI 12-Lead ECG - Phase 4: Master Clinical Dashboard Launcher
===============================================================================
Author: Dr. Youssef Ahmed (MD, Physician-Data Scientist)
Track: Clinical Cardiovascular Data Science & Translational AI
===============================================================================
"""

import sys
import webbrowser
from pathlib import Path
import uvicorn

def main():
    print("=" * 75)
    print("CardioAI-12Lead: Clinical Decision Support System (CDSS)")
    print("Translational Clinical AI Architecture | Dr. Youssef Ahmed (MD)")
    print("=" * 75)

    ckpt_path = Path("models/checkpoints/best_cardio_resnet1d.pth")
    if not ckpt_path.exists():
        print(f"Warning: Model checkpoint not found at {ckpt_path}")
    else:
        print(f"Certified Model Weights Detected: {ckpt_path} ({ckpt_path.stat().st_size / (1024*1024):.1f} MB)")

    print("\nStarting Clinical Dashboard on http://127.0.0.1:8050 ...")
    print("Press Ctrl+C to terminate the server.\n")

    # Automatically open the browser
    webbrowser.open("http://127.0.0.1:8050")

    from src.dashboard.server import app
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="info")

if __name__ == "__main__":
    main()
