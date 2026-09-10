"""
===============================================================================
CardioAI 12-Lead ECG - Phase 4: Production FastAPI Clinical Backend Server
===============================================================================
Author: Dr. Youssef Ahmed (MD, Physician-Data Scientist)
Track: Clinical Cardiovascular AI Translation & Decision Support
===============================================================================
"""

import io
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.dashboard.clinical_engine import (
    ClinicalInferenceEngine,
    LEAD_NAMES,
    SUPERCLASSES,
    SUPERCLASS_NAMES,
    get_preloaded_demo_case,
)

app = FastAPI(
    title="CardioAI-12Lead CDSS API",
    description="Translational AI Platform for Automated 12-Lead ECG Triage, Calibrated Inference & 1D Grad-CAM Explainability",
    version="1.0.0",
)

# Enable CORS for maximum client compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if present
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize Master Clinical Inference Engine
engine = ClinicalInferenceEngine(
    checkpoint_path="models/checkpoints/best_cardio_resnet1d.pth",
    temperature=1.28
)

DEMO_CASES = [
    {
        "id": "case_mi",
        "title": "Patient A (85yo M) — Acute Anterior & Inferolateral STEMI",
        "tag": "PTB-XL Record #02681",
        "severity": "CRITICAL",
        "history": "An 85-year-old male presenting with acute chest pressure; ECG demonstrates diagnostic ST-segment elevation in II, III, aVF, and V2-V6 with QS complexes, diagnostic of acute transmural infarction."
    },
    {
        "id": "case_cd",
        "title": "Patient B (81yo F) — Complete Left Bundle Branch Block (CLBBB)",
        "tag": "PTB-XL Record #00286",
        "severity": "HIGH",
        "history": "An 81-year-old female with cardiomyopathy; baseline tracing demonstrates classic broad, notched R-waves in lateral leads V5, V6, and deep broad QS complexes in V1 (QRS > 140 ms)."
    },
    {
        "id": "case_hyp",
        "title": "Patient C (75yo F) — Severe Ventricular Hypertrophy (LVH) with Strain",
        "tag": "PTB-XL Record #00296",
        "severity": "ATTENTION",
        "history": "A 75-year-old female with longstanding severe essential hypertension; ECG demonstrates extreme Sokolow-Lyon voltage (53.1 mm, R_V5=40 mm) with pronounced lateral repolarization strain."
    },
    {
        "id": "case_norm",
        "title": "Patient D (48yo F) — Unremarkable Normal Sinus Rhythm (NORM)",
        "tag": "PTB-XL Record #00042",
        "severity": "STABLE",
        "history": "An asymptomatic 48-year-old female undergoing routine pre-operative clearance with preserved electrical conduction and normal repolarization vectors."
    }
]


class TemperatureUpdateRequest(BaseModel):
    temperature: float


@app.get("/")
def serve_dashboard():
    """Serves the main single-page clinical dashboard."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard template not found.")
    return FileResponse(str(html_path))


@app.get("/api/health")
def health_check():
    """System status and verification endpoint."""
    return {
        "status": "ONLINE",
        "model_architecture": "1D-ResNet (8.7M Parameters)",
        "benchmark_dataset": "PhysioNet PTB-XL (21,799 Records | 18,869 Patients)",
        "macro_auroc": 0.9019,
        "zero_patient_leakage": "Verified",
        "temperature_scalar": engine.temperature,
        "lead_channels": LEAD_NAMES,
        "diagnostic_superclasses": SUPERCLASSES,
    }


@app.get("/api/cases")
def list_demo_cases():
    """Returns available preloaded clinical cases."""
    return {"cases": DEMO_CASES}


@app.get("/api/cases/{case_id}")
def get_demo_case_data(case_id: str):
    """Fetches waveform and metadata for a specific clinical demo case."""
    valid_ids = [c["id"] for c in DEMO_CASES]
    if case_id not in valid_ids:
        raise HTTPException(status_code=404, detail=f"Case ID '{case_id}' not found.")
    
    signal, title, history = get_preloaded_demo_case(case_id)
    return {
        "id": case_id,
        "title": title,
        "history": history,
        "leads": LEAD_NAMES,
        "time_seconds": 10.0,
        "sampling_rate_hz": 100,
        "signal": signal.tolist(),  # (12, 1000)
    }


@app.post("/api/analyze/case/{case_id}")
def analyze_case(case_id: str):
    """Executes full diagnostic inference and Grad-CAM on a preloaded clinical case."""
    valid_ids = [c["id"] for c in DEMO_CASES]
    if case_id not in valid_ids:
        raise HTTPException(status_code=404, detail=f"Case ID '{case_id}' not found.")

    target_map = {"case_mi": "MI", "case_cd": "CD", "case_hyp": "HYP", "case_norm": "NORM"}
    signal, title, history = get_preloaded_demo_case(case_id)
    analysis = engine.analyze(signal, target_class=target_map.get(case_id))
    
    return {
        "case_id": case_id,
        "case_title": title,
        "case_history": history,
        "signal": signal.tolist(),
        "leads": LEAD_NAMES,
        **analysis,
        "gradcam_heatmap": analysis["gradcam_heatmap"].tolist(),
    }


@app.post("/api/analyze/upload")
async def analyze_uploaded_csv(file: UploadFile = File(...)):
    """
    Accepts a 12-lead ECG CSV file (12 columns x 1000 timesteps, or 1000 rows x 12 columns).
    Executes full pipeline and returns diagnostic classification with Grad-CAM heatmaps.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are accepted.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Check if shape is (1000, 12) or (12, 1000)
        arr = df.values.astype(np.float32)
        if arr.shape == (1000, 12):
            signal = arr.T
        elif arr.shape == (12, 1000):
            signal = arr
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Incompatible ECG matrix dimensions: expected (1000, 12) or (12, 1000), got {arr.shape}."
            )

        analysis = engine.analyze(signal)
        return {
            "filename": file.filename,
            "case_title": f"Custom Clinical Upload: {file.filename}",
            "case_history": "12-Lead digital ECG waveform provided directly via clinical file upload.",
            "signal": signal.tolist(),
            "leads": LEAD_NAMES,
            **analysis,
            "gradcam_heatmap": analysis["gradcam_heatmap"].tolist(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process ECG signal: {str(e)}")


@app.post("/api/calibrate")
def update_temperature(req: TemperatureUpdateRequest):
    """Allows dynamic post-hoc temperature scaling adjustment."""
    engine.set_temperature(req.temperature)
    return {
        "temperature": engine.temperature,
        "message": f"Temperature successfully set to T = {engine.temperature:.4f}"
    }


def start_server(host: str = "127.0.0.1", port: int = 8050):
    """Programmatic entrypoint."""
    import uvicorn
    print(f"[*] CardioAI-12Lead CDSS Server listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
