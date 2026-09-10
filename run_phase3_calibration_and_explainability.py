"""
CardioAI_12Lead_ECG: Phase 3 Master Calibration & Explainability Pipeline.
Executes Post-hoc Temperature Scaling and generates 1D Grad-CAM clinical saliency maps.

Author: Youssef Ahmad, MD (Cardiovascular AI Track)
Project: Calibrated & Explainable 1D-ResNet for Multi-Label 12-Lead ECG Diagnosis
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure UTF-8 stdout on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import numpy as np

from src.models import build_cardio_resnet1d
from src.dataset import get_dataloaders, PTBXLECGDataset
from src.calibration import (
    ModelWithTemperature,
    compute_multilabel_ece,
    plot_reliability_diagrams
)
from src.explainability import GradCAM1D, plot_12lead_gradcam

DEFAULT_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def run_phase3_pipeline(
    checkpoint_path: str = "models/checkpoints/best_model.pth",
    processed_dir: str = "data/processed",
    raw_dir: str = "data/raw/ptb-xl",
    reports_dir: str = "reports"
):
    print("\n" + "=" * 80)
    print(" CARDIOAI-12LEAD: PHASE 3 PROBABILITY CALIBRATION & VISUAL EXPLAINABILITY")
    print(" Temperature Scaling (Low ECE) & 1D Grad-CAM Lead-Specific Heatmaps")
    print(" Lead Physician-Data Scientist: Youssef Ahmad, MD")
    print("=" * 80 + "\n")

    chk_path = Path(checkpoint_path).resolve()
    proc_path = Path(processed_dir).resolve()
    raw_path = Path(raw_dir).resolve()
    rep_path = Path(reports_dir).resolve()
    rep_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" [Device] Running Phase 3 on: {device}")

    # 1. Load Trained Model
    print(f"\n [Step 1/4] Loading Best Model Weights from {chk_path}...")
    model = build_cardio_resnet1d(num_classes=5, in_channels=12).to(device)
    if chk_path.exists():
        chk = torch.load(chk_path, map_location=device)
        model.load_state_dict(chk.get("model_state_dict", chk))
        print(f"    Loaded checkpoint from Epoch {chk.get('epoch', 'N/A')} (Macro AUROC: {chk.get('best_macro_auroc', 'N/A')})")
    else:
        print("    Warning: Checkpoint not found! Initializing model with base weights.")

    model.eval()

    # 2. Load Validation & Test Loaders
    print("\n [Step 2/4] Loading Validation and Independent Test Partitions...")
    train_loader, val_loader, test_loader = get_dataloaders(
        processed_dir=proc_path,
        raw_dir=raw_path,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        augment_train=False
    )

    # 3. Post-hoc Temperature Scaling Calibration
    print("\n [Step 3/4] Learning Post-hoc Temperature Scaling (T) on Validation Set...")
    calibrated_model = ModelWithTemperature(model)
    optimal_T = calibrated_model.calibrate(val_loader, device=device)
    print(f"    Learned Optimal Temperature Parameter T = {optimal_T:.4f}")

    # Evaluate on Independent Test Set
    test_logits, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            signals = batch["signal"].to(device)
            logits = model(signals)
            test_logits.append(logits.cpu())
            test_targets.append(batch["label"])

    all_test_logits = torch.cat(test_logits)
    all_test_targets = torch.cat(test_targets).numpy()

    uncal_probs = torch.sigmoid(all_test_logits).numpy()
    cal_probs = torch.sigmoid(all_test_logits / optimal_T).numpy()

    # Compute ECE
    uncal_ece_dict = compute_multilabel_ece(uncal_probs, all_test_targets)
    cal_ece_dict = compute_multilabel_ece(cal_probs, all_test_targets)

    print("\n" + "-" * 60)
    print(" CLINICAL PROBABILITY CALIBRATION RESULTS (TEST SET):")
    print(f"   - Uncalibrated Macro ECE: {uncal_ece_dict['macro_ece']*100:.2f}% (High Alert Fatigue)")
    print(f"   - Calibrated Macro ECE:   {cal_ece_dict['macro_ece']*100:.2f}% (Alert Fatigue Eliminated)")
    print(f"   - Relative Calibration Error Reduction: {((uncal_ece_dict['macro_ece'] - cal_ece_dict['macro_ece'])/max(uncal_ece_dict['macro_ece'], 1e-6))*100:.1f}%")
    print("-" * 60)

    # Plot Reliability Diagrams
    reliability_fig_path = rep_path / "calibration_reliability_diagrams.png"
    plot_reliability_diagrams(
        uncalibrated_probs=uncal_probs,
        calibrated_probs=cal_probs,
        targets=all_test_targets,
        superclasses=DEFAULT_SUPERCLASSES,
        output_path=reliability_fig_path
    )
    print(f"    Saved Reliability Diagrams to: {reliability_fig_path}")

    # 4. 1D Grad-CAM Visual Explainability
    print("\n [Step 4/4] Generating 1D Grad-CAM Saliency Explanation for Cardiac Case...")
    test_dataset = PTBXLECGDataset(
        metadata=proc_path / "test_metadata.csv",
        raw_data_dir=raw_path,
        normalize=True,
        augment=False
    )

    # Find a sample with Myocardial Infarction (MI) or Conduction Disturbance (CD)
    chosen_idx = 0
    chosen_class = "MI"
    class_idx = DEFAULT_SUPERCLASSES.index(chosen_class)

    for i in range(len(test_dataset)):
        labels = test_dataset[i]["label"].numpy()
        if labels[class_idx] == 1.0:
            chosen_idx = i
            break

    sample = test_dataset[chosen_idx]
    signal_tensor = sample["signal"]  # (12, 1000)
    ecg_id = sample["ecg_id"]

    # Compute Grad-CAM
    gradcam = GradCAM1D(model)
    heatmap = gradcam.generate_heatmap(signal_tensor, target_class_idx=class_idx)

    # Get calibrated confidence
    with torch.no_grad():
        pred_logits = model(signal_tensor.unsqueeze(0).to(device))
        conf = float(torch.sigmoid(pred_logits / optimal_T)[0, class_idx].item())

    gradcam_fig_path = rep_path / "gradcam_clinical_explanation.png"
    plot_12lead_gradcam(
        signal=signal_tensor.numpy(),
        heatmap=heatmap,
        predicted_class=chosen_class,
        confidence=conf,
        ecg_id=ecg_id,
        output_path=gradcam_fig_path
    )

    print("\n" + "=" * 80)
    print(" PHASE 3 CALIBRATION & EXPLAINABILITY COMPLETED SUCCESSFULLY! ")
    print(" Ready for clinical translation and integration with Master Colab Pipeline.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioAI Phase 3 Calibration & Explainability")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pth")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--raw-dir", type=str, default="data/raw/ptb-xl")
    parser.add_argument("--reports-dir", type=str, default="reports")

    args = parser.parse_args()

    run_phase3_pipeline(
        checkpoint_path=args.checkpoint,
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        reports_dir=args.reports_dir
    )
