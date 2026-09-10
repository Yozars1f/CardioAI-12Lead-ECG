"""
CardioAI_12Lead_ECG: Phase 2 Master Training Pipeline.
Trains 1D-ResNet with Asymmetric Focal Loss for Multi-Label 12-Lead ECG Classification.

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
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from src.models import build_cardio_resnet1d
from src.losses import AsymmetricFocalLoss
from src.dataset import get_dataloaders
from src.training import ClinicalECGTrainer


def plot_training_curves(history: dict, output_path: Path):
    """Generates a clinical training report plot showing loss and validation AUROC progression."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Loss Curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss (AFL)", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss (AFL)", color="#d62728", linewidth=2, linestyle="--")
    ax1.set_title("Asymmetric Focal Loss (AFL) Progression", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True)

    # 2. AUROC Curve
    ax2.plot(epochs, history["val_macro_auroc"], label="Validation Macro AUROC", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, history["val_macro_auprc"], label="Validation Macro AUPRC", color="#9467bd", linewidth=2, linestyle=":")
    ax2.set_title("Clinical Discrimination Metrics (AUROC & AUPRC)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True)

    plt.suptitle("CardioAI 1D-ResNet: Phase 2 Training & Validation Dynamics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" [OK] Saved training curves report to: {output_path}")


def run_phase2_training(
    processed_dir: str = "data/processed",
    raw_dir: str = "data/raw/ptb-xl",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip_margin: float = 0.05,
    checkpoint_dir: str = "models/checkpoints"
):
    print("\n" + "=" * 80)
    print(" CARDIOAI-12LEAD: PHASE 2 DEEP LEARNING MODELING & TRAINING")
    print(" Architecture: PyTorch 1D-ResNet | Loss: Asymmetric Focal Loss (AFL)")
    print(" Lead Physician-Data Scientist: Youssef Ahmad, MD")
    print("=" * 80 + "\n")

    proc_path = Path(processed_dir).resolve()
    raw_path = Path(raw_dir).resolve()
    chk_path = Path(checkpoint_dir).resolve()
    chk_path.mkdir(parents=True, exist_ok=True)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" [Hardware] Execution Device: {device}")
    if device.type == "cuda":
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load DataLoaders
    print("\n [Step 1/5] Building PyTorch DataLoaders (Train, Val, Test)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        processed_dir=proc_path,
        raw_dir=raw_path,
        batch_size=batch_size,
        num_workers=0,  # Safe cross-platform
        pin_memory=(device.type == "cuda"),
        augment_train=True
    )
    print(f"   - Train batches: {len(train_loader)} (Batch size: {batch_size})")
    print(f"   - Val batches:   {len(val_loader)}")
    print(f"   - Test batches:  {len(test_loader)}")

    # 2. Build 1D-ResNet Model
    print("\n [Step 2/5] Initializing 1D-ResNet Architecture...")
    model = build_cardio_resnet1d(num_classes=5, in_channels=12).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Total Trainable Parameters: {total_params:,}")
    print(f"   - Input Dimension:  (B, 12, 1000)")
    print(f"   - Output Dimension: (B, 5) [NORM, MI, STTC, CD, HYP]")

    # 3. Load Positive Class Weights & Build AFL Loss
    print("\n [Step 3/5] Configuring Asymmetric Focal Loss (AFL)...")
    summary_file = proc_path / "class_summary.json"
    pos_weights_tensor = None
    if summary_file.exists():
        with open(summary_file, "r") as f:
            stats = json.load(f)
        classes = stats["superclasses"]
        weights_list = [stats["recommended_pos_weights"][c] for c in classes]
        pos_weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)
        print(f"   - Loaded Positive Class Weights: {dict(zip(classes, weights_list))}")

    criterion = AsymmetricFocalLoss(
        gamma_neg=gamma_neg,
        gamma_pos=gamma_pos,
        clip_margin=clip_margin,
        pos_weight=pos_weights_tensor
    )
    print(f"   - Focusing Hyperparameters: gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, margin={clip_margin}")

    # 4. Optimizer & LR Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Execute Training Loop
    print(f"\n [Step 4/5] Launching Clinical Training Engine for {epochs} Epochs...")
    trainer = ClinicalECGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=str(chk_path)
    )

    history = trainer.fit(epochs=epochs)

    # 6. Final Evaluation on Independent Test Set (Fold 10)
    print("\n [Step 5/5] Conducting Final Clinical Evaluation on Independent Test Set (Fold 10)...")
    # Load best model
    best_weights_path = chk_path / "best_model.pth"
    if best_weights_path.exists():
        checkpoint = torch.load(best_weights_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"   - Loaded best model weights from Epoch {checkpoint['epoch']} (Val AUROC: {checkpoint['best_macro_auroc']:.4f})")

    test_loss, test_metrics = trainer.evaluate(test_loader)
    print("\n" + "-" * 60)
    print(" INDEPENDENT TEST SET (FOLD 10) CLINICAL PERFORMANCE:")
    print(f"   - Test Loss (AFL):   {test_loss:.4f}")
    print(f"   - Macro AUROC:       {test_metrics['macro_auroc']:.4f}")
    print(f"   - Macro AUPRC:       {test_metrics['macro_auprc']:.4f}")
    print("\n Per-Class Diagnostic AUROC:")
    for cls, auc in test_metrics["per_class_auroc"].items():
        ap = test_metrics["per_class_auprc"][cls]
        print(f"   - {cls:<6}: AUROC = {auc:.4f} | AUPRC = {ap:.4f}")
    print("-" * 60)

    # Plot & Save Report Curves
    plot_training_curves(history, Path("reports/training_curves.png"))

    print("\n" + "=" * 80)
    print(" PHASE 2 MODELING & TRAINING COMPLETED SUCCESSFULLY! ")
    print(" Model weights and training history safely saved in models/checkpoints/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioAI Phase 2 Model Training Runner")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--gamma-neg", type=float, default=4.0)
    parser.add_argument("--gamma-pos", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")

    args = parser.parse_args()

    run_phase2_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma_neg=args.gamma_neg,
        gamma_pos=args.gamma_pos,
        checkpoint_dir=args.checkpoint_dir
    )
