"""
CardioAI_12Lead_ECG: Clinical AI Trainer & Evaluator.
Computes Macro AUROC, AUPRC, logs losses, and manages best model checkpointing.
Author: Youssef Ahmad, MD (Cardiovascular AI Track)
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger("CardioAI_Trainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] CardioAI-Train: %(message)s"
)

DEFAULT_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


class ClinicalECGTrainer:
    """
    Manages training, validation, multi-label metric computation (AUROC, AUPRC),
    and model checkpointing for 1D-ResNet.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "models/checkpoints",
        superclasses: Optional[List[str]] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.superclasses = superclasses or DEFAULT_SUPERCLASSES

        self.best_macro_auroc = 0.0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_auroc": [],
            "val_macro_auprc": [],
            "per_class_auroc": []
        }

    def train_epoch(self, epoch: int) -> float:
        """Trains the model for a single epoch."""
        self.model.train()
        running_loss = 0.0
        total_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            signals = batch["signal"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(signals)
            loss = self.criterion(logits, labels)

            loss.backward()
            # Gradient clipping to prevent exploding gradients on noisy ECG signals
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / max(total_batches, 1)
        return epoch_loss

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the model on validation or test set.
        Computes multi-label Macro AUROC, Macro AUPRC, and per-class metrics.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in data_loader:
            signals = batch["signal"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            logits = self.model(signals)
            loss = self.criterion(logits, labels)
            running_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

        val_loss = running_loss / max(len(data_loader), 1)
        y_true = np.vstack(all_targets)
        y_pred = np.vstack(all_preds)

        metrics = self.compute_clinical_metrics(y_true, y_pred)
        return val_loss, metrics

    def compute_clinical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates macro and per-class AUROC and AUPRC scores."""
        per_class_auroc = {}
        per_class_auprc = {}

        valid_aurocs = []
        valid_auprcs = []

        for idx, cls_name in enumerate(self.superclasses):
            targets = y_true[:, idx]
            preds = y_pred[:, idx]

            # Only calculate if both positive and negative cases exist in evaluation set
            if len(np.unique(targets)) > 1:
                try:
                    auc = roc_auc_score(targets, preds)
                    ap = average_precision_score(targets, preds)
                    per_class_auroc[cls_name] = round(float(auc), 4)
                    per_class_auprc[cls_name] = round(float(ap), 4)
                    valid_aurocs.append(auc)
                    valid_auprcs.append(ap)
                except Exception:
                    per_class_auroc[cls_name] = 0.5
                    per_class_auprc[cls_name] = float(np.mean(targets))
            else:
                per_class_auroc[cls_name] = 0.5
                per_class_auprc[cls_name] = 0.0

        macro_auroc = round(float(np.mean(valid_aurocs)) if valid_aurocs else 0.5, 4)
        macro_auprc = round(float(np.mean(valid_auprcs)) if valid_auprcs else 0.0, 4)

        return {
            "macro_auroc": macro_auroc,
            "macro_auprc": macro_auprc,
            "per_class_auroc": per_class_auroc,
            "per_class_auprc": per_class_auprc
        }

    def fit(self, epochs: int = 50) -> Dict:
        """Runs the complete training and validation cycle."""
        logger.info(f"Starting CardioAI 1D-ResNet training on: {self.device} for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.evaluate(self.val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            macro_auroc = val_metrics["macro_auroc"]
            macro_auprc = val_metrics["macro_auprc"]

            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_macro_auroc"].append(macro_auroc)
            self.history["val_macro_auprc"].append(macro_auprc)
            self.history["per_class_auroc"].append(val_metrics["per_class_auroc"])

            logger.info(
                f"Epoch [{epoch:>2}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUROC: {macro_auroc:.4f} | "
                f"Val AUPRC: {macro_auprc:.4f}"
            )

            # Checkpoint best model based on Macro AUROC
            if macro_auroc > self.best_macro_auroc:
                self.best_macro_auroc = macro_auroc
                self.save_checkpoint(epoch, is_best=True)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.2f} minutes. Best Macro AUROC: {self.best_macro_auroc:.4f}")

        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves model weights and training state."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_macro_auroc": self.best_macro_auroc,
            "superclasses": self.superclasses
        }
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, path)
            logger.info(f" Saved new best model checkpoint to {path} (AUROC: {self.best_macro_auroc:.4f})")
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, path)
