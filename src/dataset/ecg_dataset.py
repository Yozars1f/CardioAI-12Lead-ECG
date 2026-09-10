"""
CardioAI_12Lead_ECG: PyTorch Dataset & DataLoader for 12-Lead ECG.
Optimized for 1D-ResNet (Shape: [Batch, 12 Leads, 1000 Timesteps]) with Lead-wise Z-Score Normalization.
Author: Youssef Ahmad, MD (Cardiovascular AI Track)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# PhysioNet WFDB reader
try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

# Standard 12-Lead ECG Order (AHA/ACC Standard)
STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEFAULT_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


class ECGSignalAugmenter:
    """
    Physiologically plausible ECG signal augmentations for 1D-ResNet training:
    1. Baseline wander (low frequency drift simulating patient respiration).
    2. Random Gaussian noise (simulating EMG / electrode tremor).
    3. Random Lead Dropout (simulating temporary loose electrode connection).
    """

    def __init__(
        self,
        noise_std: float = 0.02,
        baseline_wander_prob: float = 0.3,
        lead_dropout_prob: float = 0.1,
    ):
        self.noise_std = noise_std
        self.baseline_wander_prob = baseline_wander_prob
        self.lead_dropout_prob = lead_dropout_prob

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        # signal shape: (12, 1000)
        augmented = signal.copy()

        # 1. Additive Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=augmented.shape).astype(np.float32)
            augmented += noise

        # 2. Baseline wander (sine wave with frequency 0.1 - 0.5 Hz)
        if np.random.random() < self.baseline_wander_prob:
            time = np.linspace(0, 10, augmented.shape[1])
            drift_freq = np.random.uniform(0.15, 0.4)
            drift_amp = np.random.uniform(0.05, 0.2)
            drift = drift_amp * np.sin(2 * np.pi * drift_freq * time)
            augmented += drift

        # 3. Random Lead Dropout
        if np.random.random() < self.lead_dropout_prob:
            drop_idx = np.random.randint(0, augmented.shape[0])
            augmented[drop_idx, :] = 0.0

        return augmented.astype(np.float32)


class PTBXLECGDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL 12-Lead ECG Records.
    
    Returns:
        dict:
            - 'signal': torch.FloatTensor of shape (12, 1000)
            - 'label': torch.FloatTensor of shape (5,) [NORM, MI, STTC, CD, HYP]
            - 'ecg_id': int
            - 'patient_id': int
    """

    def __init__(
        self,
        metadata: Union[str, Path, pd.DataFrame],
        raw_data_dir: Union[str, Path],
        superclasses: Optional[List[str]] = None,
        target_length: int = 1000,
        normalize: bool = True,
        eps: float = 1e-7,
        augment: bool = False,
        augmenter: Optional[ECGSignalAugmenter] = None,
    ):
        self.raw_dir = Path(raw_data_dir).resolve()
        self.superclasses = superclasses or DEFAULT_SUPERCLASSES
        self.target_length = target_length
        self.normalize = normalize
        self.eps = eps
        self.augment = augment
        self.augmenter = augmenter or ECGSignalAugmenter() if augment else None

        if isinstance(metadata, (str, Path)):
            self.df = pd.read_csv(metadata)
        elif isinstance(metadata, pd.DataFrame):
            self.df = metadata.copy().reset_index(drop=True)
        else:
            raise TypeError("metadata must be a file path or pd.DataFrame")

        self.label_cols = [f"label_{c}" for c in self.superclasses]
        for col in self.label_cols:
            if col not in self.df.columns:
                raise KeyError(f"Required label column {col} missing from metadata!")

    def __len__(self) -> int:
        return len(self.df)

    def _load_signal(self, filename_lr: str) -> np.ndarray:
        """
        Loads the 12-lead ECG signal.
        Supports both WFDB (.dat/.hea) and raw numpy fallback (.npy).
        """
        rec_path = self.raw_dir / filename_lr
        
        # Check if stored as numpy (.npy)
        npy_path = Path(str(rec_path) + ".npy")
        if npy_path.exists():
            signal = np.load(npy_path)
            return signal

        # Load with wfdb
        if HAS_WFDB:
            # strip possible extension if present
            base_str = str(rec_path).replace(".hea", "").replace(".dat", "")
            try:
                record = wfdb.rdsamp(base_str)
                signal = record[0]  # Shape: (1000, 12)
                return signal
            except Exception as e:
                # If path relative to raw_dir failed, try searching inside records100
                alt_path = self.raw_dir / "records100" / Path(filename_lr).name
                if alt_path.with_suffix(".hea").exists():
                    record = wfdb.rdsamp(str(alt_path))
                    return record[0]
                raise FileNotFoundError(f"WFDB failed to load record at {base_str}: {e}")
        else:
            raise ImportError("wfdb library is required to read PhysioNet ECG records.")

    def _leadwise_zscore(self, signal: np.ndarray) -> np.ndarray:
        """
        Computes Lead-wise Z-Score normalization:
        z = (signal - mean) / (std + eps)
        Operates on shape (12, 1000).
        """
        mean = np.mean(signal, axis=1, keepdims=True)
        std = np.std(signal, axis=1, keepdims=True)
        return (signal - mean) / (std + self.eps)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        row = self.df.iloc[idx]
        filename_lr = row["filename_lr"]

        # 1. Load raw signal -> shape: (1000, 12)
        raw_signal = self._load_signal(filename_lr)

        # 2. Reshape to channel-first format for 1D-ResNet -> shape: (12, 1000)
        if raw_signal.shape == (self.target_length, 12):
            signal = raw_signal.T
        elif raw_signal.shape == (12, self.target_length):
            signal = raw_signal
        else:
            # Handle potential length mismatches with linear interpolation/padding
            if raw_signal.shape[1] == 12:
                signal = raw_signal.T
            else:
                signal = raw_signal
            if signal.shape[1] != self.target_length:
                # Truncate or pad
                padded = np.zeros((12, self.target_length), dtype=np.float32)
                min_len = min(signal.shape[1], self.target_length)
                padded[:, :min_len] = signal[:, :min_len]
                signal = padded

        signal = signal.astype(np.float32)

        # 3. Apply physiological augmentation (if training)
        if self.augment and self.augmenter:
            signal = self.augmenter(signal)

        # 4. Lead-wise Z-Score Normalization
        if self.normalize:
            signal = self._leadwise_zscore(signal)

        # 5. Extract Multi-hot Label Vector
        labels = row[self.label_cols].values.astype(np.float32)

        # Convert to PyTorch tensors
        signal_tensor = torch.from_numpy(signal).float()        # Shape: (12, 1000)
        label_tensor = torch.from_numpy(labels).float()          # Shape: (5,)
        ecg_id = int(row.get("ecg_id", idx))
        patient_id = int(row.get("patient_id", -1))

        return {
            "signal": signal_tensor,
            "label": label_tensor,
            "ecg_id": ecg_id,
            "patient_id": patient_id
        }


def get_dataloaders(
    processed_dir: Union[str, Path] = "data/processed",
    raw_dir: Union[str, Path] = "data/raw/ptb-xl",
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoaders for Train, Validation, and Test splits.
    Automatically handles Windows multiprocessing compatibility.
    """
    proc_path = Path(processed_dir).resolve()
    raw_path = Path(raw_dir).resolve()

    train_meta = proc_path / "train_metadata.csv"
    val_meta = proc_path / "val_metadata.csv"
    test_meta = proc_path / "test_metadata.csv"

    for p in [train_meta, val_meta, test_meta]:
        if not p.exists():
            raise FileNotFoundError(f"Processed metadata not found: {p}. Run PTBXLPreprocessor first.")

    # Datasets
    train_dataset = PTBXLECGDataset(
        metadata=train_meta,
        raw_data_dir=raw_path,
        augment=augment_train,
        normalize=True
    )
    val_dataset = PTBXLECGDataset(
        metadata=val_meta,
        raw_data_dir=raw_path,
        augment=False,
        normalize=True
    )
    test_dataset = PTBXLECGDataset(
        metadata=test_meta,
        raw_data_dir=raw_path,
        augment=False,
        normalize=True
    )

    # Windows safe workers check
    if os.name == "nt" and num_workers > 0:
        actual_workers = 0  # Avoid spawn issues in interactive/script runs on Windows
    else:
        actual_workers = num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
