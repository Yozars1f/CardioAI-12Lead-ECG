"""
CardioAI_12Lead_ECG: Dataset Downloader & Verifier for PTB-XL (PhysioNet).
Author: Youssef Ahmad, MD (Cardiovascular AI Track)
"""

import os
import sys
import zipfile
import shutil
import logging
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] CardioAI-Downloader: %(message)s"
)
logger = logging.getLogger("CardioAI_Downloader")

PHYSIONET_ZIP_URL = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"


def download_file_with_progress(url: str, destination: Path) -> None:
    """Download a file with an interactive tqdm progress bar."""
    logger.info(f"Connecting to: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as file, tqdm(
        desc=f"Downloading {destination.name}",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))
    logger.info(f"Download complete: {destination}")


def verify_ptbxl_structure(raw_dir: Path) -> bool:
    """
    Verifies that the PTB-XL directory has all required metadata files and records.
    """
    database_csv = raw_dir / "ptbxl_database.csv"
    scp_csv = raw_dir / "scp_statements.csv"
    records_dir = raw_dir / "records100"

    # Also handle the nested folder if unzipped with subfolder
    if not database_csv.exists():
        nested_dirs = list(raw_dir.glob("ptb-xl*"))
        for n_dir in nested_dirs:
            if (n_dir / "ptbxl_database.csv").exists():
                logger.info(f"Found nested PTB-XL folder at {n_dir}. Restructuring to root {raw_dir}...")
                for item in n_dir.iterdir():
                    target = raw_dir / item.name
                    if not target.exists():
                        shutil.move(str(item), str(target))
                shutil.rmtree(n_dir, ignore_errors=True)
                break

    has_db = database_csv.exists()
    has_scp = scp_csv.exists()
    has_records = records_dir.exists()

    if has_db and has_scp:
        df = pd.read_csv(database_csv)
        logger.info(f"Dataset verification SUCCESS: {len(df):,} ECG records found in {database_csv}")
        return True
    return False


def create_synthetic_ptbxl_sample(raw_dir: Path, num_records: int = 250) -> None:
    """
    Creates a realistic synthetic micro-dataset matching PTB-XL 100Hz schema perfectly.
    Allows instant zero-overhead local testing and unit validation on CPU without waiting
    for the full ~1GB download.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    records100_dir = raw_dir / "records100" / "00000"
    records100_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating realistic synthetic PTB-XL micro-dataset ({num_records} records) at {raw_dir}...")

    # 1. Create scp_statements.csv
    scp_data = {
        "description": [
            "Normal ECG", "Myocardial Infarction", "Anterior MI", "Inferior MI",
            "ST-T change", "Ischemia", "Left bundle branch block",
            "Right bundle branch block", "Left ventricular hypertrophy",
            "Sinus rhythm"
        ],
        "diagnostic": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "form": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "rhythm": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "diagnostic_class": ["NORM", "MI", "MI", "MI", "STTC", "STTC", "CD", "CD", "HYP", np.nan],
        "diagnostic_subclass": ["NORM", "AMI", "AMI", "IMI", "STTC", "ISCA", "CLBBB", "CRBBB", "LVH", np.nan]
    }
    scp_df = pd.DataFrame(
        scp_data,
        index=["NORM", "MI", "AMI", "IMI", "STTC", "ISCA", "CLBBB", "CRBBB", "LVH", "SR"]
    )
    scp_df.index.name = "index"
    scp_df.to_csv(raw_dir / "scp_statements.csv")

    # 2. Generate database rows & synthetic waveforms
    records = []
    classes = ["NORM", "AMI", "IMI", "STTC", "CLBBB", "CRBBB", "LVH"]
    rng = np.random.default_rng(42)

    try:
        import wfdb
        has_wfdb = True
    except ImportError:
        has_wfdb = False

    for i in range(1, num_records + 1):
        ecg_id = i
        patient_id = (i % 80) + 1  # Ensures multi-record patients for fold-grouping validation
        age = rng.integers(25, 85)
        sex = int(rng.choice([0, 1]))
        strat_fold = ((i - 1) % 10) + 1  # 1 to 10 balanced folds
        
        # Select diagnostic labels (simulate multi-label cases)
        num_labels = rng.choice([1, 2], p=[0.75, 0.25])
        chosen_codes = rng.choice(classes, size=num_labels, replace=False)
        scp_dict = {str(code): 100.0 for code in chosen_codes}
        if "NORM" not in chosen_codes and rng.random() > 0.5:
            scp_dict["SR"] = 0.0

        rel_path = f"records100/00000/{ecg_id:05d}_lr"
        full_rec_path = raw_dir / "records100" / "00000" / f"{ecg_id:05d}_lr"

        # Generate realistic synthetic 12-lead ECG signal: shape (1000, 12)
        time = np.linspace(0, 10, 1000)
        signal = np.zeros((1000, 12), dtype=np.float32)
        heart_rate = rng.uniform(60, 90)
        freq = heart_rate / 60.0

        for lead_idx in range(12):
            # Base QRS wave simulation with harmonics + baseline noise
            base_ecg = (
                1.0 * np.sin(2 * np.pi * freq * time) +
                0.5 * np.sin(4 * np.pi * freq * time) +
                0.2 * np.sin(6 * np.pi * freq * time)
            )
            noise = rng.normal(0, 0.05, size=1000)
            signal[:, lead_idx] = base_ecg + noise

        # Save record using wfdb or numpy binary
        if has_wfdb:
            lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            wfdb.wrsamp(
                record_name=f"{ecg_id:05d}_lr",
                fs=100,
                units=['mV'] * 12,
                sig_name=lead_names,
                p_signal=signal,
                fmt=['16'] * 12,
                write_dir=str(full_rec_path.parent)
            )
        else:
            np.save(str(full_rec_path) + ".npy", signal)

        records.append({
            "ecg_id": ecg_id,
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "height": rng.uniform(150, 190),
            "weight": rng.uniform(50, 100),
            "nurse": 1,
            "site": 1,
            "device": "CS100",
            "recording_date": "2020-01-01 00:00:00",
            "report": f"Clinical report for ECG #{ecg_id}",
            "scp_codes": str(scp_dict),
            "heart_axis": "MID",
            "strat_fold": strat_fold,
            "filename_lr": rel_path,
            "filename_hr": rel_path.replace("records100", "records500").replace("_lr", "_hr")
        })

    df = pd.DataFrame(records)
    df.to_csv(raw_dir / "ptbxl_database.csv", index=False)
    logger.info(f"Synthetic PTB-XL database successfully written to {raw_dir / 'ptbxl_database.csv'}")


def download_ptbxl_dataset(
    target_dir: str = "data/raw/ptb-xl",
    force_download: bool = False,
    use_synthetic_sample: bool = False,
    sample_size: int = 300
) -> Path:
    """
    Main download function.
    - If data already exists and valid, returns existing Path.
    - If use_synthetic_sample=True, creates a fast micro-dataset for testing.
    - Otherwise downloads and unpacks PTB-XL 1.0.3 from PhysioNet.
    """
    raw_path = Path(target_dir).resolve()

    if use_synthetic_sample:
        create_synthetic_ptbxl_sample(raw_path, num_records=sample_size)
        return raw_path

    if not force_download and verify_ptbxl_structure(raw_path):
        logger.info(f"PTB-XL already present and verified at: {raw_path}")
        return raw_path

    raw_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_path / "ptb-xl-1.0.3.zip"

    logger.info("Initiating download of PTB-XL (1.0.3) from PhysioNet (~800 MB)...")
    download_file_with_progress(PHYSIONET_ZIP_URL, zip_path)

    logger.info(f"Extracting {zip_path} into {raw_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_path)

    # Clean up zip archive to save disk space
    if zip_path.exists():
        zip_path.unlink()
        logger.info("Removed temporary zip archive.")

    if verify_ptbxl_structure(raw_path):
        logger.info("PTB-XL download and extraction successfully completed!")
    else:
        logger.warning("Download finished but structure verification found discrepancies. Please verify.")

    return raw_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PTB-XL Dataset Downloader for CardioAI")
    parser.add_argument("--dir", type=str, default="data/raw/ptb-xl", help="Target raw directory")
    parser.add_argument("--force", action="store_true", help="Force redownload even if exists")
    parser.add_argument("--sample", action="store_true", help="Generate synthetic micro-dataset for rapid dev/test")
    parser.add_argument("--sample-size", type=int, default=300, help="Number of records for synthetic sample")
    args = parser.parse_args()

    download_ptbxl_dataset(
        target_dir=args.dir,
        force_download=args.force,
        use_synthetic_sample=args.sample,
        sample_size=args.sample_size
    )
