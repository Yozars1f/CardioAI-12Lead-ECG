"""
CardioAI_12Lead_ECG: PTB-XL Clinical Preprocessor & Semantic Class Mapper.
Standardized 5-class multi-label pipeline (NORM, MI, STTC, CD, HYP) with Zero Data Leakage.
Author: Youssef Ahmad, MD (Cardiovascular AI Track)
"""

import os
import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("CardioAI_Preprocessor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] CardioAI-Preproc: %(message)s"
)

# Standard diagnostic superclasses defined by PhysioNet PTB-XL benchmark
DEFAULT_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


class PTBXLPreprocessor:
    """
    Handles semantic diagnostic label mapping, patient-level stratified splitting,
    zero-data-leakage verification, and class distribution profiling for PTB-XL.
    """

    def __init__(
        self,
        raw_data_dir: str = "data/raw/ptb-xl",
        processed_data_dir: str = "data/processed",
        superclasses: Optional[List[str]] = None,
        train_folds: Optional[List[int]] = None,
        val_folds: Optional[List[int]] = None,
        test_folds: Optional[List[int]] = None,
        filter_unlabeled: bool = True,
    ):
        self.raw_dir = Path(raw_data_dir).resolve()
        self.processed_dir = Path(processed_data_dir).resolve()
        self.superclasses = superclasses or DEFAULT_SUPERCLASSES
        self.num_classes = len(self.superclasses)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.superclasses)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.superclasses)}

        # Default PhysioNet stratified fold partitioning: 1-8 Train, 9 Val, 10 Test
        self.train_folds = train_folds or [1, 2, 3, 4, 5, 6, 7, 8]
        self.val_folds = val_folds or [9]
        self.test_folds = test_folds or [10]
        self.filter_unlabeled = filter_unlabeled

        self.scp_map: Dict[str, str] = {}
        self.db_df: Optional[pd.DataFrame] = None

    def load_and_build_scp_mapping(self) -> Dict[str, str]:
        """
        Loads scp_statements.csv and creates a dictionary mapping each SCP diagnostic statement
        to its parent diagnostic superclass (e.g., 'AMI' -> 'MI', 'CLBBB' -> 'CD').
        """
        scp_path = self.raw_dir / "scp_statements.csv"
        if not scp_path.exists():
            raise FileNotFoundError(f"scp_statements.csv not found at {scp_path}")

        scp_df = pd.read_csv(scp_path, index_col=0)
        # Filter for statements that have diagnostic relevance (diagnostic == 1)
        diag_statements = scp_df[scp_df["diagnostic"] == 1]

        mapping = {}
        for code, row in diag_statements.iterrows():
            diag_class = row.get("diagnostic_class")
            if pd.notna(diag_class) and diag_class in self.superclasses:
                mapping[str(code)] = str(diag_class)

        logger.info(f"Loaded {len(mapping)} diagnostic SCP statement mappings to superclasses {self.superclasses}")
        self.scp_map = mapping
        return mapping

    def extract_superclasses_from_scp(self, scp_codes_str: str) -> List[str]:
        """
        Parses scp_codes dictionary string and returns list of unique diagnostic superclasses.
        """
        if not self.scp_map:
            self.load_and_build_scp_mapping()

        clean_str = str(scp_codes_str)
        if "np.str_" in clean_str:
            clean_str = clean_str.replace("np.str_(", "").replace(")", "")

        try:
            if isinstance(scp_codes_str, dict):
                code_dict = scp_codes_str
            else:
                code_dict = ast.literal_eval(clean_str)
        except (ValueError, SyntaxError):
            return []

        matched_classes = set()
        for code, likelihood in code_dict.items():
            if likelihood is not None and float(likelihood) >= 0:
                if code in self.scp_map:
                    matched_classes.add(self.scp_map[code])

        return sorted(list(matched_classes))

    def process_metadata(self) -> pd.DataFrame:
        """
        Reads ptbxl_database.csv, applies diagnostic mapping, creates multi-hot binary vectors,
        and adds structured columns.
        """
        db_path = self.raw_dir / "ptbxl_database.csv"
        if not db_path.exists():
            raise FileNotFoundError(f"ptbxl_database.csv not found at {db_path}")

        logger.info(f"Loading raw metadata from {db_path}...")
        df = pd.read_csv(db_path)
        logger.info(f"Initial raw ECG records count: {len(df):,}")

        # Build SCP dictionary
        if not self.scp_map:
            self.load_and_build_scp_mapping()

        # Extract superclasses for each record
        df["diagnostic_superclass"] = df["scp_codes"].apply(self.extract_superclasses_from_scp)
        df["num_superclasses"] = df["diagnostic_superclass"].apply(len)

        # Create binary multi-hot columns for each superclass
        for cls_name in self.superclasses:
            df[f"label_{cls_name}"] = df["diagnostic_superclass"].apply(
                lambda classes, c=cls_name: 1 if c in classes else 0
            )

        if self.filter_unlabeled:
            before_count = len(df)
            df = df[df["num_superclasses"] > 0].copy()
            logger.info(f"Filtered records without standard diagnostic superclasses: {before_count} -> {len(df):,}")

        self.db_df = df
        return df

    def verify_zero_data_leakage(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> bool:
        """
        Strictly verifies that no patient appears across multiple splits (zero patient leakage).
        Critical requirement for rigorous academic & FDA clinical AI standards.
        """
        train_patients = set(train_df["patient_id"].dropna().unique())
        val_patients = set(val_df["patient_id"].dropna().unique())
        test_patients = set(test_df["patient_id"].dropna().unique())

        leak_train_val = train_patients.intersection(val_patients)
        leak_train_test = train_patients.intersection(test_patients)
        leak_val_test = val_patients.intersection(test_patients)

        if leak_train_val:
            raise ValueError(f"CRITICAL LEAKAGE: {len(leak_train_val)} patients overlap between Train & Val!")
        if leak_train_test:
            raise ValueError(f"CRITICAL LEAKAGE: {len(leak_train_test)} patients overlap between Train & Test!")
        if leak_val_test:
            raise ValueError(f"CRITICAL LEAKAGE: {len(leak_val_test)} patients overlap between Val & Test!")

        logger.info("ZERO DATA LEAKAGE CONFIRMED: All splits are strictly isolated at the patient level.")
        return True

    def split_and_save(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs stratified fold splitting, runs zero leakage verification, calculates class
        imbalance statistics and positive weights for Asymmetric Focal Loss, and saves files.
        """
        if self.db_df is None:
            self.process_metadata()

        df = self.db_df
        train_df = df[df["strat_fold"].isin(self.train_folds)].copy()
        val_df = df[df["strat_fold"].isin(self.val_folds)].copy()
        test_df = df[df["strat_fold"].isin(self.test_folds)].copy()

        # Enforce zero patient leakage
        self.verify_zero_data_leakage(train_df, val_df, test_df)

        # Ensure output directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Save partitioned CSVs
        train_path = self.processed_dir / "train_metadata.csv"
        val_path = self.processed_dir / "val_metadata.csv"
        test_path = self.processed_dir / "test_metadata.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Saved processed metadata partitions to {self.processed_dir}")

        # Compute summary statistics & class weights
        stats = self._compute_summary_statistics(train_df, val_df, test_df)
        with open(self.processed_dir / "class_summary.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

        # Compute label co-occurrence matrix (multi-label clinical correlation)
        co_occurrence = self._compute_co_occurrence(train_df)
        co_occurrence.to_csv(self.processed_dir / "co_occurrence_matrix.csv")

        logger.info("Preprocessing and statistical analysis completed successfully!")
        return train_df, val_df, test_df

    def _compute_summary_statistics(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict:
        """Computes sample counts, positive class prevalence, and positive focal weights."""
        label_cols = [f"label_{c}" for c in self.superclasses]
        total_train = len(train_df)
        pos_counts_train = train_df[label_cols].sum().to_dict()

        # Asymmetric Focal Loss / Weighted BCE class weight recommendation:
        # w_pos = (Total - Pos) / Pos
        pos_weights = {}
        prevalence = {}
        for c in self.superclasses:
            col = f"label_{c}"
            pos_c = int(pos_counts_train[col])
            prevalence[c] = round(pos_c / total_train, 4)
            pos_weights[c] = round((total_train - pos_c) / max(pos_c, 1), 4)

        stats = {
            "num_superclasses": self.num_classes,
            "superclasses": self.superclasses,
            "counts": {
                "train_records": len(train_df),
                "val_records": len(val_df),
                "test_records": len(test_df),
                "total_records": len(train_df) + len(val_df) + len(test_df),
            },
            "patients": {
                "train_unique_patients": int(train_df["patient_id"].nunique()),
                "val_unique_patients": int(val_df["patient_id"].nunique()),
                "test_unique_patients": int(test_df["patient_id"].nunique()),
            },
            "train_prevalence": prevalence,
            "recommended_pos_weights": pos_weights,
            "multi_label_stats": {
                "train_single_label_percent": round(
                    float((train_df["num_superclasses"] == 1).mean() * 100), 2
                ),
                "train_multi_label_percent": round(
                    float((train_df["num_superclasses"] > 1).mean() * 100), 2
                ),
            },
        }
        return stats

    def _compute_co_occurrence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes multi-label co-occurrence matrix across the 5 superclasses."""
        label_cols = [f"label_{c}" for c in self.superclasses]
        matrix = pd.DataFrame(0, index=self.superclasses, columns=self.superclasses)
        for i, c1 in enumerate(self.superclasses):
            for j, c2 in enumerate(self.superclasses):
                col1 = f"label_{c1}"
                col2 = f"label_{c2}"
                co_occur = int(((df[col1] == 1) & (df[col2] == 1)).sum())
                matrix.loc[c1, c2] = co_occur
        return matrix


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PTB-XL Preprocessor")
    parser.add_argument("--raw-dir", type=str, default="data/raw/ptb-xl")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    args = parser.parse_args()

    preproc = PTBXLPreprocessor(raw_data_dir=args.raw_dir, processed_data_dir=args.processed_dir)
    train_df, val_df, test_df = preproc.split_and_save()
    print("Train records:", len(train_df))
    print("Val records:", len(val_df))
    print("Test records:", len(test_df))
