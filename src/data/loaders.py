# src/data/loaders.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from datasets import Dataset, load_from_disk, load_dataset

logger = logging.getLogger("dataset_loader")


class DatasetLoader:
    """
    Unified dataset loader.

    Supports:
      - HF save_to_disk datasets:
            data_dir/
                train_dataset/
                val_dataset/

      - JSONL datasets:
            data_dir/
                iqa_train.jsonl
                iqa_val.jsonl

    Base dataset must contain:
      - image_path
      - mos_score
    """

    def __init__(self, data_dir: str | Path, use_jsonl: bool = False):
        self.data_dir = Path(data_dir)
        self.use_jsonl = use_jsonl

        # HF save_to_disk paths
        self.train_disk = self.data_dir / "train_dataset"
        self.val_disk = self.data_dir / "val_dataset"

        # JSONL paths
        self.train_jsonl = self.data_dir / "iqa_train.jsonl"
        self.val_jsonl = self.data_dir / "iqa_val.jsonl"

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def load_train_val(self) -> Tuple[Dataset, Dataset]:
        if self.use_jsonl:
            return self._load_jsonl()
        return self._load_disk()

    # --------------------------------------------------
    # Internal
    # --------------------------------------------------

    def _load_disk(self) -> Tuple[Dataset, Dataset]:
        if not self.train_disk.exists():
            raise FileNotFoundError(f"Missing HF dataset: {self.train_disk}")
        if not self.val_disk.exists():
            raise FileNotFoundError(f"Missing HF dataset: {self.val_disk}")

        logger.info("Loading HF save_to_disk datasets...")
        train = load_from_disk(str(self.train_disk))
        val = load_from_disk(str(self.val_disk))

        logger.info(f"Loaded HF datasets | train={len(train)} val={len(val)}")
        return train, val

    def _load_jsonl(self) -> Tuple[Dataset, Dataset]:
        if not self.train_jsonl.exists():
            raise FileNotFoundError(f"Missing JSONL: {self.train_jsonl}")
        if not self.val_jsonl.exists():
            raise FileNotFoundError(f"Missing JSONL: {self.val_jsonl}")

        logger.info("Loading JSONL datasets...")
        train = load_dataset("json", data_files=str(self.train_jsonl))["train"]
        val = load_dataset("json", data_files=str(self.val_jsonl))["train"]

        logger.info(f"Loaded JSONL datasets | train={len(train)} val={len(val)}")
        return train, val

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    @staticmethod
    def require_columns(ds: Dataset, required: list[str], name: str = "dataset") -> None:
        missing = [c for c in required if c not in ds.column_names]
        if missing:
            raise ValueError(
                f"{name} missing required columns: {missing}. "
                f"Found columns: {ds.column_names}"
            )
