# utils/dataset_utils.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_from_disk

logger = logging.getLogger("processed_dataset_loader")


class ProcessedDatasetLoader:
    """
    Handles processed dataset paths and validation.

    Expected structure:

    data/processed/
        train_dataset/
        val_dataset/
        test_dataset/

    OR

    data/processed/
        iqa_train.jsonl
        iqa_val.jsonl
        iqa_test.jsonl
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

        # HF save_to_disk directories
        self.train_dir = self.data_dir / "train_dataset"
        self.val_dir = self.data_dir / "val_dataset"
        self.test_dir = self.data_dir / "test_dataset"

        # JSONL files
        self.train_jsonl = self.data_dir / "iqa_train.jsonl"
        self.val_jsonl = self.data_dir / "iqa_val.jsonl"
        self.test_jsonl = self.data_dir / "iqa_test.jsonl"

    # -------------------------------------------------
    # HF save_to_disk
    # -------------------------------------------------

    def load_train(self) -> Dataset:
        return load_from_disk(str(self.train_dir))

    def load_val(self) -> Dataset:
        return load_from_disk(str(self.val_dir))

    def load_test(self) -> Dataset:
        return load_from_disk(str(self.test_dir))

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    def validate_hf(self) -> bool:
        ok = True

        if not self.train_dir.exists():
            logger.error(f"Missing train dataset dir: {self.train_dir}")
            ok = False
        if not self.val_dir.exists():
            logger.error(f"Missing val dataset dir: {self.val_dir}")
            ok = False
        if not self.test_dir.exists():
            logger.warning(f"Missing test dataset dir: {self.test_dir}")

        return ok

    def validate_jsonl(self) -> bool:
        ok = True

        if not self.train_jsonl.exists():
            logger.error(f"Missing train JSONL: {self.train_jsonl}")
            ok = False
        if not self.val_jsonl.exists():
            logger.error(f"Missing val JSONL: {self.val_jsonl}")
            ok = False
        if not self.test_jsonl.exists():
            logger.warning(f"Missing test JSONL: {self.test_jsonl}")

        return ok

    # -------------------------------------------------
    # Column validation
    # -------------------------------------------------

    @staticmethod
    def require_columns(ds: Dataset, required: list[str], name: str = "dataset") -> None:
        cols = set(ds.column_names)
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(
                f"{name} missing required columns: {missing}. "
                f"Found columns: {sorted(cols)}"
            )
