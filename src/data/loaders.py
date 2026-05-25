from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple
from datasets import Dataset, load_from_disk, load_dataset

logger = logging.getLogger("dataset_loader")


class DatasetLoader:
    """
    Unified dataset loader.

    Supports:
    - HF/Arrow save_to_disk datasets:
        data_dir/
            train_dataset/
            val_dataset/
            test_dataset/
    - JSON/JSONL datasets:
        data_dir/
            base_train.jsonl
            base_val.jsonl
            base_test.jsonl

    Base dataset must contain:
    - image_path
    - mos_score
    """

    def __init__(
        self,
        data_dir: str | Path,
        use_jsonl: bool = False,
        dataset_format: Optional[str] = None,
        train_dataset_dir: Optional[str | Path] = None,
        val_dataset_dir: Optional[str | Path] = None,
        test_dataset_dir: Optional[str | Path] = None,
        train_json_path: Optional[str | Path] = None,
        val_json_path: Optional[str | Path] = None,
        test_json_path: Optional[str | Path] = None,
    ):
        """Store dataset locations and resolve the requested on-disk format."""
        self.data_dir = Path(data_dir)
        self.use_jsonl = use_jsonl
        self.dataset_format = self._normalize_dataset_format(dataset_format, use_jsonl)

        # HF save_to_disk paths
        self.train_disk = Path(train_dataset_dir) if train_dataset_dir else self.data_dir / "train_dataset"
        self.val_disk = Path(val_dataset_dir) if val_dataset_dir else self.data_dir / "val_dataset"
        self.test_disk = Path(test_dataset_dir) if test_dataset_dir else self.data_dir / "test_dataset"

        # JSON/JSONL paths
        self.train_json = Path(train_json_path) if train_json_path else self.data_dir / "base_train.jsonl"
        self.val_json = Path(val_json_path) if val_json_path else self.data_dir / "base_val.jsonl"
        self.test_json = Path(test_json_path) if test_json_path else self.data_dir / "base_test.jsonl"

    @staticmethod
    def _normalize_dataset_format(dataset_format: Optional[str], use_jsonl: bool) -> str:
        """
        Normalize dataset format aliases.

        Args:
            dataset_format: Explicit format name from config, if provided.
            use_jsonl: Legacy flag that selects JSON/JSONL loading.

        Returns:
            Either ``"arrow"`` for Hugging Face disk datasets or ``"json"`` for
            JSON/JSONL datasets.

        Raises:
            ValueError: If the requested dataset format is unsupported.
        """
        if dataset_format is None:
            return "json" if use_jsonl else "arrow"

        value = str(dataset_format).strip().lower()
        aliases = {
            "hf": "arrow",
            "hf_disk": "arrow",
            "disk": "arrow",
            "save_to_disk": "arrow",
            "jsonl": "json",
        }
        value = aliases.get(value, value)
        if value not in {"arrow", "json"}:
            raise ValueError("dataset_format must be one of: arrow, json, jsonl")
        return value

    def load_train_val(self) -> Tuple[Dataset, Dataset]:
        """
        Load the configured training and validation datasets.

        Returns:
            A ``(train, validation)`` tuple of Hugging Face ``Dataset`` objects.
        """
        if self.dataset_format == "json":
            return self._load_json_train_val()
        return self._load_disk_train_val()

    def load_test(self) -> Dataset:
        """
        Load the configured test dataset.

        Returns:
            The test split as a Hugging Face ``Dataset``.
        """
        if self.dataset_format == "json":
            return self._load_json_test()
        return self._load_disk_test()

    def _load_disk_train_val(self) -> Tuple[Dataset, Dataset]:
        """
        Load train/validation splits saved with Hugging Face ``save_to_disk``.

        Returns:
            A ``(train, validation)`` tuple loaded from the configured directories.

        Raises:
            FileNotFoundError: If either split directory is missing.
        """
        if not self.train_disk.exists():
            raise FileNotFoundError(f"Missing HF dataset: {self.train_disk}")
        if not self.val_disk.exists():
            raise FileNotFoundError(f"Missing HF dataset: {self.val_disk}")

        logger.info("Loading HF save_to_disk train/val datasets...")
        train = load_from_disk(str(self.train_disk))
        val = load_from_disk(str(self.val_disk))

        logger.info(f"Loaded HF datasets | train={len(train)} val={len(val)}")
        return train, val

    def _load_disk_test(self) -> Dataset:
        """
        Load the test split saved with Hugging Face ``save_to_disk``.

        Returns:
            The loaded test dataset.

        Raises:
            FileNotFoundError: If the test split directory is missing.
        """
        if not self.test_disk.exists():
            raise FileNotFoundError(f"Missing HF dataset: {self.test_disk}")

        logger.info("Loading HF save_to_disk test dataset...")
        test = load_from_disk(str(self.test_disk))

        logger.info(f"Loaded HF dataset | test={len(test)}")
        return test

    def _load_json_train_val(self) -> Tuple[Dataset, Dataset]:
        """
        Load train/validation splits from JSON or JSONL files.

        Returns:
            A ``(train, validation)`` tuple loaded via ``datasets.load_dataset``.

        Raises:
            FileNotFoundError: If either JSON/JSONL split file is missing.
        """
        if not self.train_json.exists():
            raise FileNotFoundError(f"Missing JSON/JSONL: {self.train_json}")
        if not self.val_json.exists():
            raise FileNotFoundError(f"Missing JSON/JSONL: {self.val_json}")

        logger.info("Loading JSON/JSONL train/val datasets...")
        train = load_dataset("json", data_files=str(self.train_json))["train"]
        val = load_dataset("json", data_files=str(self.val_json))["train"]

        logger.info(f"Loaded JSON/JSONL datasets | train={len(train)} val={len(val)}")
        return train, val

    def _load_json_test(self) -> Dataset:
        """
        Load the test split from a JSON or JSONL file.

        Returns:
            The loaded test dataset.

        Raises:
            FileNotFoundError: If the configured test JSON/JSONL file is missing.
        """
        if not self.test_json.exists():
            raise FileNotFoundError(f"Missing JSON/JSONL: {self.test_json}")

        logger.info("Loading JSON/JSONL test dataset...")
        test = load_dataset("json", data_files=str(self.test_json))["train"]

        logger.info(f"Loaded JSON/JSONL dataset | test={len(test)}")
        return test

    @staticmethod
    def require_columns(ds: Dataset, required: list[str], name: str = "dataset") -> None:
        """
        Validate that a dataset contains required columns.

        Args:
            ds: Dataset to inspect.
            required: Column names that must be present.
            name: Human-readable dataset name used in error messages.

        Raises:
            ValueError: If any required column is missing.
        """
        missing = [c for c in required if c not in ds.column_names]
        if missing:
            raise ValueError(
                f"{name} missing required columns: {missing}. "
                f"Found columns: {ds.column_names}"
            )
