"""
Dataset utilities for LDCT IQA VLM pipeline.

Works with:
- HF datasets produced by step2_build_trl_dataset.py
- dataset_stats.json written by Step 2
- both data_io_mode = "png_paths" and "tif_direct"
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

from datasets import Dataset, load_from_disk  # requires `datasets` installed
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------- Data structures ----------------------------

@dataclass
class SplitPaths:
    hf_dir: Path
    jsonl: Path


# ---------------------------- Loader class ----------------------------

class ProcessedDatasetLoader:
    """
    Helper for loading/inspecting processed datasets from `data/processed`.

    Assumes the following layout (from your pipeline):

        data/processed/
            dataset_stats.json
            train/
                _pre_split_train_pool.jsonl
                iqa_sft_train.jsonl
            val/
                iqa_sft_val.jsonl
            test/
                iqa_sft_test.jsonl
            train_dataset/
            val_dataset/
            test_dataset/
    """

    def __init__(self, data_dir: str = "data/processed") -> None:
        self.data_dir = Path(data_dir)

        # HF dataset dirs
        self.train_hf_dir = self.data_dir / "train_dataset"
        self.val_hf_dir   = self.data_dir / "val_dataset"
        self.test_hf_dir  = self.data_dir / "test_dataset"

        # JSONL files created by Step 2
        self.train_jsonl = self.data_dir / "train" / "iqa_sft_train.jsonl"
        self.val_jsonl   = self.data_dir / "val" / "iqa_sft_val.jsonl"
        self.test_jsonl  = self.data_dir / "test" / "iqa_sft_test.jsonl"

        # Stats file
        self.stats_path = self.data_dir / "dataset_stats.json"

    # ------------------------ Stats & basic info ------------------------

    def load_stats(self) -> Dict:
        """Load dataset_stats.json (or {} if missing)."""
        if not self.stats_path.exists():
            logger.warning(f"dataset_stats.json not found at {self.stats_path}")
            return {}
        return json.loads(self.stats_path.read_text(encoding="utf-8"))

    def split_paths(self) -> Dict[str, SplitPaths]:
        """Return paths for each split (HF dir + JSONL)."""
        return {
            "train": SplitPaths(hf_dir=self.train_hf_dir, jsonl=self.train_jsonl),
            "val":   SplitPaths(hf_dir=self.val_hf_dir,   jsonl=self.val_jsonl),
            "test":  SplitPaths(hf_dir=self.test_hf_dir,  jsonl=self.test_jsonl),
        }

    def available_splits(self) -> Dict[str, Dict[str, bool]]:
        """
        Return availability of HF datasets and JSONL files.

        Example:
            {
              "train": {"hf": True, "jsonl": True},
              "val":   {"hf": True, "jsonl": True},
              "test":  {"hf": True, "jsonl": True},
            }
        """
        out: Dict[str, Dict[str, bool]] = {}
        for split, sp in self.split_paths().items():
            out[split] = {
                "hf": sp.hf_dir.exists(),
                "jsonl": sp.jsonl.exists(),
            }
        return out

    # --------------------------- HF loaders ---------------------------

    def _load_hf_split(self, split: str) -> Dataset:
        sp = self.split_paths()[split]
        if not sp.hf_dir.exists():
            raise FileNotFoundError(f"HF dataset for split '{split}' not found at {sp.hf_dir}")
        return load_from_disk(str(sp.hf_dir))

    def load_train(self) -> Dataset:
        return self._load_hf_split("train")

    def load_val(self) -> Dataset:
        return self._load_hf_split("val")

    def load_test(self) -> Dataset:
        return self._load_hf_split("test")

    def load_all(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Convenience: (train, val, test)."""
        return self.load_train(), self.load_val(), self.load_test()

    # --------------------------- Validation ---------------------------

    def _infer_data_io_mode(self, ds: Dataset) -> str:
        """
        Try to infer whether this dataset is:
        - 'png_paths' : has 'image_path'
        - 'tif_direct': has 'images'
        """
        cols = set(ds.column_names)
        if "images" in cols:
            return "tif_direct"
        if "image_path" in cols:
            return "png_paths"
        # Fallback: assume png_paths (most common in your pipeline)
        logger.warning(f"Could not clearly infer data_io_mode from columns {cols}, assuming 'png_paths'")
        return "png_paths"

    def validate(self) -> bool:
        """
        Basic validation:
        - HF dirs exist
        - columns include messages + metadata + (image_path or images)
        - if stats present, sizes match stats
        """
        try:
            available = self.available_splits()
            for split, flags in available.items():
                if not flags["hf"]:
                    logger.error(f"HF dataset for split '{split}' not found at {self.split_paths()[split].hf_dir}")
                    return False

            train_ds = self.load_train()
            val_ds   = self.load_val()
            test_ds  = self.load_test()

            for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
                cols = set(ds.column_names)
                has_messages = "messages" in cols
                has_metadata = "metadata" in cols
                has_image_col = ("images" in cols) or ("image_path" in cols)
                if not (has_messages and has_metadata and has_image_col):
                    logger.error(
                        f"[{name}] Missing required columns. "
                        f"Expected at least messages + metadata + (images|image_path), found {cols}"
                    )
                    return False

            stats = self.load_stats()
            if stats and "dataset_info" in stats:
                expected = {
                    "train": stats["dataset_info"]["train_samples"],
                    "val":   stats["dataset_info"]["val_samples"],
                    "test":  stats["dataset_info"]["test_samples"],
                }
                actual = {
                    "train": len(train_ds),
                    "val":   len(val_ds),
                    "test":  len(test_ds),
                }
                if expected != actual:
                    logger.error(f"Dataset size mismatch. Expected: {expected}, Actual: {actual}")
                    return False

            logger.info("All processed datasets validated successfully.")
            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    # --------------------------- Summary ---------------------------

    def print_summary(self) -> None:
        """Print a textual summary of splits, sizes, and quality distribution."""
        stats = self.load_stats()
        avail = self.available_splits()

        print("=" * 60)
        print("LDCT IQA PROCESSED DATASET SUMMARY")
        print("=" * 60)

        if stats and "dataset_info" in stats:
            di = stats["dataset_info"]
            print(f"Total samples:      {di.get('total_samples', 'N/A')}")
            print(f"Training samples:   {di.get('train_samples', 'N/A')}")
            print(f"Validation samples: {di.get('val_samples', 'N/A')}")
            print(f"Test samples:       {di.get('test_samples', 'N/A')}")
        else:
            print("No dataset_stats.json found or missing dataset_info.")

        print("\nAvailability:")
        for split, flags in avail.items():
            hf_ok   = "✓" if flags["hf"]   else "✗"
            jn_ok   = "✓" if flags["jsonl"] else "✗"
            print(f"  {split:5s}  HF: {hf_ok}  JSONL: {jn_ok}")

        if stats and "quality_distribution" in stats:
            print("\nQuality distribution by split:")
            qd = stats["quality_distribution"]
            for split in ["train", "val", "test"]:
                if split in qd:
                    print(f"  {split.capitalize()}:")
                    total = sum(qd[split].values()) or 1
                    for cat, cnt in qd[split].items():
                        pct = 100.0 * cnt / total
                        print(f"    {cat:10s}: {cnt:4d} ({pct:5.1f}%)")

        print("\nHF dataset dirs:")
        for split, sp in self.split_paths().items():
            print(f"  {split:5s}: {sp.hf_dir}")

        print("\nJSONL files:")
        for split, sp in self.split_paths().items():
            print(f"  {split:5s}: {sp.jsonl}")
        print("")


# ---------------------------- Sample inspection ----------------------------

def inspect_sample(ds: Dataset, index: int = 0, load_image: bool = False) -> None:
    """
    Inspect a single sample from a HF dataset.

    - Prints messages structure.
    - Shows image info:
        - For png_paths: prints image_path, optionally loads and shows size/mode.
        - For tif_direct: directly inspects the PIL image.
    """
    if index < 0 or index >= len(ds):
        print(f"Index {index} is out of range. Dataset length = {len(ds)}")
        return

    sample = ds[index]
    cols = set(ds.column_names)
    io_mode = "tif_direct" if "images" in cols else ("png_paths" if "image_path" in cols else "unknown")

    print("=" * 60)
    print(f"SAMPLE INSPECTION - index {index} (io_mode={io_mode})")
    print("=" * 60)

    # Messages
    messages = sample.get("messages", [])
    print(f"Number of messages: {len(messages)}")

    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        print(f"\nMessage {i+1} ({role}):")
        for j, content in enumerate(msg.get("content", [])):
            ctype = content.get("type", "?")
            if ctype == "text":
                text = content.get("text", "")
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"  Content {j+1} (text): {text}")
            elif ctype == "image":
                print(f"  Content {j+1} (image): [IMAGE PLACEHOLDER]")
            else:
                print(f"  Content {j+1} ({ctype}): {content}")

    # Image / path info
    if io_mode == "tif_direct" and "images" in sample:
        img = sample["images"]
        print("\nImage (in-dataset):")
        print(f"  Type: {type(img).__name__}")
        if isinstance(img, Image.Image):
            print(f"  Size: {img.size}, mode: {img.mode}")
    elif io_mode == "png_paths" and "image_path" in sample:
        p = sample["image_path"]
        print("\nImage path:")
        print(f"  {p}")
        if load_image:
            try:
                img = Image.open(p)
                print(f"  Loaded image -> size: {img.size}, mode: {img.mode}")
            except Exception as e:
                print(f"  Failed to load image from path: {e}")

    # Metadata
    meta = sample.get("metadata", None)
    if meta:
        print("\nMetadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")

    print("")


# ------------------------- Small subset creation --------------------------

def make_test_subset(ds: Dataset, num_samples: int = 10) -> Dataset:
    """
    Create a small subset for quick experiments / debugging.

    - If num_samples > len(ds), it is clipped.
    - Returns a new Dataset object.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    n = min(num_samples, len(ds))
    idx = list(range(n))
    subset = ds.select(idx)
    logger.info(f"Created subset of size {len(subset)} from dataset of size {len(ds)}")
    return subset


# ------------------------------- Demo main -------------------------------

def main():
    """
    Simple demo:
      - print summary
      - validate
      - inspect first train sample
      - create a small test subset
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    loader = ProcessedDatasetLoader()

    loader.print_summary()

    print("\nValidating datasets...")
    ok = loader.validate()
    print(f"Validation result: {'✓ PASS' if ok else '✗ FAIL'}")

    if not ok:
        return

    print("\nLoading train dataset...")
    train_ds = loader.load_train()

    print("Inspecting first train sample...")
    inspect_sample(train_ds, index=0, load_image=False)

    print("\nCreating small test subset (5 samples)...")
    subset = make_test_subset(train_ds, 5)
    print(f"Subset size: {len(subset)}")


if __name__ == "__main__":
    main()
