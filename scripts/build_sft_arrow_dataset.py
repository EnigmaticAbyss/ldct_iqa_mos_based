# scripts/build_sft_arrow_dataset.py
"""
This script converts a base dataset (JSONL) into a TRL-compatible Arrow dataset.
It reads image paths, builds TRL chat-formatted messages, and saves to disk.
"""

import json
import logging
from pathlib import Path

from datasets import Dataset

from src.data.format_builders import build_format_sft_dataset
from src.data.loaders import DatasetLoader
from src.trainers.common import setup_logging


def build_arrow_from_jsonl(
    base_jsonl: Path,  # Input JSONL file
    output_dir: Path,  # Output Arrow dataset directory
    system_prompt: str = "You are a medical image quality assessment assistant.",
    user_text: str = "Predict MOS score.",
):
    """
    Converts base JSONL file to TRL Arrow dataset with messages and image paths.

    Args:
        base_jsonl: Input JSONL file containing ``image_path`` and ``mos_score``.
        output_dir: Directory where the Arrow dataset will be saved.
        system_prompt: System prompt inserted into each chat example.
        user_text: User text inserted after the image placeholder.

    Returns:
        The generated Hugging Face ``Dataset``.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base JSONL (or dataset if you want to validate columns first)
    logger = logging.getLogger("build_arrow")
    setup_logging(output_dir, log_name="build_arrow.log")
    logger.info(f"Loading base JSONL dataset from {base_jsonl}")
    
    # Load the dataset
    with base_jsonl.open("r", encoding="utf-8") as f:
        base_data = [json.loads(line) for line in f]

    # Convert base data to Dataset
    base_dataset = Dataset.from_dict({
        "image_path": [item["image_path"] for item in base_data],
        "mos_score": [item["mos_score"] for item in base_data]
    })

    # Convert to TRL dataset (build messages dynamically)
    trl_dataset = build_format_sft_dataset(
        base_dataset,
        system_prompt=system_prompt,
        user_text=user_text
    )

    # Save Arrow dataset
    logger.info(f"Saving dataset to {output_dir}")
    trl_dataset.save_to_disk(str(output_dir))
    logger.info(f"Arrow dataset saved to {output_dir}")

    return trl_dataset


def main():
    """
    Parse CLI arguments and build an SFT Arrow dataset from JSONL.

    The command is a thin wrapper around ``build_arrow_from_jsonl``.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_jsonl", type=Path, required=True, help="Input base JSONL")
    ap.add_argument("--output_dir", type=Path, required=True, help="Output directory for Arrow dataset")
    args = ap.parse_args()

    build_arrow_from_jsonl(args.base_jsonl, args.output_dir)


if __name__ == "__main__":
    main()
