# scripts/train.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    """
    Read a JSON configuration file from disk.

    Args:
        path: Config file path.

    Returns:
        Parsed JSON dictionary.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    """
    Dispatch training to the configured regression, SFT, or GRPO trainer.

    Raises:
        ValueError: If the training mode is missing or unsupported.
    """
    ap = argparse.ArgumentParser(description="LDCT IQA training entrypoint")
    ap.add_argument("--config", type=Path, required=True, help="Path to training config JSON")
    args = ap.parse_args()

    cfg = load_json(args.config)

    mode = (
        cfg.get("train_mode")
        or cfg.get("sft_mode")
        or cfg.get("grpo_mode")
        or ""
    ).strip().lower()
    if mode not in ("regression", "trl_sft", "trl_grpo", "grpo"):
        raise ValueError("config must include train_mode/sft_mode = 'regression', 'trl_sft', or 'trl_grpo'")
   
   
    cfg_for_trainer = dict(cfg)
    cfg_for_trainer.pop("train_mode", None)
    cfg_for_trainer.pop("sft_mode", None)
    cfg_for_trainer.pop("grpo_mode", None)

    # Lazy imports so errors are cleaner if optional deps missing
    if mode == "regression":
        from src.trainers.regression_trainer import LDCTRegressionTrainer

        trainer = LDCTRegressionTrainer(cfg_for_trainer)
        trainer.run()
        return

    if mode == "trl_sft":
        from src.trainers.sft_trainer import LDCTSFTTrainer

        trainer = LDCTSFTTrainer(cfg_for_trainer)
        trainer.run()
        return

    if mode in ("trl_grpo", "grpo"):
        from src.trainers.grpo_trainer import LDCTGRPOTrainer

        trainer = LDCTGRPOTrainer(cfg_for_trainer)
        trainer.run()
        return


if __name__ == "__main__":
    main()
