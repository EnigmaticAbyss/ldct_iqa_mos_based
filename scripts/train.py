# scripts/train.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="LDCT IQA training entrypoint")
    ap.add_argument("--config", type=Path, required=True, help="Path to config JSON (sft.json)")
    args = ap.parse_args()

    cfg = load_json(args.config)

    mode = cfg.get("sft_mode", "").strip().lower()
    if mode not in ("regression", "trl_sft"):
        raise ValueError("config must include sft_mode = 'regression' or 'trl_sft'")

    # Lazy imports so errors are cleaner if optional deps missing
    if mode == "regression":
        from src.trainers.regression_trainer import LDCTRegressionTrainer

        trainer = LDCTRegressionTrainer(cfg)
        trainer.run()
        return

    if mode == "trl_sft":
        from src.trainers.sft_trainer import LDCTSFTTrainer

        trainer = LDCTSFTTrainer(cfg)
        trainer.run()
        return


if __name__ == "__main__":
    main()
