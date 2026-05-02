# scripts/compare_models.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.compare import compare_models


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Compare multiple evaluated models")
    ap.add_argument("--config", type=Path, required=True, help="Path to compare config JSON")
    args = ap.parse_args()

    cfg = load_json(args.config)

    models = cfg.get("models", [])
    out_dir = cfg.get("output_dir", "output/eval/comparison")

    if not models:
        raise ValueError("Config must contain a non-empty 'models' list")

    result = compare_models(models=models, out_dir=out_dir)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()