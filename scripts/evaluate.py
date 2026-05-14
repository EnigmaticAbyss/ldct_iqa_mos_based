# scripts/evaluate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="LDCT IQA evaluation entrypoint")
    ap.add_argument("--config", type=Path, required=True, help="Path to eval config JSON")
    args = ap.parse_args()

    cfg = load_json(args.config)
    output_path = Path(cfg.get("output_path", "logs/eval/evaluation_results.json"))
    output_dir = Path(cfg.get("output_dir", output_path.parent))

    mode = str(cfg.get("eval_mode", "")).strip().lower()
    if mode not in ("regression", "sft", "grpo"):
        raise ValueError("config must include eval_mode = 'regression', 'sft', or 'grpo'")

    if mode == "regression":
        from src.evaluation.regression_evaluator import RegressionEvaluator

        evaluator = RegressionEvaluator(
            model_dir=cfg["model_dir"],
            data_dir=cfg.get("data_dir", "datasets/processed"),
            use_jsonl=cfg.get("use_jsonl", False),
            device=cfg.get("device", None),
        )
        results = evaluator.run()

    elif mode == "sft":
        from src.evaluation.sft_evaluator import SFTEvaluator

        evaluator = SFTEvaluator(
            model_dir=cfg["model_dir"],
            base_model_name=cfg.get("base_model_name", None),
            is_peft_adapter=cfg.get("is_peft_adapter", False),
            data_dir=cfg.get("data_dir", "datasets/processed"),
            use_jsonl=cfg.get("use_jsonl", False),
            dataset_format=cfg.get("dataset_format", None),
            test_dataset_dir=cfg.get("test_dataset_dir", None),
            test_json_path=cfg.get("test_json_path", cfg.get("test_jsonl_path", None)),
            system_prompt=cfg.get(
                "system_prompt",
                "You are a medical image quality assessment assistant.",
            ),
            user_text=cfg.get("user_text", "Predict MOS score."),
            device=cfg.get("device", None),
            output_dir=output_dir,
        )
        results = evaluator.run()

    else:
        from src.evaluation.grpo_evaluator import GRPOEvaluator

        evaluator = GRPOEvaluator(
            model_dir=cfg["model_dir"],
            base_model_name=cfg.get("base_model_name", None),
            is_peft_adapter=cfg.get("is_peft_adapter", False),
            data_dir=cfg.get("data_dir", "datasets/processed"),
            use_jsonl=cfg.get("use_jsonl", False),
            dataset_format=cfg.get("dataset_format", None),
            test_dataset_dir=cfg.get("test_dataset_dir", None),
            test_json_path=cfg.get("test_json_path", cfg.get("test_jsonl_path", None)),
            system_prompt=cfg.get(
                "system_prompt",
                "You are a medical image quality assessment assistant.",
            ),
            user_text=cfg.get("user_text", "Predict MOS score."),
            device=cfg.get("device", None),
            output_dir=output_dir,
        )
        results = evaluator.run()

    evaluator.save_results(results, output_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    # python -m scripts.compare_models --config config/compare_models.json
