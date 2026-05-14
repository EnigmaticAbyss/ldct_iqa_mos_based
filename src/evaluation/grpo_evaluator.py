from __future__ import annotations

from src.evaluation.generative_mos_evaluator import GenerativeMOSEvaluator


class GRPOEvaluator(GenerativeMOSEvaluator):
    """Evaluates GRPO-refined generative MOS models."""

    evaluator_name = "grpo"
