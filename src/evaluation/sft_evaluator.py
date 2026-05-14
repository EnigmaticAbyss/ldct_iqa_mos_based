from __future__ import annotations

from src.evaluation.generative_mos_evaluator import GenerativeMOSEvaluator


class SFTEvaluator(GenerativeMOSEvaluator):
    """Evaluates TRL-SFT generative MOS models."""

    evaluator_name = "sft"
