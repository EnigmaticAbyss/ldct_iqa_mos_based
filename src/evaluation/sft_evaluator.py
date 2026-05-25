from __future__ import annotations

from src.evaluation.generative_mos_evaluator import GenerativeMOSEvaluator


class SFTEvaluator(GenerativeMOSEvaluator):
    """
    Evaluates TRL-SFT generative MOS models.

    This subclass only changes the evaluator name used in result metadata while
    reusing the shared generative MOS evaluation pipeline.
    """

    evaluator_name = "sft"
