# src/data/format_builders.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from datasets import Dataset

logger = logging.getLogger("format_builders")


def build_assistant_answer_only_mos(mos: float) -> str:
    """
    Minimal assistant output that contains ONLY the MOS as the 'rating'.
    This keeps it consistent with your goal: no reasoning/prompting.
    """
    # Keep it as JSON so later GRPO can reward format reliably.
    # You can switch to raw "2.7" if you really want, but JSON is safer.
    return json.dumps({"rating": float(mos)}, ensure_ascii=False)


def build_messages(
    mos: float,
    system_prompt: str,
    user_text: str,
) -> list[dict]:
    """
    TRL chat format with an image placeholder.
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": build_assistant_answer_only_mos(mos)}],
        },
    ]


def build_format_sft_dataset(
    base_ds: Dataset,
    system_prompt: str = "You are a medical image quality assessment assistant.",
    user_text: str = "Predict MOS score.",
) -> Dataset:
    """
    Map base dataset -> TRL dataset with messages.

    Input columns required:
      - image_path
      - mos_score

    Output columns:
      - messages
      - image_path
      - mos_score
    """

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        mos = float(ex["mos_score"])
        return {
            "messages": build_messages(mos, system_prompt=system_prompt, user_text=user_text),
            "image_path": ex["image_path"],
            "mos_score": mos,
        }

    out = base_ds.map(_map, remove_columns=[c for c in base_ds.column_names if c not in ("image_path", "mos_score")])
    logger.info(f"Built format-SFT dataset with {len(out)} samples")
    return out
