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

    Args:
        mos: Ground-truth MOS value to encode as the assistant answer.
        system_prompt: System instruction placed before the user turn.
        user_text: User instruction paired with the image placeholder.

    Returns:
        A system/user/assistant chat message list for SFT.
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


def build_prompt_messages(
    system_prompt: str,
    user_text: str,
) -> list[dict]:
    """
    Prompt-only chat format for GRPO.
    The image itself is stored in the dataset's image column; the message keeps
    the image placeholder that TRL fills before generation.
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
    ]


def strip_assistant_turns(messages: list[dict]) -> list[dict]:
    """
    Convert SFT chat rows into prompt-only rows by keeping turns before the
    first assistant response.
    """
    prompt = []
    for message in messages:
        if message.get("role") == "assistant":
            break
        prompt.append(message)
    return prompt


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
        """Convert a base MOS row into one TRL chat-training row."""
        mos = float(ex["mos_score"])
        return {
            "messages": build_messages(mos, system_prompt=system_prompt, user_text=user_text),
            "image_path": ex["image_path"],
            "mos_score": mos,
        }

    out = base_ds.map(_map, remove_columns=[c for c in base_ds.column_names if c not in ("image_path", "mos_score")])
    logger.info(f"Built format-SFT dataset with {len(out)} samples")
    return out


def build_format_grpo_dataset(
    base_ds: Dataset,
    system_prompt: str = "You are a medical image quality assessment assistant.",
    user_text: str = "Predict MOS score.",
    image_column: str = "image",
    cast_image_column: bool = True,
) -> Dataset:
    """
    Map either base JSON rows or prebuilt SFT Arrow rows -> TRL GRPO VLM rows.

    Accepted input columns:
      - image_path
      - mos_score
      - messages (optional; when present, assistant turns are removed)

    Output columns:
      - prompt
      - image
      - image_path
      - mos_score
    """

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a base or SFT row into one GRPO prompt row."""
        mos = float(ex["mos_score"])
        messages = ex.get("messages")
        prompt = strip_assistant_turns(messages) if messages else build_prompt_messages(
            system_prompt=system_prompt,
            user_text=user_text,
        )
        return {
            "prompt": prompt,
            image_column: ex["image_path"],
            "image_path": ex["image_path"],
            "mos_score": mos,
        }

    out = base_ds.map(_map, remove_columns=list(base_ds.column_names))

    if cast_image_column and image_column in out.column_names:
        try:
            from datasets import Image as HFImage

            out = out.cast_column(image_column, HFImage(decode=True))
        except Exception as e:
            logger.warning(f"Could not cast {image_column!r} to datasets.Image: {e}")

    logger.info(f"Built format-GRPO dataset with {len(out)} samples")
    return out
