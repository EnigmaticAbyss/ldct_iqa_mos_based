# src/data/collators.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger("collators")


class MOSRegressionCollator:
    """
    Collator for MOS regression-head training (HF Trainer).

    Input example keys:
      - image_path: str
      - mos_score: float

    Output:
      - processor batch tensors
      - labels: float tensor (B,)
    """

    def __init__(self, processor, max_length: Optional[int] = 256, prompt_text: str = "Predict MOS score."):
        self.processor = processor
        self.max_length = max_length
        self.prompt_text = prompt_text

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [Image.open(ex["image_path"]).convert("RGB") for ex in examples]

        # Many VLM processors expect text + image together. We keep text minimal and constant.
        texts = [self.prompt_text] * len(examples)

        # Some VLMs want nested list of images: [[img], [img], ...]
        nested_images = [[img] for img in images]

        batch = self.processor(
            text=texts,
            images=nested_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = torch.tensor([float(ex["mos_score"]) for ex in examples], dtype=torch.float32)
        batch["labels"] = labels
        return batch


class FormatSFTCollator:
    """
    Collator for TRL SFT training.

    Input example keys:
      - messages: TRL chat list with image placeholder in user content
      - image_path: str
      - mos_score: float (not strictly needed for SFT, but kept for debugging)

    Output:
      - processor batch tensors
      - labels: masked input_ids for causal LM loss
    """

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [
            self.processor.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for ex in examples
        ]

        images = [Image.open(ex["image_path"]).convert("RGB") for ex in examples]
        nested_images = [[img] for img in images]

        batch = self.processor(
            text=texts,
            images=nested_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = batch["input_ids"].clone()
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        batch["labels"] = labels
        return batch
