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

    def __init__(self, processor, max_length: int = 2048, assistant_only_loss: bool = True):
        self.processor = processor
        self.max_length = max_length
        self.assistant_only_loss = assistant_only_loss
        self._warned_empty_assistant_labels = False

    @staticmethod
    def _prompt_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt = []
        for message in messages:
            if message.get("role") == "assistant":
                break
            prompt.append(message)
        return prompt or messages

    def _mask_prompt_tokens(
        self,
        labels: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        prompt_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        attention_mask = batch.get("attention_mask")
        prompt_attention_mask = prompt_batch.get("attention_mask")
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)

        for i in range(labels.shape[0]):
            if attention_mask is not None:
                nonpad_positions = attention_mask[i].bool().nonzero(as_tuple=False).flatten()
            elif pad_id is not None:
                nonpad_positions = (batch["input_ids"][i] != pad_id).nonzero(as_tuple=False).flatten()
            else:
                nonpad_positions = torch.arange(labels.shape[1], device=labels.device)

            if prompt_attention_mask is not None:
                prompt_len = int(prompt_attention_mask[i].sum().item())
            elif pad_id is not None:
                prompt_len = int((prompt_batch["input_ids"][i] != pad_id).sum().item())
            else:
                prompt_len = int(prompt_batch["input_ids"][i].numel())

            labels[i, nonpad_positions[: min(prompt_len, len(nonpad_positions))]] = -100

        if not self._warned_empty_assistant_labels and int((labels != -100).sum().item()) == 0:
            logger.warning(
                "All SFT labels are masked. The assistant answer may be truncated; "
                "increase max_length or check the chat template."
            )
            self._warned_empty_assistant_labels = True

        return labels

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
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100

        if self.assistant_only_loss:
            prompt_texts = [
                self.processor.apply_chat_template(
                    self._prompt_messages(ex["messages"]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for ex in examples
            ]
            prompt_batch = self.processor(
                text=prompt_texts,
                images=nested_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            labels = self._mask_prompt_tokens(labels, batch, prompt_batch)

        batch["labels"] = labels
        return batch
