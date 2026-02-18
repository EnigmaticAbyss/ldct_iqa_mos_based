# src/trainers/common.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import BitsAndBytesConfig, TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger("trainer_common")


def setup_logging(logging_dir: str | Path, log_name: str = "train.log", level: int = logging.INFO) -> Path:
    """
    Create console + file logging with UTF-8 (important on Windows).
    Returns the log file path.
    """
    logging_dir = Path(logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)
    log_path = logging_dir / log_name

    # Avoid adding handlers multiple times if called repeatedly
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, encoding="utf-8"),
            ],
        )
    logger.info(f"Logging to: {log_path}")
    return log_path


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_bnb_config(use_4bit: bool, use_8bit: bool, compute_dtype: str = "bf16") -> Optional[BitsAndBytesConfig]:
    """
    Build BitsAndBytesConfig for 4-bit or 8-bit loading.
    If neither is enabled, returns None.
    """
    if not use_4bit and not use_8bit:
        return None

    dtype = torch.bfloat16 if compute_dtype.lower() in ("bf16", "bfloat16") else torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=bool(use_4bit),
        load_in_8bit=bool(use_8bit),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


class ProcessorSaveCallback(TrainerCallback):
    """
    Saves processor (tokenizer + image processor) into each checkpoint folder.
    Works for both HF Trainer and TRL trainers.
    """

    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return

        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.processor.save_pretrained(str(ckpt_dir))
            logger.info(f"✅ Processor saved to: {ckpt_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save processor to {ckpt_dir}: {e}")


@dataclass
class DeviceInfo:
    device: torch.device
    is_cuda: bool
    bf16_supported: bool


def get_device_info() -> DeviceInfo:
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    bf16_supported = False
    if is_cuda:
        try:
            bf16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_supported = False
    return DeviceInfo(device=device, is_cuda=is_cuda, bf16_supported=bf16_supported)
