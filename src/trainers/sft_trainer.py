# src/trainers/sft_trainer.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor

from trl import SFTTrainer, SFTConfig
from peft import get_peft_model

from src.data.loaders import DatasetLoader
from src.data.format_builders import build_format_sft_dataset
from src.data.collators import FormatSFTCollator
from src.models.lora_adapters import resolve_lora_targets, build_lora_config_from_settings
from src.trainers.common import setup_logging, set_seed, build_bnb_config, ProcessorSaveCallback

logger = logging.getLogger("sft_trainer")


@dataclass
class SFTTrainConfig:
    # I/O
    model_name: str
    data_dir: str = "data/processed"
    use_jsonl: bool = False
    output_dir: str = "models/medgemma-trl-sft"
    logging_dir: str = "logs/sft"
    seed: int = 42

    # Chat formatting (NO stored prompts in dataset; we build messages here)
    system_prompt: str = "You are a medical image quality assessment assistant."
    user_text: str = "Predict MOS score."
    assistant_format: str = "answer_json"  # reserved for future variants

    # Sequence length
    max_length: int = 2048

    # Training
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Quantization
    use_4bit: bool = True
    use_8bit: bool = False
    bnb_compute_dtype: str = "bf16"

    # TRL-required
    remove_unused_columns: bool = False
    dataset_kwargs: Optional[dict] = None  # e.g. {"skip_prepare_dataset": True}

    # LoRA control
    lora_enabled: bool = True
    lora_scope: str = "llm"                # "llm" | "vision" | "both"
    lora_coverage: str = "linear_only"     # "linear_only" | "linear_and_conv" | "full_finetune"
    lora_include_patterns: Optional[list[str]] = None
    lora_exclude_patterns: Optional[list[str]] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class LDCTSFTTrainer:
    def __init__(self, config_dict: Dict[str, Any]):
        self.cfg = SFTTrainConfig(**config_dict)

        setup_logging(self.cfg.logging_dir, log_name="sft.log")
        set_seed(self.cfg.seed)

        self.model = None
        self.processor = None
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.trainer: Optional[SFTTrainer] = None

        logger.info(f"Initialized TRL SFT Trainer | model={self.cfg.model_name}")

    # ---------------- Data ---------------- #

    def load_data(self):
        loader = DatasetLoader(data_dir=self.cfg.data_dir, use_jsonl=self.cfg.use_jsonl)
        train, val = loader.load_train_val()

        # Base required columns
        DatasetLoader.require_columns(train, ["image_path", "mos_score"], name="train")
        DatasetLoader.require_columns(val, ["image_path", "mos_score"], name="val")

        # Build messages from MOS (no prompts stored in dataset)
        self.train_ds = build_format_sft_dataset(
            train,
            system_prompt=self.cfg.system_prompt,
            user_text=self.cfg.user_text,
        )
        self.val_ds = build_format_sft_dataset(
            val,
            system_prompt=self.cfg.system_prompt,
            user_text=self.cfg.user_text,
        )

        logger.info(f"Built TRL-format datasets | train={len(self.train_ds)} val={len(self.val_ds)}")
        DatasetLoader.require_columns(self.train_ds, ["messages", "image_path", "mos_score"], name="train_sft")
        DatasetLoader.require_columns(self.val_ds, ["messages", "image_path", "mos_score"], name="val_sft")

    # ---------------- Model ---------------- #

    def load_model(self):
        bnb = build_bnb_config(self.cfg.use_4bit, self.cfg.use_8bit, compute_dtype=self.cfg.bnb_compute_dtype)

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            device_map="auto",
            quantization_config=bnb,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)

        # Apply LoRA if enabled and not full_finetune
        if self.cfg.lora_enabled and self.cfg.lora_coverage != "full_finetune":
            plan = resolve_lora_targets(
                model,
                scope=self.cfg.lora_scope,
                coverage=self.cfg.lora_coverage,
                include_patterns=self.cfg.lora_include_patterns,
                exclude_patterns=self.cfg.lora_exclude_patterns,
            )
            if not plan.target_modules:
                logger.warning("LoRA enabled but no targets found. Proceeding without LoRA.")
            else:
                lora_cfg = build_lora_config_from_settings(
                    task_type="CAUSAL_LM",
                    r=self.cfg.lora_r,
                    alpha=self.cfg.lora_alpha,
                    dropout=self.cfg.lora_dropout,
                    target_modules=plan.target_modules,
                )
                model = get_peft_model(model, lora_cfg)
                logger.info("✅ LoRA adapters attached to model.")
        else:
            logger.info("LoRA disabled OR full_finetune selected (no adapters).")

        self.model = model
        self.processor = processor
        logger.info("Model + processor ready.")

    # ---------------- Trainer ---------------- #

    def build_trainer(self):
        if self.model is None or self.processor is None:
            raise ValueError("Call load_model() before build_trainer().")
        if self.train_ds is None or self.val_ds is None:
            raise ValueError("Call load_data() before build_trainer().")

        collator = FormatSFTCollator(self.processor, max_length=self.cfg.max_length)

        args = SFTConfig(
            output_dir=self.cfg.output_dir,
            logging_dir=self.cfg.logging_dir,

            num_train_epochs=self.cfg.num_train_epochs,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,

            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,

            logging_steps=self.cfg.logging_steps,
            save_steps=self.cfg.save_steps,
            eval_steps=self.cfg.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=self.cfg.save_total_limit,

            # IMPORTANT for custom multimodal collator
            remove_unused_columns=self.cfg.remove_unused_columns,
            dataset_kwargs=self.cfg.dataset_kwargs,

            fp16=self.cfg.fp16,
            bf16=self.cfg.bf16,

            report_to=["tensorboard"],
            label_pad_token_id=-100,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            data_collator=collator,
            processing_class=self.processor,
        )
        self.trainer.add_callback(ProcessorSaveCallback(self.processor))
        logger.info("TRL SFTTrainer built.")

    # ---------------- Run ---------------- #

    def run(self):
        logger.info("=== TRL SFT TRAINING START ===")
        self.load_data()
        self.load_model()
        self.build_trainer()

        out = self.trainer.train()
        logger.info("Training finished.")

        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        results = {
            "train_loss": getattr(out, "training_loss", None),
            "metrics": getattr(out, "metrics", {}),
        }
        (Path(self.cfg.output_dir) / "training_results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        # Save final model + processor
        self.trainer.save_model(self.cfg.output_dir)
        try:
            self.processor.save_pretrained(self.cfg.output_dir)
        except Exception as e:
            logger.warning(f"Failed to save processor: {e}")

        logger.info("=== TRL SFT TRAINING END ===")
        return results
