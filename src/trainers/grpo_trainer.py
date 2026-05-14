# src/trainers/grpo_trainer.py
from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import Dataset
from peft import PeftModel, get_peft_model
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoModelClass
except ImportError:
    from transformers import AutoModelForCausalLM as AutoModelClass

from trl import GRPOConfig, GRPOTrainer

from src.data.format_builders import build_format_grpo_dataset
from src.data.loaders import DatasetLoader
from src.models.lora_adapters import resolve_lora_targets, build_lora_config_from_settings
from src.trainers.common import (
    ProcessorSaveCallback,
    build_bnb_config,
    set_seed,
    setup_logging,
    train_with_optional_resume,
)
from src.trainers.grpo_rewards import MOSRewardConfig, make_mos_reward_function

logger = logging.getLogger("grpo_trainer")


@dataclass
class GRPOTrainConfig:
    model_name: str
    base_model_name: Optional[str] = None
    adapter_model_dir: Optional[str] = None

    dataset_format: str = "arrow"
    load_prebuilt_sft_dataset: Optional[bool] = None
    data_dir: str = "datasets/processed"
    use_jsonl: bool = False
    train_dataset_dir: Optional[str] = None
    val_dataset_dir: Optional[str] = None
    test_dataset_dir: Optional[str] = None
    train_json_path: Optional[str] = None
    val_json_path: Optional[str] = None
    test_json_path: Optional[str] = None
    train_jsonl_path: Optional[str] = None
    val_jsonl_path: Optional[str] = None
    test_jsonl_path: Optional[str] = None

    system_prompt: str = "You are a medical image quality assessment assistant."
    user_text: str = "Predict MOS score."
    image_column: str = "image"

    output_dir: str = "output/model/medgemma15_iqa_grpo"
    logging_dir: str = "logs/grpo"
    seed: int = 42

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0

    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 3
    report_to: Optional[list[str] | str] = None

    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[dict] = None
    remove_unused_columns: bool = False

    max_prompt_length: Optional[int] = None
    max_completion_length: int = 32
    num_generations: int = 2
    num_generations_eval: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    generation_kwargs: Optional[dict] = None

    beta: float = 0.0
    num_iterations: int = 1
    epsilon: float = 0.2
    loss_type: Optional[str] = None
    scale_rewards: Optional[str] = None
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.6
    ref_model_sync_steps: int = 512
    log_completions: bool = True
    num_completions_to_print: Optional[int] = None

    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1

    use_4bit: bool = True
    use_8bit: bool = False
    bnb_compute_dtype: str = "bf16"
    torch_dtype: Optional[str] = None

    lora_enabled: bool = True
    lora_scope: str = "both"
    lora_coverage: str = "linear_only"
    lora_include_patterns: Optional[list[str]] = None
    lora_exclude_patterns: Optional[list[str]] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    mos_min: float = 0.0
    mos_max: float = 4.0
    reward_kind: str = "neg_abs_error"
    reward_scale: float = 1.0
    reward_offset: float = 0.0
    missing_reward: float = -4.0
    clamp_prediction: bool = True


class LDCTGRPOTrainer:
    def __init__(self, config_dict: Dict[str, Any]):
        self.cfg = GRPOTrainConfig(**config_dict)

        setup_logging(self.cfg.logging_dir, log_name="grpo.log")
        set_seed(self.cfg.seed)

        self.model = None
        self.processor = None
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.trainer: Optional[GRPOTrainer] = None

        logger.info(f"Initialized TRL GRPO Trainer | model={self.cfg.model_name}")

    # ---------------- Data ---------------- #

    @staticmethod
    def _first_path(*values: Optional[str]) -> Optional[str]:
        for value in values:
            if value:
                return value
        return None

    def _dataset_format(self) -> str:
        if self.cfg.load_prebuilt_sft_dataset is True:
            return "arrow"
        if self.cfg.load_prebuilt_sft_dataset is False and self.cfg.use_jsonl:
            return "json"
        return self.cfg.dataset_format

    def load_data(self):
        loader = DatasetLoader(
            data_dir=self.cfg.data_dir,
            use_jsonl=self.cfg.use_jsonl,
            dataset_format=self._dataset_format(),
            train_dataset_dir=self.cfg.train_dataset_dir,
            val_dataset_dir=self.cfg.val_dataset_dir,
            test_dataset_dir=self.cfg.test_dataset_dir,
            train_json_path=self._first_path(self.cfg.train_json_path, self.cfg.train_jsonl_path),
            val_json_path=self._first_path(self.cfg.val_json_path, self.cfg.val_jsonl_path),
            test_json_path=self._first_path(self.cfg.test_json_path, self.cfg.test_jsonl_path),
        )
        train, val = loader.load_train_val()

        DatasetLoader.require_columns(train, ["image_path", "mos_score"], name="train")
        DatasetLoader.require_columns(val, ["image_path", "mos_score"], name="val")

        self.train_ds = build_format_grpo_dataset(
            train,
            system_prompt=self.cfg.system_prompt,
            user_text=self.cfg.user_text,
            image_column=self.cfg.image_column,
        )
        self.val_ds = build_format_grpo_dataset(
            val,
            system_prompt=self.cfg.system_prompt,
            user_text=self.cfg.user_text,
            image_column=self.cfg.image_column,
        )

        required = ["prompt", self.cfg.image_column, "image_path", "mos_score"]
        DatasetLoader.require_columns(self.train_ds, required, name="train_grpo")
        DatasetLoader.require_columns(self.val_ds, required, name="val_grpo")
        logger.info(f"Built TRL-GRPO datasets | train={len(self.train_ds)} val={len(self.val_ds)}")

    # ---------------- Model ---------------- #

    @staticmethod
    def _resolve_torch_dtype(name: Optional[str]):
        if name is None:
            return None
        value = name.strip().lower()
        if value in {"auto", "none"}:
            return value if value == "auto" else None
        if value in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if value in {"fp16", "float16"}:
            return torch.float16
        if value in {"fp32", "float32"}:
            return torch.float32
        raise ValueError("torch_dtype must be one of: auto, bf16, fp16, fp32")

    def load_model(self):
        adapter_model_dir = self.cfg.adapter_model_dir
        model_name = self.cfg.base_model_name or self.cfg.model_name
        if adapter_model_dir is None and (Path(self.cfg.model_name) / "adapter_config.json").exists():
            if not self.cfg.base_model_name:
                raise ValueError(
                    "model_name points to a PEFT adapter. Set base_model_name or use adapter_model_dir."
                )
            adapter_model_dir = self.cfg.model_name

        bnb = build_bnb_config(self.cfg.use_4bit, self.cfg.use_8bit, compute_dtype=self.cfg.bnb_compute_dtype)
        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "quantization_config": bnb,
            "trust_remote_code": True,
        }
        dtype = self._resolve_torch_dtype(self.cfg.torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        model = AutoModelClass.from_pretrained(model_name, **model_kwargs)
        if adapter_model_dir:
            logger.info(f"Loading trainable PEFT adapter from: {adapter_model_dir}")
            model = PeftModel.from_pretrained(model, adapter_model_dir, is_trainable=True)

        try:
            processor = AutoProcessor.from_pretrained(
                adapter_model_dir or model_name,
                trust_remote_code=True,
                use_fast=False,
            )
        except Exception as e:
            if not adapter_model_dir:
                raise
            logger.warning(
                f"Could not load processor from adapter dir: {e}. Falling back to base model processor."
            )
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

        tokenizer = getattr(processor, "tokenizer", processor)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        if adapter_model_dir:
            logger.info("Using loaded PEFT adapter as the trainable GRPO adapter.")
        elif self.cfg.lora_enabled and self.cfg.lora_coverage != "full_finetune":
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
                logger.info("LoRA adapters attached to model.")
        else:
            logger.info("LoRA disabled OR full_finetune selected (no adapters).")

        self.model = model
        self.processor = processor
        logger.info("Model + processor ready.")

    # ---------------- Trainer ---------------- #

    def _grpo_args(self) -> GRPOConfig:
        candidates: Dict[str, Any] = {
            "output_dir": self.cfg.output_dir,
            "logging_dir": self.cfg.logging_dir,
            "num_train_epochs": self.cfg.num_train_epochs,
            "max_steps": self.cfg.max_steps,
            "per_device_train_batch_size": self.cfg.per_device_train_batch_size,
            "per_device_eval_batch_size": self.cfg.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "learning_rate": self.cfg.learning_rate,
            "weight_decay": self.cfg.weight_decay,
            "warmup_ratio": self.cfg.warmup_ratio,
            "optim": self.cfg.optim,
            "max_grad_norm": self.cfg.max_grad_norm,
            "logging_steps": self.cfg.logging_steps,
            "save_steps": self.cfg.save_steps,
            "eval_steps": self.cfg.eval_steps,
            "eval_strategy": self.cfg.eval_strategy,
            "save_strategy": self.cfg.save_strategy,
            "save_total_limit": self.cfg.save_total_limit,
            "report_to": self.cfg.report_to or ["tensorboard"],
            "fp16": self.cfg.fp16,
            "bf16": self.cfg.bf16,
            "gradient_checkpointing": self.cfg.gradient_checkpointing,
            "gradient_checkpointing_kwargs": self.cfg.gradient_checkpointing_kwargs,
            "remove_unused_columns": self.cfg.remove_unused_columns,
            "max_prompt_length": self.cfg.max_prompt_length,
            "max_completion_length": self.cfg.max_completion_length,
            "num_generations": self.cfg.num_generations,
            "num_generations_eval": self.cfg.num_generations_eval,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "top_k": self.cfg.top_k,
            "min_p": self.cfg.min_p,
            "repetition_penalty": self.cfg.repetition_penalty,
            "generation_kwargs": self.cfg.generation_kwargs,
            "beta": self.cfg.beta,
            "num_iterations": self.cfg.num_iterations,
            "epsilon": self.cfg.epsilon,
            "loss_type": self.cfg.loss_type,
            "scale_rewards": self.cfg.scale_rewards,
            "sync_ref_model": self.cfg.sync_ref_model,
            "ref_model_mixup_alpha": self.cfg.ref_model_mixup_alpha,
            "ref_model_sync_steps": self.cfg.ref_model_sync_steps,
            "log_completions": self.cfg.log_completions,
            "num_completions_to_print": self.cfg.num_completions_to_print,
            "use_vllm": self.cfg.use_vllm,
            "vllm_mode": self.cfg.vllm_mode,
            "vllm_gpu_memory_utilization": self.cfg.vllm_gpu_memory_utilization,
            "vllm_tensor_parallel_size": self.cfg.vllm_tensor_parallel_size,
        }

        signature = inspect.signature(GRPOConfig.__init__)
        allowed = set(signature.parameters) - {"self"}
        if "eval_strategy" not in allowed and "evaluation_strategy" in allowed:
            candidates["evaluation_strategy"] = candidates.pop("eval_strategy")

        filtered = {
            key: value
            for key, value in candidates.items()
            if key in allowed and value is not None
        }
        return GRPOConfig(**filtered)

    def build_trainer(self):
        if self.model is None or self.processor is None:
            raise ValueError("Call load_model() before build_trainer().")
        if self.train_ds is None or self.val_ds is None:
            raise ValueError("Call load_data() before build_trainer().")

        reward_cfg = MOSRewardConfig(
            mos_min=self.cfg.mos_min,
            mos_max=self.cfg.mos_max,
            reward_kind=self.cfg.reward_kind,
            reward_scale=self.cfg.reward_scale,
            reward_offset=self.cfg.reward_offset,
            missing_reward=self.cfg.missing_reward,
            clamp_prediction=self.cfg.clamp_prediction,
        )
        reward_func = make_mos_reward_function(reward_cfg)

        kwargs = {
            "model": self.model,
            "args": self._grpo_args(),
            "reward_funcs": [reward_func],
            "train_dataset": self.train_ds,
            "eval_dataset": self.val_ds,
        }
        trainer_sig = inspect.signature(GRPOTrainer.__init__)
        if "processing_class" in trainer_sig.parameters:
            kwargs["processing_class"] = self.processor
        elif "tokenizer" in trainer_sig.parameters:
            kwargs["tokenizer"] = self.processor

        self.trainer = GRPOTrainer(**kwargs)
        self.trainer.add_callback(ProcessorSaveCallback(self.processor))
        logger.info("TRL GRPOTrainer built.")

    # ---------------- Run ---------------- #

    def run(self):
        logger.info("=== TRL GRPO TRAINING START ===")
        self.load_data()
        self.load_model()
        self.build_trainer()

        out = train_with_optional_resume(self.trainer, self.cfg.output_dir, logger)
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

        self.trainer.save_model(self.cfg.output_dir)
        try:
            self.processor.save_pretrained(self.cfg.output_dir)
        except Exception as e:
            logger.warning(f"Failed to save processor: {e}")

        logger.info("=== TRL GRPO TRAINING END ===")
        return results
