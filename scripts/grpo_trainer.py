# File: src/grpo_trainer.py
"""
GRPO Trainer for LDCT IQA Project - Phase 3 (FINAL, MedGemma VLM-safe)

Key fixes vs your earlier version:
- ✅ Trains on TRAIN (and VAL) splits, NOT on TEST
- ✅ Keeps images in the GRPO dataset (does NOT drop them)
- ✅ Uses processor (not tokenizer) for TRL processing_class
- ✅ Ensures MOS is expanded to match num_generations
- ✅ Forces image-conditioned generation by subclassing GRPOTrainer
  (so MedGemma actually “sees” the CT slice during rollouts)
"""

import os
import gc
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any

import torch
from datasets import Dataset
from PIL import Image

from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback, TrainerState, TrainerControl

from config import grpo_config, model_config, logging_config
from src.model_setup import ModelLoader, load_model_checkpoint
from src.grpo_utils import GRPORewardFunction

logger = logging.getLogger(__name__)


class ProcessorSaveCallback(TrainerCallback):
    """Save processor at checkpoints."""
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            output_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
            try:
                self.processor.save_pretrained(output_dir)
                logger.info(f"✅ Processor saved to checkpoint: {output_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save processor: {e}")


class MedGemmaVLMGRPOTrainer(GRPOTrainer):
    """
    TRL GRPOTrainer does generation internally; for VLMs this can accidentally become text-only
    unless images are explicitly passed to processor/model.generate().
    This subclass forces image-conditioned generation for each prompt.
    """

    def __init__(self, *args, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        if processor is None:
            raise ValueError("processor must be provided for VLM GRPO.")
        self._vlm_processor = processor

    def _build_text(self, prompt_messages: List[Dict[str, Any]]) -> str:
        # prompt_messages is list of messages (system+user) where user content includes {"type":"image"}
        return self._vlm_processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate_completions(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generate num_generations completions per prompt, conditioning on image+text.
        Returns a flat list of length = batch_size * num_generations.
        """
        prompts = batch["prompt"]              # list of messages
        images = batch["images"]              # PIL Image or list/tuple containing PIL Image
        K = int(self.args.num_generations)

        completions_all: List[str] = []

        # Normalize images per sample to a single PIL.Image
        pil_images: List[Image.Image] = []
        for img in images:
            if isinstance(img, (list, tuple)):
                pil_images.append(img[0])
            else:
                pil_images.append(img)

        device = next(self.model.parameters()).device

        for prompt_messages, img in zip(prompts, pil_images):
            text = self._build_text(prompt_messages)

            # Processor expects nested list for VLM: images=[[img]]
            inputs = self._vlm_processor(
                text=[text],
                images=[[img]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=getattr(self.args, "max_prompt_length", 1024),
            )
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            for _ in range(K):
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=getattr(self.args, "max_completion_length", 128),
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=self._vlm_processor.tokenizer.eos_token_id,
                    )

                # decode only new tokens
                gen_ids = out[0][inputs["input_ids"].shape[1]:]
                gen_text = self._vlm_processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                completions_all.append(gen_text)

        return completions_all


class LDCTGRPOTrainer:
    """
    GRPO training pipeline (MedGemma VLM-safe).
    Keeps your “think/answer” rewards, monitoring, logging, and class structure.
    """

    def __init__(self, config_path: Optional[str] = None, sft_model_path: Optional[str] = None):
        self.config_path = config_path
        self.sft_model_path = sft_model_path

        self.grpo_config = grpo_config
        self.model_config = model_config
        self.logging_config = logging_config

        self.model = None
        self.processor = None
        self.trainer = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        self.reward_function = GRPORewardFunction(reward_weights=self.grpo_config.reward_weights)

        self.reward_history: List[Dict[str, float]] = []

        self._setup_logging()

        logger.info("Initialized LDCT GRPO Trainer")
        logger.info(f"Output directory: {self.grpo_config.output_dir}")
        logger.info(f"SFT model path: {self.sft_model_path}")

    def _setup_logging(self):
        os.makedirs(self.grpo_config.logging_dir, exist_ok=True)
        log_file = os.path.join(self.grpo_config.logging_dir, "grpo_training.log")

        # avoid duplicate handlers if you re-init in notebook
        for h in list(logger.handlers):
            logger.removeHandler(h)

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.INFO)

        logger.info(f"GRPO training logs will be saved to: {log_file}")

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _print_memory_status(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    # -------------------------
    # Setup model and datasets
    # -------------------------
    def setup_model_and_data(self):
        logger.info("Setting up model and data for GRPO training...")

        # Load model+processor
        if self.sft_model_path and os.path.exists(self.sft_model_path):
            logger.info(f"Loading SFT-trained model from: {self.sft_model_path}")
            self.model, self.processor = load_model_checkpoint(self.sft_model_path, self.model_config)
            logger.info("✅ SFT model loaded successfully for GRPO")
        else:
            logger.info("No SFT checkpoint provided; loading base model")
            model_loader = ModelLoader(self.model_config)
            self.model, self.processor = model_loader.setup_complete_model()

        # Ensure pad token exists
        tok = self.processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Load datasets
        self._load_grpo_datasets()

        logger.info("✅ Model and data setup completed for GRPO training")

    def _load_grpo_datasets(self):
        """
        IMPORTANT:
        - Train GRPO on TRAIN (and VAL), not on TEST.
        - Keep images.
        """
        from src.dataset_utils import DatasetLoader  # local import to avoid circulars

        logger.info("Loading datasets for GRPO training...")
        loader = DatasetLoader()

        train_full = loader.load_train_dataset()

        # If val exists, use it. Otherwise split train.
        try:
            val_full = loader.load_val_dataset()
        except Exception:
            val_full = None

        if val_full is None:
            train_size = int(0.9 * len(train_full))
            self.train_dataset = train_full.select(range(train_size))
            self.val_dataset = train_full.select(range(train_size, len(train_full)))
        else:
            self.train_dataset = train_full
            self.val_dataset = val_full

        logger.info(f"✅ GRPO datasets loaded: train={len(self.train_dataset)}, val={len(self.val_dataset)}")

        self._convert_to_prompt_dataset()

    def _convert_to_prompt_dataset(self):
        """
        Convert HF (messages, images, metadata) dataset to GRPO dataset:
        - prompt: messages[:2] (system + user)
        - images: keep the PIL image(s)
        - mos_score: float
        """
        logger.info("Converting dataset to GRPO prompt format (KEEPING images)...")

        def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
            prompt_messages = sample["messages"][:2]

            img = sample.get("images", None)
            if img is None:
                raise ValueError("Sample has no 'images' column. GRPO VLM requires images.")

            # metadata may be dict or missing depending on your pipeline
            mos = 0.0
            if "metadata" in sample and sample["metadata"] is not None:
                mos = sample["metadata"].get("mos_score", 0.0)
            else:
                mos = sample.get("mos_score", 0.0)

            return {
                "prompt": prompt_messages,
                "images": img,
                "mos_score": float(mos),
            }

        self.train_dataset = self.train_dataset.map(
            convert_sample,
            remove_columns=self.train_dataset.column_names,
        )
        self.val_dataset = self.val_dataset.map(
            convert_sample,
            remove_columns=self.val_dataset.column_names,
        )

        logger.info(f"Sample keys after conversion: {list(self.train_dataset[0].keys())}")
        logger.info("✅ Dataset converted (prompt+images+mos_score).")

    # -------------------------
    # Reward wrapper
    # -------------------------
    def _monitor_rewards(self, rewards: List[float]):
        if not rewards:
            return
        stats = {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "num": int(len(rewards)),
        }
        self.reward_history.append(stats)
        logger.debug(f"Reward stats: {stats}")

        if stats["std"] < 0.01:
            logger.warning("⚠️ Low reward variance detected (possible collapse).")

    def _reward_wrapper(self, prompts: List[Any], completions: List[str], **kwargs) -> List[float]:
        """
        TRL calls reward funcs with completions and extra columns.
        We must ensure mos_score aligns with completions when num_generations > 1.
        """
        try:
            mos_scores = kwargs.get("mos_score", None)
            if mos_scores is None:
                logger.warning("No mos_score found in kwargs; using default 2.0")
                mos_scores = [2.0] * len(completions)

            # make list
            if not isinstance(mos_scores, (list, tuple, np.ndarray)):
                mos_scores = [float(mos_scores)]

            K = int(getattr(self.trainer.args, "num_generations", self.grpo_config.num_generations))

            # If mos_scores is batch-sized, expand each mos K times:
            if len(mos_scores) * K == len(completions):
                mos_scores = [m for m in mos_scores for _ in range(K)]
            elif len(mos_scores) != len(completions):
                # broadcast fallback
                mos_scores = (mos_scores * (len(completions) // len(mos_scores) + 1))[:len(completions)]

            self._cleanup_memory()

            with torch.no_grad():
                rewards = self.reward_function.compute_reward(
                    prompts=prompts if prompts is not None else [""] * len(completions),
                    completions=completions,
                    mos_scores=mos_scores,
                    **kwargs
                )

            self._monitor_rewards(rewards)
            self._cleanup_memory()
            return rewards

        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            return [0.0] * len(completions)

    # -------------------------
    # Trainer setup + training
    # -------------------------
    def setup_trainer(self):
        logger.info("Setting up GRPO trainer...")

        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be set up first.")
        if self.train_dataset is None:
            raise ValueError("Train dataset must be loaded first.")

        args = GRPOConfig(
            output_dir=self.grpo_config.output_dir,
            logging_dir=self.grpo_config.logging_dir,

            num_train_epochs=self.grpo_config.num_train_epochs,
            per_device_train_batch_size=self.grpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.grpo_config.gradient_accumulation_steps,
            learning_rate=self.grpo_config.learning_rate,
            weight_decay=self.grpo_config.weight_decay,
            warmup_ratio=self.grpo_config.warmup_ratio,

            beta=self.grpo_config.beta,
            num_generations=self.grpo_config.num_generations,

            logging_steps=self.grpo_config.logging_steps,
            save_steps=self.grpo_config.save_steps,
            save_total_limit=self.grpo_config.save_total_limit,

            optim=self.grpo_config.optim,
            lr_scheduler_type=self.grpo_config.lr_scheduler_type,
            fp16=self.grpo_config.fp16,
            bf16=self.grpo_config.bf16,

            max_grad_norm=self.grpo_config.max_grad_norm,
            gradient_checkpointing=self.grpo_config.gradient_checkpointing,

            report_to=self.logging_config.report_to,

            # These may exist in your config; safe defaults if not:
            max_prompt_length=getattr(self.grpo_config, "max_prompt_length", 1024),
            max_completion_length=getattr(self.grpo_config, "max_completion_length", 128),
        )

        reward_functions = [self._reward_wrapper]

        # Use VLM-safe trainer subclass (forces processor(text, images))
        self.trainer = MedGemmaVLMGRPOTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            reward_funcs=reward_functions,
            processing_class=self.processor,   # processor, not tokenizer
            processor=self.processor,          # also stored for generate_completions override
        )

        self.trainer.add_callback(ProcessorSaveCallback(self.processor))
        logger.info("✅ GRPO trainer setup completed")

        self._log_training_config(args)

    def _log_training_config(self, args: GRPOConfig):
        logger.info("GRPO Training Configuration:")
        logger.info(f"  output_dir: {args.output_dir}")
        logger.info(f"  epochs: {args.num_train_epochs}")
        logger.info(f"  per_device_batch: {args.per_device_train_batch_size}")
        logger.info(f"  grad_accum: {args.gradient_accumulation_steps}")
        logger.info(f"  lr: {args.learning_rate}")
        logger.info(f"  beta: {args.beta}")
        logger.info(f"  num_generations: {args.num_generations}")
        logger.info(f"  max_prompt_length: {getattr(args, 'max_prompt_length', 'N/A')}")
        logger.info(f"  max_completion_length: {getattr(args, 'max_completion_length', 'N/A')}")

    def train(self) -> Dict[str, Any]:
        logger.info("🚀 Starting GRPO training...")

        if self.trainer is None:
            raise ValueError("Trainer must be set up first.")

        self._cleanup_memory()
        self._print_memory_status()

        out = self.trainer.train()

        results = {
            "train_loss": getattr(out, "training_loss", None),
            "train_runtime": out.metrics.get("train_runtime", 0.0),
            "train_samples_per_second": out.metrics.get("train_samples_per_second", 0.0),
            "train_steps_per_second": out.metrics.get("train_steps_per_second", 0.0),
            "epoch": out.metrics.get("epoch", 0.0),
            "reward_history": self.reward_history,
            "metrics": out.metrics,
        }

        logger.info("✅ GRPO training completed successfully")
        self._save_model()
        self._save_training_results(results)

        self._cleanup_memory()
        self._print_memory_status()
        return results

    def _save_model(self):
        save_path = os.path.join(self.grpo_config.output_dir, "final_model")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving GRPO model to: {save_path}")
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        logger.info("✅ GRPO model saved successfully")

    def _save_training_results(self, results: Dict[str, Any]):
        os.makedirs(self.grpo_config.output_dir, exist_ok=True)
        path = os.path.join(self.grpo_config.output_dir, "grpo_training_results.json")

        def convert(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o

        json_results = json.loads(json.dumps(results, default=convert))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Training results saved to: {path}")
