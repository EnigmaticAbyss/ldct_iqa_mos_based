"""
src/ldct_sft_trainer.py

Final class-based SFT trainer for LDCT IQA VLM.

- Dataset format:
  columns: "messages", "image_path", "metadata"
  messages: TRL chat format with image placeholder in user content

- Supports reading:
  (A) HF datasets saved_to_disk in data/processed/{train,val,test}_dataset
  (B) OR JSONL files in data/processed/iqa_sft_{train,val,test}.jsonl

- Uses:
  * TRL SFTTrainer + SFTConfig
  * Custom collator for VLM text+image batches
  * LoRA (PEFT) and optional 4bit/8bit quantization
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from PIL import Image
from datasets import Dataset, load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_utils import ProcessedDatasetLoader  # noqa: E402


logger = logging.getLogger("ldct_sft_trainer")


# ---------------- Collator ---------------- #

class VLMDataCollator:
    """
    Collator that:
    - builds text via processor.apply_chat_template(messages)
    - loads image via image_path
    - processor() -> tensors
    - creates labels with pad masked to -100
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
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        batch["labels"] = labels
        return batch


# ---------------- Callback ---------------- #

class ProcessorSaveCallback(TrainerCallback):
    """Save processor whenever a checkpoint is saved."""

    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            try:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.processor.save_pretrained(str(ckpt_dir))
                logger.info(f"Processor saved to {ckpt_dir}")
            except Exception as e:
                logger.warning(f"Failed to save processor to {ckpt_dir}: {e}")


# ---------------- Config ---------------- #

@dataclass
class TrainConfig:
    # Model + data
    model_name: str
    output_dir: str
    logging_dir: str = "logs/sft"
    data_dir: str = "data/processed"
    use_jsonl: bool = False

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
    save_strategy: str = "steps"            # "steps" or "epoch"
    save_steps: int = 100
    evaluation_strategy: str = "steps"      # "steps" or "epoch"
    eval_steps: int = 100
    save_total_limit: int = 3

    # VLM / TRL required
    remove_unused_columns: bool = False
    dataset_kwargs: Optional[dict] = None   # e.g. {"skip_prepare_dataset": True}

    # Optim / scheduler
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Grad settings
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True

    # Quantization
    use_4bit: bool = True
    use_8bit: bool = False

    # LoRA (flattened)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # e.g. ["q_proj", "v_proj"]

    # Optional balanced debug subset
    use_balanced_debug_subset: bool = False
    debug_subset_size: int = 0


def load_config(path: Path) -> TrainConfig:
    d = json.loads(path.read_text(encoding="utf-8"))

    lora = d.pop("lora", {})
    d["lora_r"] = lora.get("r", d.get("lora_r", 16))
    d["lora_alpha"] = lora.get("alpha", d.get("lora_alpha", 32))
    d["lora_dropout"] = lora.get("dropout", d.get("lora_dropout", 0.05))
    d["lora_target_modules"] = lora.get("target_modules", d.get("lora_target_modules", ["q_proj", "v_proj"]))

    return TrainConfig(**d)


# ---------------- Balanced subset (optional) ---------------- #

def select_balanced_subset(ds: Dataset, target_size: int, key: str = "quality_category") -> Dataset:
    if target_size <= 0 or target_size >= len(ds):
        return ds

    import random
    random.seed(42)

    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(ds):
        cat = s.get("metadata", {}).get(key, "Unknown")
        groups.setdefault(cat, []).append(i)

    cats = list(groups.keys())
    if not cats:
        return ds

    base = target_size // len(cats)
    rem = target_size % len(cats)

    selected: List[int] = []
    for i, cat in enumerate(cats):
        want = base + (1 if i < rem else 0)
        idxs = groups[cat]
        n = min(len(idxs), want)
        if n > 0:
            selected.extend(random.sample(idxs, n))

    if not selected:
        return ds
    return ds.select(selected)


# ---------------- Trainer class ---------------- #

class LDCTSFTTrainer:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cfg = load_config(config_path)

        self.model = None
        self.processor = None
        self.trainer: Optional[SFTTrainer] = None
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None

        self._setup_logging()

    def _setup_logging(self):
        Path(self.cfg.logging_dir).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(Path(self.cfg.logging_dir) / "sft_training.log", encoding="utf-8"),
            ],
        )
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Logging: {Path(self.cfg.logging_dir) / 'sft_training.log'}")

    def load_data(self):
        loader = ProcessedDatasetLoader(data_dir=Path(self.cfg.data_dir))

        if self.cfg.use_jsonl:
            logger.info("Loading train/val from JSONL...")
            self.train_ds = load_dataset("json", data_files=str(loader.train_jsonl))["train"]
            self.val_ds = load_dataset("json", data_files=str(loader.val_jsonl))["train"]
        else:
            logger.info("Loading train/val from HF save_to_disk...")
            if not loader.validate():
                logger.warning("HF dataset validation failed (missing paths/columns). Proceeding anyway.")
            self.train_ds = loader.load_train()
            self.val_ds = loader.load_val()

        logger.info(f"Train samples: {len(self.train_ds)} | Val samples: {len(self.val_ds)}")

        if self.cfg.use_balanced_debug_subset and self.cfg.debug_subset_size > 0:
            logger.info(f"Balanced debug subset enabled: {self.cfg.debug_subset_size}")
            self.train_ds = select_balanced_subset(self.train_ds, self.cfg.debug_subset_size)

    def load_model(self):
        bnb = None
        if self.cfg.use_4bit or self.cfg.use_8bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=self.cfg.use_4bit,
                load_in_8bit=self.cfg.use_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            device_map="auto",
            quantization_config=bnb,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=True,
        )

        if self.cfg.gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled.")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")

        self.model.train()
        logger.info("Model + processor loaded.")

    def build_trainer(self):
        if self.train_ds is None or self.val_ds is None:
            raise ValueError("Call load_data() before build_trainer().")
        if self.model is None or self.processor is None:
            raise ValueError("Call load_model() before build_trainer().")

        target_modules = self.cfg.lora_target_modules or ["q_proj", "v_proj"]
        lora = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        collator = VLMDataCollator(self.processor, max_length=self.cfg.max_length)

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

            save_strategy=self.cfg.save_strategy,
            save_steps=self.cfg.save_steps if self.cfg.save_strategy == "steps" else None,
            evaluation_strategy=self.cfg.evaluation_strategy,
            eval_steps=self.cfg.eval_steps if self.cfg.evaluation_strategy == "steps" else None,
            save_total_limit=self.cfg.save_total_limit,

            remove_unused_columns=self.cfg.remove_unused_columns,
            dataset_kwargs=self.cfg.dataset_kwargs,

            optim=self.cfg.optim,
            lr_scheduler_type=self.cfg.lr_scheduler_type,

            fp16=self.cfg.fp16,
            bf16=self.cfg.bf16,

            max_grad_norm=self.cfg.max_grad_norm,

            report_to=["tensorboard"],
            label_pad_token_id=-100,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            data_collator=collator,
            peft_config=lora,
            processing_class=self.processor,
        )
        self.trainer.add_callback(ProcessorSaveCallback(self.processor))
        logger.info("Trainer built successfully.")

    def train(self) -> Dict[str, Any]:
        if self.trainer is None:
            raise ValueError("Call build_trainer() before train().")

        logger.info("Starting SFT training...")
        out = self.trainer.train()
        logger.info("Training finished.")

        metrics = out.metrics if hasattr(out, "metrics") else {}
        results = {"train_loss": getattr(out, "training_loss", None), "metrics": metrics}

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "training_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info(f"Saved {out_dir / 'training_results.json'}")

        return results

    def save_final(self):
        if self.trainer is None:
            raise ValueError("Trainer is not built yet.")

        out_dir = Path(self.cfg.output_dir)
        logger.info(f"Saving model/adapters to {out_dir}")
        self.trainer.save_model(str(out_dir))
        try:
            self.processor.save_pretrained(str(out_dir))
        except Exception as e:
            logger.warning(f"Failed to save processor: {e}")

    def run(self) -> Dict[str, Any]:
        logger.info("=== LDCT SFT PIPELINE START ===")
        self.load_data()
        self.load_model()
        self.build_trainer()
        results = self.train()
        self.save_final()
        logger.info("=== LDCT SFT PIPELINE END ===")
        return results


def main():
    import argparse
    ap = argparse.ArgumentParser(description="LDCT IQA VLM SFT Trainer")
    ap.add_argument("--config", type=Path, default=Path("config/train_sft_vlm_iqa.json"))
    args = ap.parse_args()

    trainer = LDCTSFTTrainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
