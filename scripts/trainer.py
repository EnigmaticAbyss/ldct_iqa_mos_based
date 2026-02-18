"""
src/ldct_train.py

Unified trainer for LDCT IQA VLM with two modes:
1) mode="regression": MOS regression (image -> float MOS) using HF Trainer
2) mode="format_sft": short TRL SFT to teach response formatting (e.g. <answer> JSON)

Uses ONE config JSON to switch modes and LoRA scope/coverage.

Dataset expected (from your pipeline):
- HF dataset saved_to_disk OR JSONL
- For regression: needs columns "image_path", "metadata" (metadata.mos_score)
- For format_sft: we will build messages on the fly from image_path + mos_score
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from datasets import Dataset, load_dataset

from transformers import (
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_utils import ProcessedDatasetLoader  # noqa: E402


logger = logging.getLogger("ldct_train")


# ---------------- Callback ---------------- #

class ProcessorSaveCallback(TrainerCallback):
    """Save processor whenever a checkpoint is saved."""
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            try:
                self.processor.save_pretrained(str(ckpt_dir))
                logger.info(f"Processor saved to {ckpt_dir}")
            except Exception as e:
                logger.warning(f"Failed to save processor to {ckpt_dir}: {e}")


# ---------------- Config ---------------- #

@dataclass
class TrainConfig:
    # Global switch
    mode: str  # "regression" or "format_sft"

    # Model
    model_name: str
    output_dir: str
    logging_dir: str = "logs/train"
    data_dir: str = "data/processed"
    use_jsonl: bool = False

    # Quantization
    use_4bit: bool = True
    use_8bit: bool = False

    # Precision
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # LoRA controls
    # scope: "vision" | "llm" | "both"
    # coverage: "linear_only" | "linear_and_conv" | "full_finetune"
    lora: Dict[str, Any] = None

    # ---------- Regression training ----------
    reg: Dict[str, Any] = None

    # ---------- Format SFT training ----------
    fmt: Dict[str, Any] = None


def load_config(path: Path) -> TrainConfig:
    d = json.loads(path.read_text(encoding="utf-8"))
    return TrainConfig(**d)


# ---------------- Helpers: dataset loading ---------------- #

def load_train_val(cfg: TrainConfig) -> Tuple[Dataset, Dataset]:
    loader = ProcessedDatasetLoader(data_dir=Path(cfg.data_dir))

    if cfg.use_jsonl:
        train_ds = load_dataset("json", data_files=str(loader.train_jsonl))["train"]
        val_ds = load_dataset("json", data_files=str(loader.val_jsonl))["train"]
    else:
        train_ds = loader.load_train()
        val_ds = loader.load_val()

    return train_ds, val_ds


# ---------------- Helpers: LoRA targeting ---------------- #

def _matches_any(name: str, patterns: List[str]) -> bool:
    if not patterns:
        return True
    for p in patterns:
        if re.search(p, name):
            return True
    return False


def discover_target_leaf_names(
    model: nn.Module,
    include_module_types: Tuple[type, ...],
    include_name_patterns: List[str],
    exclude_name_patterns: List[str],
) -> List[str]:
    """
    Return leaf attribute names (e.g., q_proj, v_proj, ...) for modules of given types
    whose full module name matches include patterns and not exclude patterns.
    PEFT matches target_modules by substring of module names.
    """
    leafs = set()
    for full_name, m in model.named_modules():
        if not isinstance(m, include_module_types):
            continue
        if include_name_patterns and not _matches_any(full_name, include_name_patterns):
            continue
        if exclude_name_patterns and _matches_any(full_name, exclude_name_patterns):
            continue
        leafs.add(full_name.split(".")[-1])
    return sorted(leafs)


def build_lora(
    base_model: nn.Module,
    lora_cfg: Dict[str, Any],
) -> nn.Module:
    """
    Apply one of:
    - full_finetune: no LoRA; set requires_grad True everywhere
    - linear_only: LoRA on Linear
    - linear_and_conv: LoRA on Linear + Conv2d

    scope controls which subtrees are targeted by name patterns.
    """
    if lora_cfg is None:
        lora_cfg = {}

    enabled = bool(lora_cfg.get("enabled", True))
    if not enabled:
        logger.info("LoRA disabled (enabled=false). Full model trainable? -> NO (frozen remains as loaded).")
        return base_model

    coverage = str(lora_cfg.get("coverage", "linear_only")).lower()
    scope = str(lora_cfg.get("scope", "both")).lower()

    # You can override patterns in config if MedGemma names differ on your side:
    # include_patterns / exclude_patterns are regexes applied to full module names.
    include_patterns = lora_cfg.get("include_patterns", None)
    exclude_patterns = lora_cfg.get("exclude_patterns", None)

    if include_patterns is None:
        # reasonable defaults:
        # - "vision" scope tries common prefixes
        # - "llm" scope tries common prefixes
        vision_p = [r"vision", r"visual", r"image", r"encoder", r"vit", r"clip"]
        llm_p = [r"language", r"llm", r"decoder", r"model", r"transformer", r"text"]

        if scope == "vision":
            include_patterns = vision_p
        elif scope == "llm":
            include_patterns = llm_p
        else:
            include_patterns = []  # both -> include everything
    if exclude_patterns is None:
        exclude_patterns = []

    if coverage == "full_finetune":
        # True “all layers” training (not LoRA)
        for p in base_model.parameters():
            p.requires_grad = True
        logger.info("coverage=full_finetune: all parameters set trainable (this is NOT LoRA).")
        return base_model

    # LoRA module types
    module_types: Tuple[type, ...] = (nn.Linear,)
    if coverage == "linear_and_conv":
        module_types = (nn.Linear, nn.Conv2d)

    target_modules = discover_target_leaf_names(
        model=base_model,
        include_module_types=module_types,
        include_name_patterns=include_patterns,
        exclude_name_patterns=exclude_patterns,
    )

    if not target_modules:
        logger.warning(
            "No LoRA target modules discovered. Check scope/include_patterns for your MedGemma naming."
        )

    logger.info(f"LoRA scope={scope}, coverage={coverage}")
    logger.info(f"LoRA include_patterns={include_patterns}")
    logger.info(f"LoRA exclude_patterns={exclude_patterns}")
    logger.info(f"LoRA target leaf names: {target_modules}")

    peft = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
        task_type=str(lora_cfg.get("task_type", "FEATURE_EXTRACTION")),
    )
    wrapped = get_peft_model(base_model, peft)

    try:
        wrapped.print_trainable_parameters()
    except Exception:
        pass

    return wrapped


# ---------------- Regression: collator + model ---------------- #

class MOSDataCollator:
    def __init__(self, processor, dummy_text: str, mos_key: str = "mos_score"):
        self.processor = processor
        self.dummy_text = dummy_text
        self.mos_key = mos_key

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [Image.open(ex["image_path"]).convert("RGB") for ex in examples]
        mos = []
        for ex in examples:
            md = ex.get("metadata", {}) or {}
            mos.append(float(md[self.mos_key]))
        texts = [self.dummy_text for _ in examples]
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        batch["labels"] = torch.tensor(mos, dtype=torch.float32)
        return batch


def infer_hidden_size(backbone: nn.Module) -> int:
    cfg = getattr(backbone, "config", None)
    if cfg is None:
        raise ValueError("Model config not found; cannot infer hidden size.")
    for attr in ("hidden_size", "d_model"):
        if hasattr(cfg, attr):
            v = getattr(cfg, attr)
            if isinstance(v, int):
                return v
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    raise ValueError("Could not infer hidden size from config.")


class VLMForMOSRegression(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, loss_type: str, huber_delta: float,
                 mos_min: float, mos_max: float):
        super().__init__()
        self.backbone = backbone
        self.reg_head = nn.Linear(hidden_size, 1)
        self.loss_type = loss_type
        self.mos_min = mos_min
        self.mos_max = mos_max
        self.huber = nn.SmoothL1Loss(beta=huber_delta)

    def forward(self, labels: Optional[torch.Tensor] = None, **inputs):
        outputs = self.backbone(**inputs)

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            h = outputs.last_hidden_state
            pooled = h[:, -1, :]
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            h = outputs[0]
            pooled = h[:, -1, :]
        else:
            raise RuntimeError("No last_hidden_state found; adjust pooling for your MedGemma outputs.")

        pred = self.reg_head(pooled).squeeze(-1)
        pred = torch.clamp(pred, self.mos_min, self.mos_max)

        loss = None
        if labels is not None:
            if self.loss_type == "mse":
                loss = torch.mean((pred - labels) ** 2)
            else:
                loss = self.huber(pred, labels)

        return {"loss": loss, "predictions": pred}


# ---------------- Format SFT: dataset building + collator ---------------- #

def build_format_sft_dataset(ds: Dataset, fmt_cfg: Dict[str, Any]) -> Dataset:
    """
    Build a TRL SFT dataset that teaches the response format only.
    We create a constant (minimal) instruction and deterministic assistant response.
    """
    fmt_cfg = fmt_cfg or {}
    template = fmt_cfg.get(
        "assistant_template",
        '<answer>{"rating": {rating:.3f}}</answer>'
    )
    # Optional: include explanation key as well if you want
    include_expl = bool(fmt_cfg.get("include_explanation", False))
    if include_expl:
        template = fmt_cfg.get(
            "assistant_template",
            '<answer>{"rating": {rating:.3f}, "explanation": "MOS prediction"}</answer>'
        )

    system_text = fmt_cfg.get("system_text", "")
    user_text = fmt_cfg.get("user_text", "")  # intentionally empty/minimal (not prompt engineering)

    def to_example(ex: Dict[str, Any]) -> Dict[str, Any]:
        mos = float((ex.get("metadata") or {}).get("mos_score"))
        assistant_text = template.format(rating=mos)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
        return {
            "messages": messages,
            "image_path": ex["image_path"],
            "metadata": ex.get("metadata", {}),
        }

    # Optional tiny subset for “short” format run
    max_samples = int(fmt_cfg.get("max_samples", 0))
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds.map(to_example, remove_columns=[c for c in ds.column_names if c not in ("image_path", "metadata")])


class VLMFormatCollator:
    """
    Same idea as your old collator, but loads image_path and uses messages.
    """
    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [
            self.processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
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


# ---------------- Main unified class ---------------- #

class LDCTUnifiedTrainer:
    def __init__(self, config_path: Path):
        self.cfg = load_config(config_path)
        self.config_path = config_path
        self.processor = None
        self.model = None

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None

        self.hf_trainer: Optional[Trainer] = None
        self.trl_trainer: Optional[SFTTrainer] = None

        self._setup_logging()

    def _setup_logging(self):
        Path(self.cfg.logging_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(Path(self.cfg.logging_dir) / "train.log", encoding="utf-8"),
            ],
        )
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Mode: {self.cfg.mode}")

    def _bnb(self) -> Optional[BitsAndBytesConfig]:
        if not (self.cfg.use_4bit or self.cfg.use_8bit):
            return None
        return BitsAndBytesConfig(
            load_in_4bit=self.cfg.use_4bit,
            load_in_8bit=self.cfg.use_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def load_data(self):
        self.train_ds, self.val_ds = load_train_val(self.cfg)

        # minimal checks for your pipeline
        for ds_name, ds in [("train", self.train_ds), ("val", self.val_ds)]:
            if "image_path" not in ds.column_names:
                raise ValueError(f"{ds_name} dataset missing 'image_path'. Columns={ds.column_names}")
            if "metadata" not in ds.column_names:
                raise ValueError(f"{ds_name} dataset missing 'metadata'. Columns={ds.column_names}")

        logger.info(f"Train={len(self.train_ds)} | Val={len(self.val_ds)}")

    # -------- regression mode -------- #

    def setup_regression(self):
        reg = self.cfg.reg or {}
        dummy_text = reg.get("dummy_text", "")
        loss_type = str(reg.get("loss_type", "mse")).lower()
        huber_delta = float(reg.get("huber_delta", 0.5))
        mos_min = float(reg.get("mos_min", 0.0))
        mos_max = float(reg.get("mos_max", 4.0))

        self.processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)

        backbone = AutoModel.from_pretrained(
            self.cfg.model_name,
            device_map="auto",
            quantization_config=self._bnb(),
            trust_remote_code=True,
        )

        if self.cfg.gradient_checkpointing:
            try:
                backbone.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled.")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")

        hidden_size = infer_hidden_size(backbone)
        base_model = VLMForMOSRegression(
            backbone=backbone,
            hidden_size=hidden_size,
            loss_type=loss_type,
            huber_delta=huber_delta,
            mos_min=mos_min,
            mos_max=mos_max,
        )

        # Apply LoRA / full finetune choice
        self.model = build_lora(base_model, self.cfg.lora)

        args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            logging_dir=self.cfg.logging_dir,
            num_train_epochs=float(reg.get("num_train_epochs", 3)),
            per_device_train_batch_size=int(reg.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(reg.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(reg.get("gradient_accumulation_steps", 8)),
            learning_rate=float(reg.get("learning_rate", 1e-4)),
            weight_decay=float(reg.get("weight_decay", 0.01)),
            warmup_ratio=float(reg.get("warmup_ratio", 0.1)),
            logging_steps=int(reg.get("logging_steps", 10)),
            save_strategy=str(reg.get("save_strategy", "steps")),
            save_steps=int(reg.get("save_steps", 200)),
            evaluation_strategy=str(reg.get("evaluation_strategy", "steps")),
            eval_steps=int(reg.get("eval_steps", 200)),
            save_total_limit=int(reg.get("save_total_limit", 2)),
            max_grad_norm=float(reg.get("max_grad_norm", 1.0)),
            fp16=bool(reg.get("fp16", self.cfg.fp16)),
            bf16=bool(reg.get("bf16", self.cfg.bf16)),
            report_to=["tensorboard"] if reg.get("report_to", "tensorboard") != "none" else [],
            remove_unused_columns=False,
        )

        collator = MOSDataCollator(self.processor, dummy_text=dummy_text, mos_key="mos_score")

        self.hf_trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            data_collator=collator,
        )
        self.hf_trainer.add_callback(ProcessorSaveCallback(self.processor))

    def run_regression(self):
        assert self.hf_trainer is not None
        out = self.hf_trainer.train()
        self.hf_trainer.save_model(self.cfg.output_dir)
        self.processor.save_pretrained(self.cfg.output_dir)
        return out.metrics if hasattr(out, "metrics") else {}

    # -------- format_sft mode -------- #

    def setup_format_sft(self):
        fmt = self.cfg.fmt or {}

        self.processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            device_map="auto",
            quantization_config=self._bnb(),
            trust_remote_code=True,
        )

        if self.cfg.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled.")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")

        # Apply LoRA / full finetune choice
        self.model = build_lora(model, self.cfg.lora)

        # Build format dataset from existing image_path + mos
        tr_ds = build_format_sft_dataset(self.train_ds, fmt)
        va_ds = build_format_sft_dataset(self.val_ds, fmt)

        max_len = int(fmt.get("max_length", 1024))
        collator = VLMFormatCollator(self.processor, max_length=max_len)

        args = SFTConfig(
            output_dir=self.cfg.output_dir,
            logging_dir=self.cfg.logging_dir,
            num_train_epochs=float(fmt.get("num_train_epochs", 1.0)),
            per_device_train_batch_size=int(fmt.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(fmt.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(fmt.get("gradient_accumulation_steps", 4)),
            learning_rate=float(fmt.get("learning_rate", 2e-4)),
            weight_decay=float(fmt.get("weight_decay", 0.0)),
            warmup_ratio=float(fmt.get("warmup_ratio", 0.1)),
            logging_steps=int(fmt.get("logging_steps", 10)),
            save_steps=int(fmt.get("save_steps", 200)),
            eval_steps=int(fmt.get("eval_steps", 200)),
            save_total_limit=int(fmt.get("save_total_limit", 2)),
            remove_unused_columns=False,
            dataset_kwargs=fmt.get("dataset_kwargs", {"skip_prepare_dataset": True}),
            optim=str(fmt.get("optim", "paged_adamw_8bit")),
            lr_scheduler_type=str(fmt.get("lr_scheduler_type", "cosine")),
            fp16=bool(fmt.get("fp16", self.cfg.fp16)),
            bf16=bool(fmt.get("bf16", self.cfg.bf16)),
            max_grad_norm=float(fmt.get("max_grad_norm", 1.0)),
            report_to=["tensorboard"] if fmt.get("report_to", "tensorboard") != "none" else [],
            label_pad_token_id=-100,
        )

        self.trl_trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=tr_ds,
            eval_dataset=va_ds,
            data_collator=collator,
            processing_class=self.processor,
        )
        self.trl_trainer.add_callback(ProcessorSaveCallback(self.processor))

    def run_format_sft(self):
        assert self.trl_trainer is not None
        out = self.trl_trainer.train()
        self.trl_trainer.save_model(self.cfg.output_dir)
        self.processor.save_pretrained(self.cfg.output_dir)
        return out.metrics if hasattr(out, "metrics") else {}

    # -------- run -------- #

    def run(self) -> Dict[str, Any]:
        logger.info("=== PIPELINE START ===")
        self.load_data()

        if self.cfg.mode == "regression":
            self.setup_regression()
            metrics = self.run_regression()
        elif self.cfg.mode == "format_sft":
            self.setup_format_sft()
            metrics = self.run_format_sft()
        else:
            raise ValueError("mode must be 'regression' or 'format_sft'")

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        logger.info("=== PIPELINE END ===")
        return {"mode": self.cfg.mode, "metrics": metrics, "output_dir": self.cfg.output_dir}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/train_unified.json"))
    args = ap.parse_args()
    LDCTUnifiedTrainer(args.config).run()


if __name__ == "__main__":
    main()
