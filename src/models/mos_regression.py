# src/models/mos_regression.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModelForImageTextToText as AutoModelClass
except ImportError:
    from transformers import AutoModelForCausalLM as AutoModelClass

logger = logging.getLogger("mos_regression")

REGRESSION_CONFIG_NAME = "regression_config.json"
REGRESSION_WEIGHTS_NAME = "pytorch_model.bin"


def infer_hidden_size(backbone) -> int:
    """
    Best-effort hidden size inference across HF model types.
    Works for most causal LMs used in VLMs.
    """
    cfg = getattr(backbone, "config", None)
    if cfg is None:
        raise ValueError("Backbone has no config; cannot infer hidden_size.")

    for attr in ("hidden_size", "n_embd", "d_model", "dim"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))

    raise ValueError("Could not infer hidden_size from backbone.config.")


@dataclass
class RegressionSettings:
    """
    Configuration values for MOS regression loss and prediction range.

    Attributes:
        loss_type: Regression loss name, either ``mse`` or ``huber``.
        huber_delta: Delta parameter used by Huber loss.
        mos_min: Minimum allowed MOS prediction.
        mos_max: Maximum allowed MOS prediction.
    """

    loss_type: str = "mse"     # "mse" | "huber"
    huber_delta: float = 0.5
    mos_min: float = 0.0
    mos_max: float = 4.0


class MOSHead(nn.Module):
    """
    Simple regression head: hidden -> hidden/2 -> 1

    Args:
        hidden_size: Size of the pooled backbone hidden state.
    """
    def __init__(self, hidden_size: int):
        """Create the two-layer projection from backbone hidden states to MOS."""
        super().__init__()
        mid = max(64, hidden_size // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Linear(mid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict one scalar MOS value per pooled hidden state.

        Args:
            x: Pooled hidden states with shape ``(batch, hidden_size)``.

        Returns:
            MOS logits with shape ``(batch,)``.
        """
        return self.net(x).squeeze(-1)  # (B,)


class VLMForMOSRegression(nn.Module):
    """
    Wrap a causal LM backbone with a regression head.

    Forward expects:
      - input_ids, attention_mask, pixel_values (depends on processor/model)
      - labels: float tensor (B,) with MOS values

    Returns:
      dict(loss=..., mos_pred=...)
    """

    def __init__(
        self,
        backbone,
        hidden_size: int,
        loss_type: str = "mse",
        huber_delta: float = 0.5,
        mos_min: float = 0.0,
        mos_max: float = 4.0,
        base_model_name: Optional[str] = None,
    ):
        """Attach a regression head to a pretrained VLM backbone."""
        super().__init__()
        self.backbone = backbone
        self.head = MOSHead(hidden_size)

        self.loss_type = loss_type.lower().strip()
        self.huber_delta = float(huber_delta)
        self.mos_min = float(mos_min)
        self.mos_max = float(mos_max)
        self.base_model_name = base_model_name or getattr(
            getattr(backbone, "config", None),
            "_name_or_path",
            None,
        )

        # We need hidden states from the LM
        if hasattr(self.backbone, "config"):
            self.backbone.config.output_hidden_states = True

    def _pool_hidden(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Pool token hidden states -> single vector per sample.

        Strategy:
          - If attention_mask exists: take hidden at last non-pad token (common for causal LM)
          - Else: take last token
        """
        # hidden_states: (B, T, H)
        if attention_mask is None:
            return hidden_states[:, -1, :]

        # last non-pad index per row
        lengths = attention_mask.long().sum(dim=1) - 1  # (B,)
        b = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[b, lengths, :]  # (B, H)

    def forward(self, **batch):
        """
        Run the backbone, pool hidden states, and optionally compute loss.

        Args:
            **batch: Processor tensors accepted by the backbone, plus optional
                float ``labels`` containing MOS targets.

        Returns:
            Dictionary containing ``mos_pred`` and, when labels are present, ``loss``.

        Raises:
            ValueError: If ``loss_type`` is not supported.
        """
        labels = batch.pop("labels", None)
        batch.setdefault("output_hidden_states", True)

        outputs = self.backbone(**batch)

        # Hidden states: tuple(layer0..layerN), take last layer
        hs = outputs.hidden_states[-1]  # (B, T, H)
        pooled = self._pool_hidden(hs, batch.get("attention_mask", None))  # (B, H)

        mos_pred = self.head(pooled)  # (B,)
        mos_pred = torch.clamp(mos_pred, self.mos_min, self.mos_max)

        out = {"mos_pred": mos_pred}

        if labels is not None:
            labels = labels.to(mos_pred.device).float()

            if self.loss_type == "mse":
                loss = F.mse_loss(mos_pred, labels)
            elif self.loss_type == "huber":
                loss = F.huber_loss(mos_pred, labels, delta=self.huber_delta)
            else:
                raise ValueError(f"Unknown loss_type='{self.loss_type}' (use mse|huber)")

            out["loss"] = loss

        return out

    def regression_config(self) -> dict:
        """
        Return serializable metadata needed to reload the regression wrapper.

        Returns:
            Dictionary containing base model, loss, and MOS range settings.
        """
        return {
            "base_model_name": self.base_model_name,
            "loss_type": self.loss_type,
            "huber_delta": self.huber_delta,
            "mos_min": self.mos_min,
            "mos_max": self.mos_max,
        }

    def save_pretrained(self, save_directory: str | Path, **kwargs) -> None:
        """
        Save regression metadata and weights using a HF-style directory layout.

        Args:
            save_directory: Directory where config and weights should be written.
            **kwargs: Accepted for Hugging Face API compatibility; unused.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        (save_dir / REGRESSION_CONFIG_NAME).write_text(
            json.dumps(self.regression_config(), indent=2),
            encoding="utf-8",
        )
        torch.save(self.state_dict(), save_dir / REGRESSION_WEIGHTS_NAME)

    @staticmethod
    def _load_regression_config(model_dir: Path) -> dict:
        """Read regression metadata from a model directory when present."""
        config_path = model_dir / REGRESSION_CONFIG_NAME
        if not config_path.exists():
            return {}
        return json.loads(config_path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_state_dict(model_dir: Path) -> Optional[dict]:
        """Load regression weights from PyTorch or safetensors files if available."""
        weights_path = model_dir / REGRESSION_WEIGHTS_NAME
        if weights_path.exists():
            return torch.load(weights_path, map_location="cpu")

        safetensors_path = model_dir / "model.safetensors"
        if safetensors_path.exists():
            from safetensors.torch import load_file

            return load_file(str(safetensors_path), device="cpu")

        return None

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        base_model_name: Optional[str] = None,
        is_peft_adapter: Optional[bool] = None,
        device_map=None,
        quantization_config=None,
        trust_remote_code: bool = True,
        **model_kwargs,
    ):
        """
        Load a regression wrapper from full weights or a PEFT adapter directory.

        Args:
            model_dir: Directory containing regression weights, metadata, or adapter files.
            base_model_name: Optional base model override used for adapter loading.
            is_peft_adapter: Whether ``model_dir`` should be treated as a PEFT adapter.
            device_map: Device map forwarded to the backbone loader.
            quantization_config: Optional bitsandbytes quantization config.
            trust_remote_code: Whether to allow custom Hugging Face model code.
            **model_kwargs: Extra keyword arguments forwarded to ``from_pretrained``.

        Returns:
            A ``VLMForMOSRegression`` instance or PEFT-wrapped regression model.

        Raises:
            ValueError: If an adapter is loaded without a known base model.
        """
        model_dir = Path(model_dir)
        cfg = cls._load_regression_config(model_dir)
        adapter_config = model_dir / "adapter_config.json"
        load_as_adapter = adapter_config.exists() if is_peft_adapter is None else is_peft_adapter

        backbone_name = base_model_name or cfg.get("base_model_name")
        if not backbone_name:
            if load_as_adapter:
                raise ValueError(
                    f"{model_dir} looks like a PEFT adapter but no base model is known. "
                    "Pass base_model_name or make sure regression_config.json was saved."
                )
            backbone_name = str(model_dir)

        backbone = AutoModelClass.from_pretrained(
            backbone_name,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        model = cls(
            backbone=backbone,
            hidden_size=infer_hidden_size(backbone),
            loss_type=cfg.get("loss_type", "mse"),
            huber_delta=cfg.get("huber_delta", 0.5),
            mos_min=cfg.get("mos_min", 0.0),
            mos_max=cfg.get("mos_max", 4.0),
            base_model_name=backbone_name,
        )

        if load_as_adapter:
            from peft import PeftModel

            return PeftModel.from_pretrained(model, str(model_dir))

        state_dict = cls._load_state_dict(model_dir)
        if state_dict is not None:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Missing regression weights while loading {model_dir}: {missing}")
            if unexpected:
                logger.warning(f"Unexpected regression weights while loading {model_dir}: {unexpected}")

        return model
