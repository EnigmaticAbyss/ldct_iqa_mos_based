# src/models/mos_regression.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("mos_regression")


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
    loss_type: str = "mse"     # "mse" | "huber"
    huber_delta: float = 0.5
    mos_min: float = 0.0
    mos_max: float = 4.0


class MOSHead(nn.Module):
    """
    Simple regression head: hidden -> hidden/2 -> 1
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        mid = max(64, hidden_size // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Linear(mid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    ):
        super().__init__()
        self.backbone = backbone
        self.head = MOSHead(hidden_size)

        self.loss_type = loss_type.lower().strip()
        self.huber_delta = float(huber_delta)
        self.mos_min = float(mos_min)
        self.mos_max = float(mos_max)

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
        labels = batch.pop("labels", None)

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
