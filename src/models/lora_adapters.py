# src/models/lora_adapters.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch.nn as nn
from peft import LoraConfig

logger = logging.getLogger("lora_adapters")


def _matches_any(name: str, patterns: Optional[List[str]]) -> bool:
    """True if name matches ANY regex in patterns. If patterns None/empty -> True."""
    if not patterns:
        return True
    return any(re.search(p, name) for p in patterns)


@dataclass
class LoraPlan:
    """
    A resolved LoRA plan:
      - target_modules: leaf names (e.g., q_proj, v_proj, fc1, ...)
      - summary: counts for debugging
    """
    target_modules: List[str]
    summary: Dict[str, int]


def discover_target_leaf_names(
    root: nn.Module,
    include_module_types: Tuple[type, ...],
    include_name_patterns: Optional[List[str]] = None,
    exclude_name_patterns: Optional[List[str]] = None,
) -> LoraPlan:
    """
    Discover PEFT target_modules as LEAF attribute names (not full dotted paths).

    PEFT matches target_modules by substring on module names.
    The safest approach is to collect leaf names like: "q_proj", "v_proj", "fc1", etc.
    """
    leaf_names: List[str] = []
    type_hits = 0
    full_hits = 0

    for full_name, module in root.named_modules():
        if full_name == "":
            continue

        if not isinstance(module, include_module_types):
            continue
        type_hits += 1

        if not _matches_any(full_name, include_name_patterns):
            continue
        if exclude_name_patterns and _matches_any(full_name, exclude_name_patterns):
            continue
        full_hits += 1

        leaf = full_name.split(".")[-1]
        leaf_names.append(leaf)

    # unique + stable order
    seen = set()
    uniq = []
    for x in leaf_names:
        if x not in seen:
            uniq.append(x)
            seen.add(x)

    summary = {
        "modules_of_included_types": type_hits,
        "modules_after_name_filters": full_hits,
        "unique_leaf_targets": len(uniq),
    }

    return LoraPlan(target_modules=uniq, summary=summary)


def pick_submodule_by_scope(model: nn.Module, scope: str) -> nn.Module:
    """
    Returns the module subtree you want to apply LoRA to.

    scope:
      - "llm"   : text model only (if discoverable)
      - "vision": vision encoder only (if discoverable)
      - "both"  : whole model
    """
    scope = scope.lower().strip()

    if scope == "both":
        return model

    # Best-effort: many VLMs expose these common attributes.
    # If MedGemma exposes different names, adapt here once.
    candidates = []
    if scope == "vision":
        candidates = ["vision_tower", "vision_model", "vision_encoder", "visual", "vision"]
    elif scope == "llm":
        candidates = ["language_model", "text_model", "llm", "model", "decoder"]

    for attr in candidates:
        if hasattr(model, attr):
            sub = getattr(model, attr)
            if isinstance(sub, nn.Module):
                logger.info(f"LoRA scope='{scope}' resolved to model.{attr}")
                return sub

    logger.warning(
        f"LoRA scope='{scope}' requested but no known submodule attribute found. "
        f"Falling back to entire model."
    )
    return model


def build_lora_config_from_settings(
    *,
    task_type: str,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: List[str],
) -> LoraConfig:
    """
    Create PEFT LoraConfig.
    """
    if not target_modules:
        raise ValueError("target_modules is empty. LoRA would attach to nothing.")

    return LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias="none",
        task_type=task_type,
    )


def resolve_lora_targets(
    model: nn.Module,
    *,
    scope: str,
    coverage: str,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> LoraPlan:
    """
    Resolve which module leaf names should get LoRA adapters.

    coverage:
      - "linear_only"      : nn.Linear only
      - "linear_and_conv"  : nn.Linear + nn.Conv2d
      - "full_finetune"    : NOT LoRA (handled outside) -> returns empty plan

    include/exclude patterns apply to FULL module names.
    """
    coverage = coverage.lower().strip()

    if coverage == "full_finetune":
        return LoraPlan(target_modules=[], summary={"mode": "full_finetune"})

    include_types: Tuple[type, ...]
    if coverage == "linear_only":
        include_types = (nn.Linear,)
    elif coverage == "linear_and_conv":
        include_types = (nn.Linear, nn.Conv2d)
    else:
        raise ValueError(f"Unknown coverage='{coverage}'. Use linear_only|linear_and_conv|full_finetune")

    subtree = pick_submodule_by_scope(model, scope)

    plan = discover_target_leaf_names(
        subtree,
        include_module_types=include_types,
        include_name_patterns=include_patterns,
        exclude_name_patterns=exclude_patterns,
    )

    logger.info(
        f"LoRA target discovery done. scope={scope}, coverage={coverage}, "
        f"summary={plan.summary}"
    )
    if plan.target_modules:
        logger.info(f"Example LoRA leaf targets: {plan.target_modules[:20]}")
    else:
        logger.warning("No LoRA targets found after discovery.")
    return plan
