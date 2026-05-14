from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from src.evaluation.parsers import extract_rating

logger = logging.getLogger("grpo_rewards")


@dataclass
class MOSRewardConfig:
    mos_min: float = 0.0
    mos_max: float = 4.0
    reward_kind: str = "neg_abs_error"
    reward_scale: float = 1.0
    reward_offset: float = 0.0
    missing_reward: float = -4.0
    clamp_prediction: bool = True


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        return _content_to_text(completion.get("content", ""))

    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(_content_to_text(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    return str(completion)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(text for text in texts if text)
    return str(content)


def _as_float_list(values: Optional[Iterable[Any]], expected_len: int) -> list[float]:
    if values is None:
        raise ValueError("MOS reward requires the dataset column 'mos_score'.")
    out = [float(value) for value in values]
    if len(out) != expected_len:
        raise ValueError(f"Expected {expected_len} MOS targets, got {len(out)}.")
    return out


def _score_prediction(
    pred: Optional[float],
    target: float,
    cfg: MOSRewardConfig,
) -> tuple[float, Optional[float], Optional[float]]:
    if pred is None:
        return float(cfg.missing_reward), None, None

    pred_value = float(pred)
    if cfg.clamp_prediction:
        pred_value = min(max(pred_value, cfg.mos_min), cfg.mos_max)

    abs_error = abs(pred_value - target)
    reward_kind = cfg.reward_kind.strip().lower()

    if reward_kind == "neg_abs_error":
        reward = -abs_error
    elif reward_kind == "neg_squared_error":
        reward = -(pred_value - target) ** 2
    elif reward_kind == "bounded_linear":
        mos_range = max(cfg.mos_max - cfg.mos_min, 1e-8)
        reward = max(0.0, 1.0 - abs_error / mos_range)
    else:
        raise ValueError(
            "reward_kind must be one of: neg_abs_error, neg_squared_error, bounded_linear"
        )

    reward = reward * cfg.reward_scale + cfg.reward_offset
    return float(reward), pred_value, float(abs_error)


def make_mos_reward_function(cfg: MOSRewardConfig):
    def mos_score_reward(
        prompts: list[Any],
        completions: list[Any],
        mos_score: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> list[float]:
        targets = _as_float_list(mos_score, len(completions))
        rewards: list[float] = []
        parsed_scores: list[Optional[float]] = []
        abs_errors: list[Optional[float]] = []

        for completion, target in zip(completions, targets):
            text = completion_to_text(completion)
            parsed = extract_rating(text)
            reward, parsed_score, abs_error = _score_prediction(parsed, target, cfg)
            rewards.append(reward)
            parsed_scores.append(parsed_score)
            abs_errors.append(abs_error)

        log_extra = kwargs.get("log_extra")
        if callable(log_extra):
            log_extra("target_mos", targets)
            log_extra("pred_mos", parsed_scores)
            log_extra("abs_error", abs_errors)

        log_metric = kwargs.get("log_metric")
        valid_errors = [err for err in abs_errors if err is not None]
        if callable(log_metric):
            log_metric("reward/parse_rate", len(valid_errors) / len(abs_errors) if abs_errors else 0.0)
            if valid_errors:
                log_metric("reward/mean_abs_error", sum(valid_errors) / len(valid_errors))

        return rewards

    mos_score_reward.__name__ = "mos_score_reward"
    return mos_score_reward
