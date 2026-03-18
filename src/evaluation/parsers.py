# src/evaluation/parsers.py
from __future__ import annotations

import json
import re
from typing import Optional


def parse_rating_from_json(text: str) -> Optional[float]:
    """
    Extract rating from JSON in generated text.

    Example accepted formats:
        {"rating": 2.7}
        {"rating":2.7}
    """
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            return None

        obj = json.loads(match.group())
        if "rating" in obj:
            return float(obj["rating"])

    except Exception:
        pass

    return None


def parse_rating_from_answer_tag(text: str) -> Optional[float]:
    """
    Support optional format:

    <answer>
    {"rating": 2.7}
    </answer>
    """
    try:
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not match:
            return None

        inner = match.group(1).strip()
        return parse_rating_from_json(inner)

    except Exception:
        return None


def parse_rating_from_number(text: str) -> Optional[float]:
    """
    Fallback: extract first float number in text.

    Example:
        "MOS score is 2.7"
        "Predicted score: 3.1"
    """
    try:
        match = re.search(r"\d+\.\d+", text)
        if match:
            return float(match.group())
    except Exception:
        pass

    return None


def extract_rating(text: str) -> Optional[float]:
    """
    Main unified rating extractor.

    Order:
        1. JSON rating
        2. <answer> JSON
        3. numeric fallback
    """

    if not text:
        return None

    r = parse_rating_from_json(text)
    if r is not None:
        return r

    r = parse_rating_from_answer_tag(text)
    if r is not None:
        return r

    r = parse_rating_from_number(text)
    if r is not None:
        return r

    return None