from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------- Logging ---------------- #

def setup_logger(log_path: Path, level: str = "INFO") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"preprocess::{log_path.stem}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    logger.handlers = []  # avoid duplicates

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging to: {log_path}")
    return logger


# ---------------- Helpers ---------------- #

def load_mos_map(path: Path) -> Dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def find_existing_file(tif_dir: Path, filename: str) -> Optional[Path]:
    cand = tif_dir / filename
    if cand.exists():
        return cand

    stem = Path(filename).stem
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        cand2 = tif_dir / (stem + ext)
        if cand2.exists():
            return cand2

    lower = filename.lower()
    for p in tif_dir.glob("*"):
        if p.name.lower() == lower:
            return p

    return None


def read_tiff_as_gray_array(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def normalize_to_uint8(arr: np.ndarray, norm: str, p_low: float, p_high: float) -> np.ndarray:
    arr = arr.astype(np.float32)

    if norm == "minmax":
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    elif norm == "percentile":
        lo = float(np.percentile(arr, p_low))
        hi = float(np.percentile(arr, p_high))
    else:
        raise ValueError("norm must be one of: minmax | percentile")

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)

    x = (arr - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def to_rgb_pil(gray8: np.ndarray, resize: bool, w: int, h: int) -> Image.Image:
    img = Image.fromarray(gray8, mode="L").convert("RGB")
    if resize:
        img = img.resize((w, h), Image.Resampling.BICUBIC)
    return img


def write_jsonl(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_train_val(rows: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)

    val_n = int(round(len(rows) * val_ratio))
    val_idx = set(idx[:val_n].tolist())

    train_rows, val_rows = [], []
    for i, r in enumerate(rows):
        (val_rows if i in val_idx else train_rows).append(r)
    return train_rows, val_rows


# ---------------- Per-split processing ---------------- #

def build_rows_keep_tif(tif_dir: Path, mos_map: Dict[str, float], logger: logging.Logger) -> Tuple[List[Dict], int]:
    rows = []
    missing = 0
    for fname, mos in mos_map.items():
        src = find_existing_file(tif_dir, fname)
        if src is None:
            missing += 1
            continue
        rows.append({"image_path": str(src.as_posix()), "mos_score": float(mos)})
    logger.info(f"[keep_tif] rows={len(rows)} missing={missing}")
    return rows, missing


def build_rows_to_png(
    tif_dir: Path,
    mos_map: Dict[str, float],
    png_dir: Path,
    logger: logging.Logger,
    norm: str,
    p_low: float,
    p_high: float,
    resize: bool,
    resize_w: int,
    resize_h: int,
) -> Tuple[List[Dict], int, int]:
    png_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = 0
    failed = 0

    for fname, mos in tqdm(list(mos_map.items()), desc=f"TIFF -> PNG ({tif_dir.name})"):
        src = find_existing_file(tif_dir, fname)
        if src is None:
            missing += 1
            continue

        dst = png_dir / f"{src.stem}.png"
        try:
            arr = read_tiff_as_gray_array(src)
            gray8 = normalize_to_uint8(arr, norm=norm, p_low=p_low, p_high=p_high)
            img = to_rgb_pil(gray8, resize=resize, w=resize_w, h=resize_h)
            img.save(dst, format="PNG")
            rows.append({"image_path": str(dst.as_posix()), "mos_score": float(mos)})
        except Exception as e:
            failed += 1
            logger.warning(f"Failed {src.name}: {e}")

    logger.info(f"[to_png] rows={len(rows)} missing={missing} failed={failed} png_dir={png_dir}")
    return rows, missing, failed


def process_split(name: str, split_cfg: Dict[str, Any], global_cfg: Dict[str, Any], logger: logging.Logger):
    tif_dir = Path(split_cfg["tif_dir"])
    mos_json = Path(split_cfg["mos_json"])
    out_jsonl = Path(split_cfg["out_jsonl"])

    mode = split_cfg.get("mode", global_cfg.get("mode", "keep_tif"))
    png_dir = split_cfg.get("png_dir", global_cfg.get("png_dir"))

    norm = split_cfg.get("norm", global_cfg.get("norm", "percentile"))
    p_low = float(split_cfg.get("p_low", global_cfg.get("p_low", 0.5)))
    p_high = float(split_cfg.get("p_high", global_cfg.get("p_high", 99.5)))

    resize = bool(split_cfg.get("resize", global_cfg.get("resize", False)))
    resize_w = int(split_cfg.get("resize_w", global_cfg.get("resize_w", 512)))
    resize_h = int(split_cfg.get("resize_h", global_cfg.get("resize_h", 512)))

    make_val_split = bool(split_cfg.get("make_val_split", False))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    seed = int(split_cfg.get("seed", 42))

    logger.info("=" * 60)
    logger.info(f"[{name}] tif_dir={tif_dir}")
    logger.info(f"[{name}] mos_json={mos_json}")
    logger.info(f"[{name}] out_jsonl={out_jsonl}")
    logger.info(f"[{name}] mode={mode} norm={norm} p_low={p_low} p_high={p_high} resize={resize}")

    if not tif_dir.exists():
        raise FileNotFoundError(f"[{name}] tif_dir not found: {tif_dir}")
    if not mos_json.exists():
        raise FileNotFoundError(f"[{name}] mos_json not found: {mos_json}")

    mos_map = load_mos_map(mos_json)
    if not mos_map:
        raise ValueError(f"[{name}] MOS JSON loaded empty/invalid: {mos_json}")

    if mode == "keep_tif":
        rows, missing = build_rows_keep_tif(tif_dir, mos_map, logger)
        failed = 0
    elif mode == "to_png":
        if not png_dir:
            raise ValueError(f"[{name}] png_dir is required when mode=to_png")
        rows, missing, failed = build_rows_to_png(
            tif_dir=tif_dir,
            mos_map=mos_map,
            png_dir=Path(png_dir),
            logger=logger,
            norm=norm,
            p_low=p_low,
            p_high=p_high,
            resize=resize,
            resize_w=resize_w,
            resize_h=resize_h,
        )
    else:
        raise ValueError(f"[{name}] mode must be keep_tif or to_png")

    if not rows:
        logger.warning(f"[{name}] No rows produced. Check MOS filenames vs folder.")
        write_jsonl([], out_jsonl)
        return

    # Only create val split from train
    if name.lower() == "train" and make_val_split:
        train_rows, val_rows = split_train_val(rows, val_ratio=val_ratio, seed=seed)

        train_out = out_jsonl
        val_out = out_jsonl.with_name(out_jsonl.stem.replace("train", "val") + out_jsonl.suffix)

        write_jsonl(train_rows, train_out)
        write_jsonl(val_rows, val_out)

        logger.info(f"[{name}] wrote train_jsonl={train_out} ({len(train_rows)})")
        logger.info(f"[{name}] wrote val_jsonl={val_out} ({len(val_rows)})")
    else:
        write_jsonl(rows, out_jsonl)
        logger.info(f"[{name}] wrote jsonl={out_jsonl} ({len(rows)})")

    logger.info(f"[{name}] missing={missing} failed={failed}")


# ---------------- Main ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    log_dir = Path(cfg.get("log_dir", "logs/preprocess"))
    log_level = cfg.get("log_level", "INFO")
    logger = setup_logger(log_dir / "preprocess_all.log", level=log_level)

    splits = cfg.get("splits", {})
    if not splits:
        raise ValueError("Config must contain: splits: { train: {...}, test: {...} }")

    # Run in order if present
    for split_name in ["train", "test"]:
        if split_name in splits:
            process_split(split_name, splits[split_name], cfg, logger)

    # Any additional splits
    for split_name, split_cfg in splits.items():
        if split_name not in ("train", "test"):
            process_split(split_name, split_cfg, cfg, logger)

    logger.info("✅ ALL preprocessing done.")


if __name__ == "__main__":
    main()
