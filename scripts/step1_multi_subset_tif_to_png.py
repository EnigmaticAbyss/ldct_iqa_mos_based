#!/usr/bin/env python3
import json, glob, logging, argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------- helpers ----------------

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Explicit UTF-8 for file logs; console uses system encoding (avoid unicode glyphs)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Logger initialized -> {log_path}")
    return logger

def load_tif_as_float(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.asarray(im, dtype=np.float32)
    if arr.ndim == 3:  # RGB -> grayscale
        arr = arr.mean(axis=2)
    return arr

def norm_percentile(arr: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo + 1e-8)

def norm_wlww(arr: np.ndarray, wl: float, ww: float) -> np.ndarray:
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo + 1e-8)

def norm_linear_max(arr: np.ndarray) -> np.ndarray:
    """
    Reproduce the original behavior:
    - scale by per-image max only (no min subtraction, no percentile, no fixed window)
    - clip negatives to 0 before scaling (matches old uint8 casting behavior)
    """
    mx = float(np.max(arr))
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, 0, mx)
    return arr / mx  # 0..1

def to_png_rgb01(arr01: np.ndarray) -> Image.Image:
    arr8 = (np.clip(arr01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr8).convert("RGB")

def _glob_tiffs(src_dir: Path) -> List[Path]:
    # Case-insensitive *.tif / *.tiff for Windows/mac/linux
    pats = [str(src_dir / "*.tif"), str(src_dir / "*.TIF"),
            str(src_dir / "*.tiff"), str(src_dir / "*.TIFF")]
    paths = []
    for pat in pats:
        paths.extend(glob.glob(pat))
    # Deduplicate and sort
    uniq = sorted({Path(p) for p in paths})
    return uniq

def convert_subset(
    name: str,
    src_dir: Path,
    mos_json: Path,
    out_dir: Path,
    mode: str,
    p_low: float,
    p_high: float,
    wl: float,
    ww: float,
    png_quality: int,
) -> Dict[str, int]:
    logger = setup_logger(out_dir / "_step1.log")

    # Check src dir
    if not src_dir.exists():
        logger.error(f"[{name}] Source dir does not exist: {src_dir}")
        return {"total_json": 0, "total_fs": 0, "converted": 0, "failed": 0, "missing": 0, "extra": 0}

    # Load MOS mapping
    try:
        mos = json.loads(mos_json.read_text(encoding="utf-8"))
        # Normalize keys: use stem (name without extension), keep a map for debugging
        mos_stem = {Path(k).stem.lower(): float(v) for k, v in mos.items()}
    except Exception as e:
        logger.error(f"[{name}] Failed to read MOS JSON {mos_json}: {e}")
        return {"total_json": 0, "total_fs": 0, "converted": 0, "failed": 0, "missing": 0, "extra": 0}

    # List TIFFs (.tif and .tiff)
    tif_list = _glob_tiffs(src_dir)
    # Build index by stem (lowercased)
    stem_to_paths: Dict[str, List[Path]] = {}
    for p in tif_list:
        stem = p.stem.lower()
        stem_to_paths.setdefault(stem, []).append(p)

    # Warn about duplicate stems (e.g., both 0001.tif and 0001.tiff)
    dups = {s: ps for s, ps in stem_to_paths.items() if len(ps) > 1}
    if dups:
        for s, ps in list(dups.items())[:5]:
            logger.warning(f"[{name}] Duplicate stems for '{s}': {ps}")
        logger.warning(f"[{name}] Found {len(dups)} stems with duplicates. The first path will be used.")

    json_stems = set(mos_stem.keys())
    fs_stems   = set(stem_to_paths.keys())

    missing_in_fs = sorted(json_stems - fs_stems)
    extra_in_fs   = sorted(fs_stems - json_stems)

    if missing_in_fs:
        ex = [f"{s}.(tif|tiff)" for s in missing_in_fs[:5]]
        logger.warning(f"[{name}] {len(missing_in_fs)} MOS entries missing in filesystem. Examples: {ex}")
    if extra_in_fs:
        ex = [str(stem_to_paths[s][0].name) for s in extra_in_fs[:5]]
        logger.warning(f"[{name}] {len(extra_in_fs)} TIFF files not present in MOS JSON. Examples: {ex}")

    # Files to convert = intersection only, sorted numerically if possible
    common = list(json_stems & fs_stems)
    try:
        common.sort(key=lambda s: int(s))  # stems like "0001"
    except ValueError:
        common.sort()  # lexical fallback

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{name}] Converting {len(common)} files from {src_dir} -> {out_dir}")
    if mode == "percentile":
        logger.info(f"[{name}] Normalization: percentile [{p_low}, {p_high}]")
    elif mode == "wlww":
        logger.info(f"[{name}] Normalization: wlww (wl={wl}, ww={ww})")
    elif mode == "linear_max":
        logger.info(f"[{name}] Normalization: linear_max (per-image max only)")
    else:
        logger.error(f"[{name}] Unknown mode: {mode}")
        return {"total_json": 0, "total_fs": 0, "converted": 0, "failed": 0, "missing": 0, "extra": 0}

    converted = failed = 0
    for stem in tqdm(common, desc=f"[{name}] TIFF -> PNG", leave=True):
        src = stem_to_paths[stem][0]  # first if duplicates
        out = out_dir / f"{stem}.png"
        try:
            arr = load_tif_as_float(src)
            if mode == "percentile":
                arr01 = norm_percentile(arr, p_low, p_high)
            elif mode == "wlww":
                arr01 = norm_wlww(arr, wl, ww)
            else:  # linear_max
                arr01 = norm_linear_max(arr)

            to_png_rgb01(arr01).save(out, quality=png_quality)
            converted += 1
        except Exception as e:
            logger.error(f"[{name}] FAIL {src.name}: {e}")
            failed += 1

    logger.info(f"[{name}] === SUMMARY ===")
    logger.info(f"[{name}] SRC_DIR: {src_dir}")
    logger.info(f"[{name}] OUT_DIR: {out_dir}")
    logger.info(f"[{name}] MOS_JSON: {mos_json}")
    logger.info(f"[{name}] Total in JSON (unique stems): {len(json_stems)}")
    logger.info(f"[{name}] Total in FS (unique stems):   {len(fs_stems)}")
    logger.info(f"[{name}] Converted:     {converted}")
    logger.info(f"[{name}] Failed:        {failed}")
    logger.info(f"[{name}] Missing:       {len(missing_in_fs)}")
    logger.info(f"[{name}] Extra:         {len(extra_in_fs)}")

    return {
        "total_json": len(json_stems),
        "total_fs": len(fs_stems),
        "converted": converted,
        "failed": failed,
        "missing": len(missing_in_fs),
        "extra": len(extra_in_fs),
    }

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Step 1: Convert multiple subsets (train/test) of LDCT .tif/.tiff -> 8-bit RGB .png using JSON config."
    )
    ap.add_argument(
        "--config", type=Path, default=Path("config/step1_subsets.json"),
        help="JSON config with normalization params and subset paths."
    )
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    mode = cfg.get("mode", "percentile")
    p_low = float(cfg.get("p_low", 0.5))
    p_high = float(cfg.get("p_high", 99.5))
    wl = float(cfg.get("wl", 40.0))
    ww = float(cfg.get("ww", 400.0))
    png_quality = int(cfg.get("png_quality", 95))

    subsets = cfg["subsets"]

    agg = {}
    for s in subsets:
        name = s["name"]
        res = convert_subset(
            name=name,
            src_dir=Path(s["src_dir"]),
            mos_json=Path(s["mos_json"]),
            out_dir=Path(s["out_dir"]),
            mode=mode,
            p_low=p_low,
            p_high=p_high,
            wl=wl,
            ww=ww,
            png_quality=png_quality,
        )
        agg[name] = res

    # ASCII-only summary for Windows consoles
    print("\n==== STEP 1 (ALL SUBSETS) SUMMARY ====")
    for name, r in agg.items():
        print(
            f"[{name}] Json={r['total_json']} FS={r['total_fs']} "
            f"Converted={r['converted']} Failed={r['failed']} "
            f"Missing={r['missing']} Extra={r['extra']}"
        )

if __name__ == "__main__":
    main()
