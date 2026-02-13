import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"  # tiny, fast to download
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
m = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
print("Loaded tiny model in 4-bit. Params:", sum(p.numel() for p in m.parameters()))
print("OK")






#!/usr/bin/env python3
import os, re, json, argparse, logging, random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
from PIL import Image

# Optional HF datasets
try:
    from datasets import Dataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ---------------------------- Logging ----------------------------
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step2_build_trl_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Logger initialized -> {log_path}")
    return logger

# ------------------------ Config structures ----------------------
@dataclass
class SubsetCfg:
    name: str
    images_dir: Path
    mos_json: Path
    out_jsonl: Path

@dataclass
class GroupingCfg:
    mode: str           # "none", "regex", "prefix"
    regex: str = "^(.*?)[_.-]\\d+$"
    prefix_len: int = 0

@dataclass
class ValSplitCfg:
    from_subset: str
    val_ratio: float
    grouping: GroupingCfg
    stratify_by_category: bool

@dataclass
class BuildCfg:
    subsets: List[SubsetCfg]
    system_prompt: str
    assistant_template: str
    output_hf_dataset: bool
    hf_out_root: Path
    random_seed: int
    val_split: ValSplitCfg | None
    data_io_mode: str            # "png_paths" or "tif_direct"
    norm_mode: str               # used if data_io_mode == "tif_direct"
    p_low: float
    p_high: float
    wl: float
    ww: float

# -------------------- Quality bins (closed-open) -----------------
QUALITY_BINS: List[Tuple[str, float, float, bool]] = [
    ("Poor",      0.0, 0.8,  False),
    ("Fair",      0.8, 1.8,  False),
    ("Good",      1.8, 2.8,  False),
    ("High",      2.8, 3.8,  False),
    ("Excellent", 3.8, 4.01, True),
]
def mos_to_category(m: float) -> str:
    for name, lo, hi, inclusive_hi in QUALITY_BINS:
        if (m >= lo) and ((m < hi) or (inclusive_hi and m <= hi)):
            return name
    return "Unknown"

# --------------------- Prompt bank / generators ------------------
USER_PROMPTS = [
    "Please evaluate this LDCT image for overall image quality. Consider factors such as noise, contrast, sharpness, and diagnostic value.",
    "Assess the quality of this low-dose CT scan. Focus on image clarity, noise characteristics, and clinical utility.",
    "Analyze this LDCT image and provide an image quality assessment considering diagnostic adequacy, artifact presence, and overall visual quality.",
    "Evaluate this low-dose CT image for clinical quality, including noise levels, contrast resolution, and structural visibility."
]
REASONING_TEMPLATES = {
    "Poor": [
        "Marked noise severely obscures fine structures; contrast resolution is insufficient for confident diagnosis.",
        "Severe grain and low contrast degrade visibility of anatomy; diagnostic value is very limited."
    ],
    "Fair": [
        "Moderate noise and limited contrast reduce fine detail visibility; basic interpretation is possible with caution.",
        "Usable but suboptimal: noticeable noise and some blurring impact confidence for subtle findings."
    ],
    "Good": [
        "Noise is controlled and contrast adequate; anatomy is well delineated for routine diagnostic use.",
        "Overall clear depiction with manageable noise; fine structures are reasonably preserved."
    ],
    "High": [
        "Minimal noise and strong contrast; sharp delineation of structures supports high diagnostic confidence.",
        "Excellent anatomical definition with low noise and robust soft-tissue contrast."
    ],
    "Excellent": [
        "Outstanding clarity with negligible noise and superb contrast; supports the most confident interpretation.",
        "Exceptionally crisp structures and very low noise; diagnostic quality is maximal."
    ],
}
def gen_user_prompt(i: int) -> str:
    return USER_PROMPTS[i % len(USER_PROMPTS)]
def gen_reasoning(category: str, mos: float) -> str:
    bank = REASONING_TEMPLATES.get(category, REASONING_TEMPLATES["Good"])
    txt = bank[hash((category, round(mos, 2))) % len(bank)]
    return f"{txt} Based on the technical assessment, this image receives a quality score of {mos:.1f} out of 4.0."
def gen_explanation(category: str, mos: float) -> str:
    brief = {
        "Poor":      "Image quality is poor with significant noise/contrast issues.",
        "Fair":      "Image quality is fair with moderate limitations.",
        "Good":      "Image quality is good with controlled noise and adequate contrast.",
        "High":      "Image quality is high with excellent clarity and minimal artifacts.",
        "Excellent": "Image quality is excellent with superior clarity and diagnostic value.",
    }
    return f"{brief.get(category,'Image quality assessment complete.') } Score: {mos:.1f}/4.0."

# ---------------- Normalization helpers (for tif_direct) --------
def load_tif_as_float(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.asarray(im, dtype=np.float32)
    if arr.ndim == 3:  # RGB or multi-channel → grayscale
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
    mx = float(np.max(arr))
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, 0, mx)
    return arr / mx

def load_and_normalize_image(path: Path,
                             norm_mode: str,
                             p_low: float,
                             p_high: float,
                             wl: float,
                             ww: float) -> Image.Image:
    arr = load_tif_as_float(path)
    if norm_mode == "percentile":
        arr01 = norm_percentile(arr, p_low, p_high)
    elif norm_mode == "wlww":
        arr01 = norm_wlww(arr, wl, ww)
    else:  # linear_max
        arr01 = norm_linear_max(arr)
    arr8 = (np.clip(arr01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr8).convert("RGB")

# --------------------- IO utilities ------------------------------
def load_mos(mos_json: Path) -> Dict[str, float]:
    data = json.loads(mos_json.read_text(encoding="utf-8"))
    return {Path(k).stem.lower(): float(v) for k, v in data.items()}

def list_pngs(images_dir: Path) -> Dict[str, Path]:
    images_dir = images_dir.resolve()
    return {p.stem.lower(): p for p in images_dir.glob("*.png")}

def glob_tiffs(images_dir: Path) -> Dict[str, Path]:
    images_dir = images_dir.resolve()
    pats = ["*.tif", "*.TIF", "*.tiff", "*.TIFF"]
    paths = []
    for pat in pats:
        paths.extend(images_dir.glob(pat))
    # map by stem, prefer first occurrence
    stem_to_path: Dict[str, Path] = {}
    for p in sorted(set(paths)):
        stem = p.stem.lower()
        if stem not in stem_to_path:
            stem_to_path[stem] = p
    return stem_to_path

# --------------------- Record builders ---------------------------
def build_records_png_paths(images_dir: Path,
                            mos_json: Path,
                            system_prompt: str,
                            assistant_template: str,
                            logger: logging.Logger) -> List[Dict[str, Any]]:
    mos_map = load_mos(mos_json)
    png_map = list_pngs(images_dir)

    missing_imgs = sorted(set(mos_map.keys()) - set(png_map.keys()))
    extra_imgs   = sorted(set(png_map.keys()) - set(mos_map.keys()))
    if missing_imgs:
        logger.warning(f"{len(missing_imgs)} entries in MOS without PNGs. Examples: {missing_imgs[:5]}")
    if extra_imgs:
        logger.warning(f"{len(extra_imgs)} PNGs without MOS. Examples: {extra_imgs[:5]}")

    common = sorted(set(mos_map.keys()) & set(png_map.keys()))
    rows: List[Dict[str, Any]] = []

    for i, stem in enumerate(tqdm(common, desc=f"Building records (PNG) from {images_dir.name}", leave=True)):
        mos = mos_map[stem]
        cat = mos_to_category(mos)
        user_prompt = gen_user_prompt(i)
        reasoning = gen_reasoning(cat, mos)
        explanation = gen_explanation(cat, mos)
        assistant_text = assistant_template.format(
            reasoning=reasoning, rating=f"{mos:.2f}", explanation=explanation
        )
        messages = [
            {"role": "system",    "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        ]
        rows.append({
            "messages": messages,
            "image_path": str(png_map[stem]),
            "metadata": {"filename": f"{stem}.png", "mos": mos, "category": cat, "stem": stem}
        })
    return rows

def build_records_tif_direct(images_dir: Path,
                             mos_json: Path,
                             system_prompt: str,
                             assistant_template: str,
                             logger: logging.Logger) -> List[Dict[str, Any]]:
    mos_map = load_mos(mos_json)
    tif_map = glob_tiffs(images_dir)

    missing_imgs = sorted(set(mos_map.keys()) - set(tif_map.keys()))
    extra_imgs   = sorted(set(tif_map.keys()) - set(mos_map.keys()))
    if missing_imgs:
        logger.warning(f"{len(missing_imgs)} entries in MOS without TIFFs. Examples: {missing_imgs[:5]}")
    if extra_imgs:
        logger.warning(f"{len(extra_imgs)} TIFFs without MOS. Examples: {[tif_map[k].name for k in extra_imgs[:5]]}")

    common = sorted(set(mos_map.keys()) & set(tif_map.keys()))
    rows: List[Dict[str, Any]] = []

    for i, stem in enumerate(tqdm(common, desc=f"Building records (TIF) from {images_dir.name}", leave=True)):
        mos = mos_map[stem]
        cat = mos_to_category(mos)
        user_prompt = gen_user_prompt(i)
        reasoning = gen_reasoning(cat, mos)
        explanation = gen_explanation(cat, mos)
        assistant_text = assistant_template.format(
            reasoning=reasoning, rating=f"{mos:.2f}", explanation=explanation
        )
        messages = [
            {"role": "system",    "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        ]
        tif_path = tif_map[stem]
        rows.append({
            "messages": messages,
            "image_path": str(tif_path),   # TIF path
            "metadata": {"filename": tif_path.name, "mos": mos, "category": cat, "stem": stem}
        })
    return rows

def write_jsonl(out_jsonl: Path, rows: List[Dict[str, Any]], logger: logging.Logger):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote JSONL -> {out_jsonl} (rows={len(rows)})")

def save_hf_dataset(rows: List[Dict[str, Any]],
                    out_dir: Path,
                    logger: logging.Logger,
                    data_io_mode: str,
                    norm_mode: str,
                    p_low: float,
                    p_high: float,
                    wl: float,
                    ww: float):
    if not HF_AVAILABLE:
        logger.warning("datasets library not available; skipping HF dataset save.")
        return

    messages = [r["messages"] for r in rows]
    metadata = [r["metadata"] for r in rows]

    if data_io_mode == "png_paths":
        # lightweight: store paths, no images
        image_paths = [r["image_path"] for r in rows]
        ds = Dataset.from_dict({
            "messages": messages,
            "image_path": image_paths,
            "metadata": metadata,
        })
    else:
        # tif_direct: load + normalize into PIL images
        images: List[Image.Image] = []
        for r in tqdm(rows, desc=f"Loading/normalizing TIFs for {out_dir.name}", leave=False):
            img = load_and_normalize_image(
                Path(r["image_path"]), norm_mode, p_low, p_high, wl, ww
            )
            images.append(img)
        ds = Dataset.from_dict({
            "messages": messages,
            "images":   images,
            "metadata": metadata,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    logger.info(f"Saved HF dataset -> {out_dir} (rows={len(ds)})")

# --------------------- Grouping / Splitting ----------------------
@dataclass
class TmpGrouping:
    mode: str
    regex: str = "^(.*?)[_.-]\\d+$"
    prefix_len: int = 0

def make_group_id(stem: str, grouping: TmpGrouping) -> str:
    s = stem.lower()
    if grouping.mode == "none":
        return s
    if grouping.mode == "regex":
        m = re.match(grouping.regex, s)
        return (m.group(1) if m and m.groups() else s)
    if grouping.mode == "prefix":
        n = max(0, int(grouping.prefix_len))
        return s[:n] if n > 0 else s
    return s

def stratified_group_split(rows: List[Dict[str, Any]],
                           val_ratio: float,
                           grouping: TmpGrouping,
                           stratify_by_category: bool,
                           seed: int,
                           logger: logging.Logger) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(seed)

    # group rows
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        gid = make_group_id(r["metadata"]["stem"], grouping)
        by_group.setdefault(gid, []).append(r)

    groups = list(by_group.keys())
    random.shuffle(groups)

    if not stratify_by_category:
        target_val = int(round(len(rows) * val_ratio))
        val_groups, train_groups, val_count = [], [], 0
        for g in groups:
            if val_count < target_val:
                val_groups.append(g)
                val_count += len(by_group[g])
            else:
                train_groups.append(g)
        val_rows   = [r for g in val_groups   for r in by_group[g]]
        train_rows = [r for g in train_groups for r in by_group[g]]
        return train_rows, val_rows

    # category-aware (approximate)
    cat_total: Dict[str, int] = {}
    for r in rows:
        c = r["metadata"]["category"]
        cat_total[c] = cat_total.get(c, 0) + 1
    cat_target = {c: int(round(v * val_ratio)) for c, v in cat_total.items()}
    cat_alloc: Dict[str, int] = {c: 0 for c in cat_total}

    val_groups, train_groups = [], []
    for g in groups:
        g_cats: Dict[str, int] = {}
        for r in by_group[g]:
            c = r["metadata"]["category"]
            g_cats[c] = g_cats.get(c, 0) + 1

        under = sum(max(0, cat_target[c] - cat_alloc[c]) for c in cat_target)
        benefit = sum(min(g_cats.get(c, 0), max(0, cat_target[c] - cat_alloc[c])) for c in cat_target)
        if under > 0 and benefit > 0:
            val_groups.append(g)
            for c, k in g_cats.items():
                cat_alloc[c] = cat_alloc.get(c, 0) + k
        else:
            train_groups.append(g)

    val_rows   = [r for g in val_groups   for r in by_group[g]]
    train_rows = [r for g in train_groups for r in by_group[g]]

    logger.info(f"Stratified group split targets: {cat_target}")
    got_val: Dict[str, int] = {}
    for r in val_rows:
        c = r["metadata"]["category"]
        got_val[c] = got_val.get(c, 0) + 1
    logger.info(f"Stratified group split achieved: {got_val}")

    return train_rows, val_rows

# ------------------------------ Main -----------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical assistant specialized in Low-Dose CT image quality assessment. "
    "Provide careful, concise reasoning and a final quality score."
)
DEFAULT_ASSISTANT_TEMPLATE = (
    "Reasoning:\n{reasoning}\n\n"
    "Final Rating (MOS): {rating}\n"
    "Explanation: {explanation}"
)

def parse_config(cfg_path: Path) -> BuildCfg:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    subsets = [SubsetCfg(
        name=s["name"],
        images_dir=Path(s["images_dir"]),
        mos_json=Path(s["mos_json"]),
        out_jsonl=Path(s["out_jsonl"]),
    ) for s in cfg["subsets"]]

    val_cfg = None
    if "val_split" in cfg and cfg["val_split"]:
        g = cfg["val_split"]["grouping"]
        val_cfg = ValSplitCfg(
            from_subset=cfg["val_split"]["from_subset"],
            val_ratio=float(cfg["val_split"]["val_ratio"]),
            grouping=GroupingCfg(
                mode=g.get("mode", "none"),
                regex=g.get("regex", "^(.*?)[_.-]\\d+$"),
                prefix_len=int(g.get("prefix_len", 0)),
            ),
            stratify_by_category=bool(cfg["val_split"].get("stratify_by_category", True)),
        )

    return BuildCfg(
        subsets=subsets,
        system_prompt=cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        assistant_template=cfg.get("assistant_template", DEFAULT_ASSISTANT_TEMPLATE),
        output_hf_dataset=bool(cfg.get("output_hf_dataset", True)),
        hf_out_root=Path(cfg.get("hf_out_root", "data/processed")),
        random_seed=int(cfg.get("random_seed", 42)),
        val_split=val_cfg,
        data_io_mode=cfg.get("data_io_mode", "png_paths"),
        norm_mode=cfg.get("mode", "linear_max"),
        p_low=float(cfg.get("p_low", 0.5)),
        p_high=float(cfg.get("p_high", 99.5)),
        wl=float(cfg.get("wl", 40.0)),
        ww=float(cfg.get("ww", 400.0)),
    )

def main():
    ap = argparse.ArgumentParser(
        description="Step 2: Build TRL-ready datasets from PNGs or directly from TIFFs."
    )
    ap.add_argument("--config", type=Path, default=Path("config/step2_build_dataset.json"),
                    help="JSON config defining subsets and output options.")
    args = ap.parse_args()

    cfg = parse_config(args.config)
    log_path = Path("data/processed/_step2.log")
    logger = setup_logger(log_path)

    random.seed(cfg.random_seed)

    all_stats = {"subsets": {}, "bins": {name: 0 for name, *_ in QUALITY_BINS}}
    built_rows: Dict[str, List[Dict[str, Any]]] = {}

    # 1) Build rows per subset (train pool + test)
    for subset in cfg.subsets:
        logger.info(f"[{subset.name}] images_dir={subset.images_dir}")
        logger.info(f"[{subset.name}] mos_json={subset.mos_json}")
        logger.info(f"[{subset.name}] out_jsonl={subset.out_jsonl}")
        logger.info(f"[{subset.name}] data_io_mode={cfg.data_io_mode}")

        if cfg.data_io_mode == "tif_direct":
            rows = build_records_tif_direct(
                images_dir=subset.images_dir,
                mos_json=subset.mos_json,
                system_prompt=cfg.system_prompt,
                assistant_template=cfg.assistant_template,
                logger=logger
            )
        else:
            rows = build_records_png_paths(
                images_dir=subset.images_dir,
                mos_json=subset.mos_json,
                system_prompt=cfg.system_prompt,
                assistant_template=cfg.assistant_template,
                logger=logger
            )

        built_rows[subset.name] = rows
        write_jsonl(subset.out_jsonl, rows, logger)
        if cfg.output_hf_dataset:
            save_hf_dataset(
                rows,
                cfg.hf_out_root / f"{subset.name}_dataset",
                logger,
                cfg.data_io_mode,
                cfg.norm_mode,
                cfg.p_low,
                cfg.p_high,
                cfg.wl,
                cfg.ww,
            )

        # per-subset stats
        cat_counts: Dict[str, int] = {}
        for r in rows:
            c = r["metadata"]["category"]
            cat_counts[c] = cat_counts.get(c, 0) + 1
        all_stats["subsets"][subset.name] = {"num_rows": len(rows), "category_distribution": cat_counts}
        for k, v in cat_counts.items():
            all_stats["bins"][k] = all_stats["bins"].get(k, 0) + v

    # 2) Create VAL from TRAIN pool (slice-wise or group-wise depending on config)
    if cfg.val_split is not None:
        pool_name = cfg.val_split.from_subset
        if pool_name not in built_rows:
            logger.error(f"val_split.from_subset='{pool_name}' not found in subsets.")
        else:
            pool_rows = built_rows[pool_name]
            logger.info(f"[val_split] splitting from subset '{pool_name}' with {len(pool_rows)} rows...")

            grouping = TmpGrouping(
                mode=cfg.val_split.grouping.mode,
                regex=cfg.val_split.grouping.regex,
                prefix_len=cfg.val_split.grouping.prefix_len
            )
            train_rows, val_rows = stratified_group_split(
                rows=pool_rows,
                val_ratio=cfg.val_split.val_ratio,
                grouping=grouping,
                stratify_by_category=cfg.val_split.stratify_by_category,
                seed=cfg.random_seed,
                logger=logger
            )

            # write split outputs
            train_out = Path("data/processed/train/iqa_sft_train.jsonl")
            val_out   = Path("data/processed/val/iqa_sft_val.jsonl")
            write_jsonl(train_out, train_rows, logger)
            write_jsonl(val_out,   val_rows,   logger)

            if cfg.output_hf_dataset:
                save_hf_dataset(
                    train_rows,
                    cfg.hf_out_root / "train_dataset",
                    logger,
                    cfg.data_io_mode,
                    cfg.norm_mode,
                    cfg.p_low,
                    cfg.p_high,
                    cfg.wl,
                    cfg.ww,
                )
                save_hf_dataset(
                    val_rows,
                    cfg.hf_out_root / "val_dataset",
                    logger,
                    cfg.data_io_mode,
                    cfg.norm_mode,
                    cfg.p_low,
                    cfg.p_high,
                    cfg.wl,
                    cfg.ww,
                )

            # stats entries for final train/val
            def _count(rows: List[Dict[str, Any]]):
                cc: Dict[str, int] = {}
                for r in rows:
                    cc[r["metadata"]["category"]] = cc.get(r["metadata"]["category"], 0) + 1
                return cc
            all_stats["subsets"]["train"] = {"num_rows": len(train_rows), "category_distribution": _count(train_rows)}
            all_stats["subsets"]["val"]   = {"num_rows": len(val_rows),   "category_distribution": _count(val_rows)}

    # 3) global stats
    stats_path = Path("data/processed/dataset_stats.json")
    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    logger.info(f"Wrote stats -> {stats_path}")

if __name__ == "__main__":
    main()
