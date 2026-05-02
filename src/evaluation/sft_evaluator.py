
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from peft import PeftModel
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from src.data.loaders import DatasetLoader
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.parsers import extract_rating
from src.evaluation.io import save_predictions_csv
from src.evaluation.plotting import save_scatter_plot, save_error_histogram

logger = logging.getLogger("sft_evaluator")


class SFTEvaluator:
    """
    Evaluates TRL-SFT models.

    Pipeline:
        image -> prompt -> generate -> parse MOS -> metrics
    """

    def __init__(
        self,
        model_dir: str,
        data_dir: str = "datasets/processed",
        use_jsonl: bool = False,
        system_prompt: str = "You are a medical image quality assessment assistant.",
        user_text: str = "Predict MOS score.",
        device: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = data_dir
        self.use_jsonl = use_jsonl

        self.system_prompt = system_prompt
        self.user_text = user_text

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: AutoModelForImageTextToText | None = None
        self.processor: AutoProcessor | None = None

    def load_model(self):
        logger.info(f"Loading SFT model from {self.model_dir}")
        base_model_name = "google/medgemma-1.5-4b-it"   # replace if different

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
    )

        self.model = PeftModel.from_pretrained(
            base_model,
            self.model_dir,
    )

        self.processor = AutoProcessor.from_pretrained(
        base_model_name,
            trust_remote_code=True,
    )

        self.model.eval()
        logger.info("Base model + adapter + processor loaded"
                 )
    def load_dataset(self) -> Dataset:
        loader = DatasetLoader(data_dir=self.data_dir, use_jsonl=self.use_jsonl)

        test_ds = loader.load_test()

        DatasetLoader.require_columns(
            test_ds,
            ["image_path", "mos_score"],
            name="test",
        )

        logger.info(f"Loaded test dataset | samples={len(test_ds)}")
        return test_ds

    def generate_predictions(self, dataset: Dataset):
        predictions: List[Optional[float]] = []
        raw_outputs: List[str] = []

        model_device = next(self.model.parameters()).device

        for sample in dataset:
            image = Image.open(sample["image_path"]).convert("RGB")

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.user_text},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=text,
                images=[[image]],
                return_tensors="pt",
                padding=True,
            )

            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    use_cache=True,
                )

            gen_text = self.processor.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            raw_outputs.append(gen_text)
            rating = extract_rating(gen_text)
            predictions.append(rating)

            torch.cuda.empty_cache()

        return predictions, raw_outputs

    def run(self) -> Dict:
        self.load_model()
        test_ds = self.load_dataset()

        logger.info("Generating predictions")
        preds, outputs = self.generate_predictions(test_ds)

        y_true = [float(x) for x in test_ds["mos_score"]]
        metrics = compute_all_metrics(y_true, preds)

        logger.info("Evaluation results")
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")

        save_scatter_plot(y_true, preds, "output/eval/sft_scatter.png")
        save_error_histogram(y_true, preds, "output/eval/sft_error_hist.png")

        image_paths = list(test_ds["image_path"])
        save_predictions_csv(
            image_paths=image_paths,
            y_true=y_true,
            y_pred=preds,
            raw_outputs=outputs,
            out_path="outputs/eval/sft_predictions.csv",
        )

        return {
            "metrics": metrics,
            "num_samples": len(test_ds),
            "prediction_failures": preds.count(None),
        }

    def save_results(self, results: Dict, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved evaluation results → {path}")