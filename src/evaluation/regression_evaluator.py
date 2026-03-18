# src/evaluation/regression_evaluator.py

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor

from src.data.loaders import DatasetLoader
from src.models.mos_regression import VLMForMOSRegression
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.plotting import save_scatter_plot, save_error_histogram


logger = logging.getLogger("regression_evaluator")


class RegressionEvaluator:
    """
    Evaluates MOS regression models.

    Pipeline:
        image → model → predicted MOS → metrics
    """

    def __init__(
        self,
        model_dir: str,
        data_dir: str = "datasets/processed",
        use_jsonl: bool = False,
        device: str | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = data_dir
        self.use_jsonl = use_jsonl

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: VLMForMOSRegression | None = None
        self.processor: AutoProcessor | None = None

    # -------------------------------------------------------
    # Load model
    # -------------------------------------------------------

    def load_model(self):

        logger.info(f"Loading regression model from {self.model_dir}")

        self.model = VLMForMOSRegression.from_pretrained(self.model_dir)
        self.processor = AutoProcessor.from_pretrained(self.model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    # -------------------------------------------------------
    # Load test dataset
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # Predict MOS
    # -------------------------------------------------------

    def predict(self, dataset: Dataset) -> List[float]:

        preds: List[float] = []

        for sample in dataset:

            image = Image.open(sample["image_path"]).convert("RGB")

            inputs = self.processor(
                images=image,
                return_tensors="pt",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            mos_pred = outputs["mos"].item()

            preds.append(float(mos_pred))

        return preds

    # -------------------------------------------------------
    # Run evaluation
    # -------------------------------------------------------

    def run(self) -> Dict:

        self.load_model()

        test_ds = self.load_dataset()

        logger.info("Running MOS predictions")

        preds = self.predict(test_ds)

        y_true = [float(x) for x in test_ds["mos_score"]]

        metrics = compute_all_metrics(y_true, preds)

        logger.info("Evaluation results")
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")

        return {
            "metrics": metrics,
            "num_samples": len(test_ds),
        }

    # -------------------------------------------------------
    # Save results
    # -------------------------------------------------------

    def save_results(self, results: Dict, output_path: str):

        path = Path(output_path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved evaluation results → {path}")