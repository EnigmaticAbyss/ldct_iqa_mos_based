# src/evaluation/regression_evaluator.py

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor

from src.data.loaders import DatasetLoader
from src.models.mos_regression import VLMForMOSRegression
from src.trainers.common import build_bnb_config
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.plotting import save_scatter_plot, save_error_histogram
from src.evaluation.io import save_predictions_csv


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
        base_model_name: Optional[str] = None,
        is_peft_adapter: Optional[bool] = None,
        output_dir: str | Path = "output/eval/regression",
        dataset_format: Optional[str] = None,
        test_dataset_dir: Optional[str] = None,
        test_json_path: Optional[str] = None,
        test_jsonl_path: Optional[str] = None,
        prompt_text: str = "Predict MOS score.",
        use_4bit: bool = True,
        use_8bit: bool = False,
        bnb_compute_dtype: str = "bf16",
    ):
        """Store model, dataset, prompt, and quantization settings for evaluation."""
        self.model_dir = Path(model_dir)
        self.base_model_name = base_model_name
        self.is_peft_adapter = is_peft_adapter
        self.output_dir = Path(output_dir)

        self.data_dir = data_dir
        self.use_jsonl = use_jsonl
        self.dataset_format = dataset_format
        self.test_dataset_dir = test_dataset_dir
        self.test_json_path = test_json_path or test_jsonl_path
        self.prompt_text = prompt_text

        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.bnb_compute_dtype = bnb_compute_dtype

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: VLMForMOSRegression | None = None
        self.processor: AutoProcessor | None = None

    # -------------------------------------------------------
    # Load model
    # -------------------------------------------------------

    def load_model(self):
        """
        Load the regression wrapper, processor, and optional quantization config.

        The method restores either a full regression model directory or a PEFT
        adapter backed by ``base_model_name``.
        """

        logger.info(f"Loading regression model from {self.model_dir}")

        bnb = build_bnb_config(
            self.use_4bit,
            self.use_8bit,
            compute_dtype=self.bnb_compute_dtype,
        )
        self.model = VLMForMOSRegression.from_pretrained(
            self.model_dir,
            base_model_name=self.base_model_name,
            is_peft_adapter=self.is_peft_adapter,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=bnb,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_dir)

        if bnb is None:
            self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    # -------------------------------------------------------
    # Load test dataset
    # -------------------------------------------------------

    def load_dataset(self) -> Dataset:
        """
        Load and validate the configured test dataset.

        Returns:
            Test dataset containing ``image_path`` and ``mos_score`` columns.

        Raises:
            ValueError: If required columns are missing.
            FileNotFoundError: If the configured dataset path is missing.
        """

        loader = DatasetLoader(
            data_dir=self.data_dir,
            use_jsonl=self.use_jsonl,
            dataset_format=self.dataset_format,
            test_dataset_dir=self.test_dataset_dir,
            test_json_path=self.test_json_path,
        )

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
        """
        Run MOS regression inference for each sample in the dataset.

        Args:
            dataset: Test dataset containing image paths.

        Returns:
            One predicted MOS float per dataset sample.
        """

        preds: List[float] = []
        model_device = next(self.model.parameters()).device

        for sample in dataset:

            image = Image.open(sample["image_path"]).convert("RGB")

            inputs = self.processor(
                text=[self.prompt_text],
                images=[[image]],
                return_tensors="pt",
                padding=True,
            )

            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            mos_pred = outputs["mos_pred"].item()

            preds.append(float(mos_pred))

        return preds

    # -------------------------------------------------------
    # Run evaluation
    # -------------------------------------------------------

    def run(self) -> Dict:
        """
        Execute the full regression evaluation pipeline.

        Returns:
            Evaluation summary with metrics, sample count, and evaluator name.
            Prediction CSV and plots are written to ``output_dir``.
        """

        self.load_model()

        test_ds = self.load_dataset()

        logger.info("Running MOS predictions")

        preds = self.predict(test_ds)

        y_true = [float(x) for x in test_ds["mos_score"]]

        metrics = compute_all_metrics(y_true, preds)

        logger.info("Evaluation results")
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_scatter_plot(y_true, preds, self.output_dir / "scatter.png")
        save_error_histogram(y_true, preds, self.output_dir / "error_hist.png")
        save_predictions_csv(
            image_paths=list(test_ds["image_path"]),
            y_true=y_true,
            y_pred=preds,
            raw_outputs=[str(value) for value in preds],
            out_path=self.output_dir / "predictions.csv",
        )

        return {
            "metrics": metrics,
            "num_samples": len(test_ds),
            "prediction_failures": 0,
            "evaluator": "regression",
        }

    # -------------------------------------------------------
    # Save results
    # -------------------------------------------------------

    def save_results(self, results: Dict, output_path: str):
        """
        Persist final evaluation metrics to a JSON file.

        Args:
            results: Evaluation summary returned by ``run``.
            output_path: Destination JSON path.
        """

        path = Path(output_path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved evaluation results → {path}")
