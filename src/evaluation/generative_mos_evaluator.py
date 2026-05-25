from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from peft import PeftModel
from PIL import Image
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from src.data.loaders import DatasetLoader
from src.evaluation.io import save_predictions_csv
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.parsers import extract_rating
from src.evaluation.plotting import save_error_histogram, save_scatter_plot

logger = logging.getLogger("generative_mos_evaluator")


class GenerativeMOSEvaluator:
    """
    Evaluates generative VLM MOS predictors.

    Pipeline:
        image -> prompt -> generate text -> parse MOS -> metrics
    """

    evaluator_name = "generative"

    def __init__(
        self,
        model_dir: str,
        data_dir: str = "datasets/processed",
        use_jsonl: bool = False,
        system_prompt: str = "You are a medical image quality assessment assistant.",
        user_text: str = "Predict MOS score.",
        device: Optional[str] = None,
        base_model_name: Optional[str] = None,
        is_peft_adapter: bool = False,
        output_dir: str | Path = "output/eval/generative",
        dataset_format: Optional[str] = None,
        test_dataset_dir: Optional[str] = None,
        test_json_path: Optional[str] = None,
        test_jsonl_path: Optional[str] = None,
    ):
        """Store model, dataset, prompt, and output settings for evaluation."""
        self.model_dir = str(model_dir)
        self.base_model_name = base_model_name
        self.is_peft_adapter = is_peft_adapter
        self.output_dir = Path(output_dir)

        self.data_dir = data_dir
        self.use_jsonl = use_jsonl
        self.dataset_format = dataset_format
        self.test_dataset_dir = test_dataset_dir
        self.test_json_path = test_json_path or test_jsonl_path

        self.system_prompt = system_prompt
        self.user_text = user_text

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: AutoModelForImageTextToText | None = None
        self.processor: AutoProcessor | None = None

    def load_model(self):
        """
        Load the generative VLM and processor.

        The method supports both plain model directories and PEFT/LoRA adapter
        directories. Adapter evaluation requires ``base_model_name`` so the base
        VLM can be restored before the adapter is attached.

        Raises:
            ValueError: If an adapter directory is detected without adapter mode,
                or adapter mode is enabled without a base model name.
        """
        logger.info(f"Loading {self.evaluator_name} evaluator model from: {self.model_dir}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        adapter_config = Path(self.model_dir) / "adapter_config.json"
        if adapter_config.exists() and not self.is_peft_adapter:
            raise ValueError(
                f"{self.model_dir} contains adapter_config.json, so it looks like a PEFT/LoRA adapter. "
                "Set is_peft_adapter=True and provide base_model_name."
            )

        if self.is_peft_adapter:
            if not self.base_model_name:
                raise ValueError(
                    "is_peft_adapter=True requires base_model_name in eval config."
                )

            logger.info(f"Loading base model: {self.base_model_name}")
            base_model = AutoModelForImageTextToText.from_pretrained(
                self.base_model_name,
                device_map="auto",
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            logger.info(f"Loading PEFT adapter from: {self.model_dir}")
            self.model = PeftModel.from_pretrained(base_model, self.model_dir)

            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_dir,
                    trust_remote_code=True,
                    use_fast=False,
                )
                logger.info("Loaded processor from adapter directory.")
            except Exception as e:
                logger.warning(
                    f"Could not load processor from adapter dir: {e}. "
                    "Falling back to base model processor."
                )
                self.processor = AutoProcessor.from_pretrained(
                    self.base_model_name,
                    trust_remote_code=True,
                    use_fast=False,
                )
        else:
            logger.info("Loading plain/base VLM without PEFT adapter.")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_dir,
                device_map="auto",
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                use_fast=False,
            )

        self.model.eval()
        logger.info("Model + processor loaded.")

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

    def generate_predictions(self, dataset: Dataset):
        """
        Generate text responses for each image and parse MOS predictions.

        Args:
            dataset: Test dataset containing image paths and MOS labels.

        Returns:
            A tuple ``(predictions, raw_outputs)`` where predictions contain
            parsed MOS floats or ``None`` and raw outputs contain generated text.
        """
        predictions: List[Optional[float]] = []
        raw_outputs: List[str] = []

        model_device = next(self.model.parameters()).device
        tokenizer = self.processor.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)

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

            inputs = {key: value.to(model_device) for key, value in inputs.items()}

            generation_kwargs = {
                "max_new_tokens": 32,
                "do_sample": False,
                "use_cache": True,
            }
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)

            gen_text = self.processor.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            raw_outputs.append(gen_text)
            predictions.append(extract_rating(gen_text))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return predictions, raw_outputs

    def run(self) -> Dict:
        """
        Execute the full generative MOS evaluation pipeline.

        Returns:
            Evaluation summary with metrics, sample count, parse failures, and
            evaluator name. Prediction CSV and plots are written to ``output_dir``.
        """
        self.load_model()
        test_ds = self.load_dataset()

        logger.info("Generating predictions")
        preds, outputs = self.generate_predictions(test_ds)

        y_true = [float(x) for x in test_ds["mos_score"]]
        metrics = compute_all_metrics(y_true, preds)

        logger.info("Evaluation results")
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_scatter_plot(y_true, preds, self.output_dir / "scatter.png")
        save_error_histogram(y_true, preds, self.output_dir / "error_hist.png")

        image_paths = list(test_ds["image_path"])
        save_predictions_csv(
            image_paths=image_paths,
            y_true=y_true,
            y_pred=preds,
            raw_outputs=outputs,
            out_path=self.output_dir / "predictions.csv",
        )

        return {
            "metrics": metrics,
            "num_samples": len(test_ds),
            "prediction_failures": preds.count(None),
            "evaluator": self.evaluator_name,
        }

    def save_results(self, results: Dict, output_path: str):
        """
        Persist final evaluation metrics to a JSON file.

        Args:
            results: Evaluation summary returned by ``run``.
            output_path: Destination JSON path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        logger.info(f"Saved evaluation results -> {path}")
