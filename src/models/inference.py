import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv

from src.data_pipeline.preprocess import TextPreprocessor
from src.models.model_registry import load_production_onnx_model

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DEFAULT_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "onnx_model")
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/onnx_model.onnx")
DEFAULT_MODEL_INFO_PATH = os.environ.get("MODEL_INFO_PATH", "artifacts/model_info.json")


class ONNXInferenceService:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        tracking_uri: str = DEFAULT_TRACKING_URI,
        model_info_path: str = DEFAULT_MODEL_INFO_PATH,
    ):
        self.model_name = model_name
        self.model_source = "registry"
        self.model_version = "unknown"
        self.run_id = None
        self.model_info_path = Path(model_info_path)
        self.text_processor = TextPreprocessor()
        mlflow.set_tracking_uri(tracking_uri)

        if model_path and Path(model_path).exists():
            self.session = ort.InferenceSession(model_path)
            self.model_source = "local"
            self.model_version = "local"
        else:
            self.session = load_production_onnx_model(model_name)

        self._load_model_metadata()

    def _load_model_metadata(self) -> None:
        if not self.model_info_path.exists():
            return

        try:
            payload = json.loads(self.model_info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid model metadata file at %s", self.model_info_path)
            return

        self.model_name = payload.get("model_name", self.model_name)
        self.model_version = str(payload.get("model_version", self.model_version))
        self.run_id = payload.get("run_id")

    @staticmethod
    def _label_to_sentiment(label: int) -> str:
        if label == 1:
            return "positive"
        if label == -1:
            return "negative"
        return "neutral"

    @staticmethod
    def _coerce_probability(probabilities: Any) -> Dict[str, float]:
        if isinstance(probabilities, dict):
            return {str(key): float(value) for key, value in probabilities.items()}
        if isinstance(probabilities, np.ndarray):
            probabilities = probabilities.tolist()
        if isinstance(probabilities, list):
            return {str(index): float(value) for index, value in enumerate(probabilities)}
        return {"score": float(probabilities)}

    def get_model_metadata(self) -> Dict[str, Optional[str]]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "model_source": self.model_source,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        cleaned_texts = self.text_processor.preprocess_batch(texts)
        inputs = np.array(cleaned_texts, dtype=object).reshape(-1, 1)
        outputs = self.session.run(None, {"input": inputs})

        sentiments = outputs[0]
        probabilities = outputs[1] if len(outputs) > 1 else [1.0] * len(cleaned_texts)

        return [
            {
                "clean_text": clean_text,
                "sentiment": self._label_to_sentiment(int(sentiment)),
                "probabilities": self._coerce_probability(probability),
            }
            for clean_text, sentiment, probability in zip(cleaned_texts, sentiments, probabilities)
        ]

    def predict_single(self, text: str) -> Dict[str, Any]:
        return self.predict_batch([text])[0]
