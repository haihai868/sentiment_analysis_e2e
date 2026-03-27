import logging
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
import onnxruntime as ort
import yaml
from mlflow.tracking import MlflowClient

from src.data_pipeline.preprocess import TextPreprocessor
from src.models.model_registry import load_production_onnx_model


load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DEFAULT_MODEL_NAME = "dfbgdf"


class InferenceService:
    """Service for model predictions."""

    def __init__(self, config_path: str, model_path: str = None):
        self.config = self._load_config(config_path)
        tracking_uri = self.config["mlflow"]["tracking_uri"]
        mlflow.set_tracking_uri(tracking_uri)

        self.model = self._load_model(model_path) if model_path else self._load_production_model()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load config file if available; otherwise use safe defaults."""
        config: Dict[str, Any] = {
            "mlflow": {
                "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
                "model_name": os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME),
            }
        }

        path = Path(config_path)
        if not path.exists():
            logger.warning("Config file not found at %s, using defaults/env", config_path)
            return config

        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}

        if "mlflow" in loaded:
            config["mlflow"].update(loaded["mlflow"])

        return config

    def _load_model(self, model_path: str):
        """Load a local model artifact."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        suffix = path.suffix.lower()
        if suffix in {".joblib", ".pkl"}:
            return joblib.load(path)

        # If this is an MLflow model directory, load with pyfunc.
        return mlflow.pyfunc.load_model(str(path))

    def _load_production_model(self):
        """Load production model from MLflow Registry."""
        try:
            model_uri = f"models:/{self.config['mlflow']['model_name']}/Production"
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.error("Error loading production model: %s", e)
            raise

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single text."""
        pass

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for multiple texts."""
        pass

class ONNXInferenceService(InferenceService):
    """Inference service using ONNX Runtime and the single Production ONNX model."""

    def __init__(self, config_path: Optional[str] = None, model_path: str = None, model_name: str = "onnx_model"):
        if model_path:
            self.session = ort.InferenceSession(model_path)
        else:
            self.session = load_production_onnx_model(model_name)
        self.text_processor = TextPreprocessor()

    @staticmethod
    def _label_to_sentiment(label: int):
        if label == 1:
            return 'positive'
        elif label ==-1:
            return 'negative'
        return 'neutral'


    def predict_batch(self, texts: List[str]):
        texts = self.text_processor.preprocess_batch(texts)
        inputs = np.array(texts).reshape(-1, 1)   # Input shape = (batch_size, 1)
        

        outputs = self.session.run(None, {"input": inputs})

        return  [
                    {   
                        'clean_text': text,
                        'sentiment': self._label_to_sentiment(sentiment),
                        'probabilities': prob,
                    }
                    for text, sentiment, prob in zip(texts, outputs[0], outputs[1])
                ]
    
    def predict_single(self, text: str):
        texts = self.predict_batch([text])
        return texts[0]
