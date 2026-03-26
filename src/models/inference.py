import logging
import os
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

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "https://b31880069539.ngrok-free.app"
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

    def _run_prediction(self, text: str) -> tuple[Any, Optional[np.ndarray]]:
        """Run prediction with multiple input fallbacks to support common pyfunc signatures."""
        candidates = [
            [text],
            pd.DataFrame({"text": [text]}),
            np.array([text], dtype=object),
        ]

        last_err: Optional[Exception] = None
        for candidate in candidates:
            try:
                pred = self.model.predict(candidate)
                proba = None
                if hasattr(self.model, "predict_proba"):
                    proba = np.asarray(self.model.predict_proba(candidate))
                return pred, proba
            except Exception as err:  # noqa: PERF203 - deliberate fallback loop
                last_err = err

        raise RuntimeError(
            "Prediction failed for all input formats. "
            "Ensure the registered model supports raw text input."
        ) from last_err

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _build_response(self, text: str, prediction: Any, probabilities: Optional[np.ndarray]) -> Dict[str, Any]:
        sentiment = str(prediction)

        if probabilities is None or probabilities.size == 0:
            # Fall back when model has no probability output (e.g., LinearSVC)
            return {
                "text": text,
                "sentiment": sentiment,
                "confidence": 1.0,
                "probabilities": {sentiment: 1.0},
            }

        row = probabilities[0] if probabilities.ndim > 1 else probabilities
        model_obj = getattr(self, "model", None)
        label_names = getattr(model_obj, "classes_", None)

        if label_names is None:
            label_names = [f"class_{i}" for i in range(len(row))]

        probability_map = {
            str(label_names[idx]): self._safe_float(prob)
            for idx, prob in enumerate(row)
        }

        confidence = max(probability_map.values()) if probability_map else 0.0
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probability_map,
        }

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single text."""
        prediction, probabilities = self._run_prediction(text)
        pred_value = prediction[0] if isinstance(prediction, (list, np.ndarray, pd.Series)) else prediction
        return self._build_response(text, pred_value, probabilities)

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for multiple texts."""
        return [self.predict_single(text) for text in texts]


class ONNXInferenceService(InferenceService):
    """Inference service using ONNX Runtime and the single Production ONNX model."""

    def __init__(self, config_path: Optional[str] = None, model_path: str = None):
        self.session = ort.InferenceSession(model_path)
        self.text_processor = TextPreprocessor()
        

    def predict_single(self, text: str):
        texts = self.predict_batch([text])
        return texts[0]
    
    def _to_sentiment(label: int):
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
                        'text': text,
                        'sentiment': self._to_sentiment(sentiment),
                        'probabilities': prob,
                    }
                    for text, sentiment, prob in zip(texts, outputs[0], outputs[1])
                ]
            
