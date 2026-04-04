import logging
import os
from typing import Optional

import mlflow
import onnxruntime as ort
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
client = MlflowClient(tracking_uri=DEFAULT_TRACKING_URI)


def register_model(run_id: str, model_path: str, model_name: str, description: Optional[str] = None) -> str:
    try:
        model_uri = f"runs:/{run_id}/{model_path}"
        registered_model = mlflow.register_model(model_uri, model_name)

        if description:
            client.update_registered_model(name=model_name, description=description)

        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="lifecycle",
            value="candidate",
        )
        logger.info("Model registered: %s v%s", model_name, registered_model.version)
        return str(registered_model.version)
    except Exception as exc:
        logger.error("Error registering model: %s", exc)
        raise


def transition_stage(model_name: str, version: str, stage: str) -> None:
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )
        logger.info("Model %s v%s transitioned to %s", model_name, version, stage)
    except Exception as exc:
        logger.error("Error transitioning model: %s", exc)
        raise


def load_production_model(model_name: str):
    try:
        model_uri = f"models:/{model_name}/Production"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        logger.error("Error loading production model: %s", exc)
        raise


def load_production_onnx_model(model_name: str = "onnx_model") -> ort.InferenceSession:
    try:
        model_uri = f"models:/{model_name}/Production"
        local_path = mlflow.artifacts.download_artifacts(model_uri=model_uri)
        return ort.InferenceSession(os.path.join(local_path, "model.onnx"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load Production model '{model_name}': {exc}") from exc

