import os
from dotenv import load_dotenv
from typing import Dict, Optional
import logging
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
import onnxruntime as ort

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
client = MlflowClient()

mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    
def register_model(run_id: str, model_path: str, model_name, description: Optional[str] = None) -> str:
    """Register model in registry"""
    try:
        # Register model
        model_uri = f"runs:/{run_id}/{model_path}"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        # Add description
        if description:
            client.update_registered_model(
                name=model_name,
                description=description
            )
        # Add version tags
        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="stage",
            value="staging"
        )
        
        logger.info(f"Model registered: {model_name} v{registered_model.version}")
        return registered_model.version
        
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise

def transition_stage(model_name, version: str, stage: str):
    """Transition model to different stage"""
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True
        )
        logger.info(f"Model {model_name} v{version} transitioned to {stage}")
    except Exception as e:
        logger.error(f"Error transitioning model: {e}")
        raise

def load_production_model(model_name):
    """Load production model from MLflow"""
    try:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        logger.error(f"Error loading production model: {e}")
        raise

def load_production_onnx_model(model_name) -> ort.InferenceSession:
    """Load onnx model from MLflow Registry stage Production"""
    try:
        model_uri = f"models:/{model_name}/Production"
        
        local_path = mlflow.artifacts.download_artifacts(model_uri=model_uri)
        return ort.InferenceSession(f"{local_path}/model.onnx")
    
    except Exception as e:
        raise RuntimeError(f"Failed to load Production model '{model_name}': {e}")

