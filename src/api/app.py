from fastapi import FastAPI, HTTPException
import uvicorn
import logging
from pathlib import Path
from typing import List

from src.models.inference import InferenceService
from src.api.schema import PredictionRequest, BatchPredictionRequest, PredictionResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Twitter Sentiment Analysis API", version="1.0.0")

# Initialize prediction service
config_path = Path(__file__).parent.parent.parent / "config/config.yaml"
prediction_service = InferenceService(str(config_path))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_version": "production"  # You can get actual version from registry
    }

@app.post("/onnx_predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict sentiment for single text"""
    try:
        result = prediction_service.predict_single(request.text)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/onnx_predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """Predict sentiment for multiple texts"""
    try:
        results = prediction_service.predict_batch(request.texts)
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))