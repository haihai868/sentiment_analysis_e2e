from fastapi import FastAPI, HTTPException, Request
import uvicorn
import logging
from pathlib import Path
from typing import List
import time

from prometheus_client import Counter, Histogram, generate_latest
from src.models.inference import ONNXInferenceService
from src.api.schema import PredictionRequest, BatchPredictionRequest, PredictionResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Twitter Sentiment Analysis API")

REQUEST_COUNT = Counter(
    "request_count",
    "Total requests",
    ["endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency",
    ["endpoint"]
)

# Initialize prediction service
# config_path = Path(__file__).parent.parent.parent / "config/config.yaml"
prediction_service = ONNXInferenceService(model_path='artifacts/onnx_pipeline.onnx')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    # get route
    route = request.scope.get("route")
    endpoint = route.path if route else "unknown"

    # method = request.method

    # group status: 200 → 2xx
    status = f"{response.status_code // 100}xx"

    # update metrics
    REQUEST_COUNT.labels(endpoint, status).inc()
    REQUEST_LATENCY.labels(endpoint).observe(latency)

    return response

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


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()