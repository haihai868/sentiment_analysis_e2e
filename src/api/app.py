import json
import logging
from pathlib import Path
import os
from fastapi import FastAPI, HTTPException, Request, Response
from typing import List
import time
import uuid

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from src.models.inference import ONNXInferenceService
from src.api.schema import PredictionRequest, BatchPredictionRequest, PredictionResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Twitter Sentiment Analysis API")

registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    "request_count_total",  # metric name
    "Total requests",   # metric description
    ["endpoint", "status"], # metric labels
    registry=registry   # register the metric with custom registry
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency",
    ["endpoint"],
    registry=registry
)
PREDICTION_COUNT = Counter(
    "prediction_count_total",
    "Predictions by sentiment",
    ["sentiment"],
    registry=registry
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Prediction confidence distribution",
    ["sentiment"],
    registry=registry
)
INPUT_TEXT_LENGTH = Histogram(
    "input_text_length",
    "Input text length",
    registry=registry
)
MONITORING_EVENTS = Counter(
    "monitoring_events_total",
    "Prediction log events",
    ["status"],
    registry=registry
)
DRIFT_ALERT = Gauge(
    "drift_alert_active",
    "Whether the latest drift report is above threshold",
    registry=registry
)

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/onnx_model.onnx")
PREDICTION_LOG_PATH = Path(os.environ.get("PREDICTION_LOG_PATH", "artifacts/monitoring/predictions.jsonl"))
DRIFT_REPORT_PATH = Path(os.environ.get("DRIFT_REPORT_PATH", "artifacts/monitoring/drift_report.json"))
prediction_service = ONNXInferenceService(model_path=MODEL_PATH)


def _max_probability(probabilities: dict) -> float:
    if not probabilities:
        return 0.0
    return max(float(value) for value in probabilities.values())


def _log_prediction(text: str, prediction: dict) -> None:
    payload = {
        "request_id": str(uuid.uuid4()),
        "timestamp": int(time.time()),
        "model": prediction_service.get_model_metadata(),
        "raw_text_length": len(text),
        "clean_text_length": len(prediction["clean_text"]),
        "predicted_label": prediction["sentiment"],
        "probabilities": prediction["probabilities"],
        "confidence": _max_probability(prediction["probabilities"]),
    }

    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _update_prediction_metrics(text: str, prediction: dict) -> None:
    sentiment = prediction["sentiment"]
    confidence = _max_probability(prediction["probabilities"])

    PREDICTION_COUNT.labels(sentiment).inc()
    PREDICTION_CONFIDENCE.labels(sentiment).observe(confidence)
    INPUT_TEXT_LENGTH.observe(len(text))

    try:
        _log_prediction(text, prediction)
        MONITORING_EVENTS.labels("success").inc()
    except Exception as exc:
        logger.warning("Failed to write prediction log: %s", exc)
        MONITORING_EVENTS.labels("failure").inc()


def _refresh_drift_gauge() -> None:
    if not DRIFT_REPORT_PATH.exists():
        DRIFT_ALERT.set(0)
        return

    try:
        payload = json.loads(DRIFT_REPORT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        DRIFT_ALERT.set(0)
        return

    DRIFT_ALERT.set(1 if payload.get("drift_detected") else 0)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    # get route
    route = request.scope.get("route")
    endpoint = route.path if route else "unknown"

    # method = request.method

    status = f"{response.status_code // 100}xx"

    # update metrics
    REQUEST_COUNT.labels(endpoint, status).inc()
    REQUEST_LATENCY.labels(endpoint).observe(latency)

    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    metadata = prediction_service.get_model_metadata()
    _refresh_drift_gauge()
    return {
        "status": "healthy",
        "model_version": metadata["model_version"]
    }

@app.post("/onnx_predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict sentiment for single text"""

    try:
        result = prediction_service.predict_single(request.text)
        _update_prediction_metrics(request.text, result)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/onnx_predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """Predict sentiment for multiple texts"""
    try:
        results = prediction_service.predict_batch(request.texts)
        for raw_text, prediction in zip(request.texts, results):
            _update_prediction_metrics(raw_text, prediction)
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    _refresh_drift_gauge()
    return Response(generate_latest(registry), media_type="text/plain; version=0.0.4; charset=utf-8")
