import importlib
import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest


def load_app_module():
    sys.modules.pop("src.api.app", None)

    fake_inference_module = types.ModuleType("src.models.inference")
    fake_service = MagicMock()
    fake_inference_module.ONNXInferenceService = MagicMock(return_value=fake_service)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "src.models.inference", fake_inference_module)
        app_module = importlib.import_module("src.api.app")

    return app_module, fake_service


def test_health_endpoint_returns_expected_payload():
    app_module, _ = load_app_module()
    response = asyncio.run(app_module.health_check())

    assert response == {"status": "healthy", "model_version": "production"}

def test_single_prediction_endpoint_uses_prediction_service():
    app_module, fake_service = load_app_module()
    fake_service.predict_single.return_value = {
        "clean_text": "love this",
        "sentiment": "positive",
        "probabilities": {"positive": 0.98},
    }
    request = app_module.PredictionRequest(text="Love this!")
    response = asyncio.run(app_module.predict(request))

    fake_service.predict_single.assert_called_once_with("Love this!")
    assert response["sentiment"] == "positive"

def test_batch_prediction_endpoint_returns_server_error_on_exception():
    app_module, fake_service = load_app_module()
    fake_service.predict_batch.side_effect = RuntimeError("model unavailable")
    request = app_module.BatchPredictionRequest(texts=["a", "b"])

    with pytest.raises(app_module.HTTPException) as exc:
        asyncio.run(app_module.predict_batch(request))

    assert exc.value.status_code == 500
    assert exc.value.detail == "model unavailable"
