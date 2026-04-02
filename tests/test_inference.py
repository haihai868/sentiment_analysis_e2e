import importlib
import sys
from unittest.mock import MagicMock, patch


def load_inference_module():
    sys.modules.pop("src.models.inference", None)
    with patch("nltk.download", return_value=True):
        return importlib.import_module("src.models.inference")


def test_predict_batch_formats_output_from_onnx_session():
    inference = load_inference_module()

    fake_session = MagicMock()
    fake_session.run.return_value = [
        [1, 0, -1],
        [
            {"positive": 0.9},
            {"neutral": 0.8},
            {"negative": 0.7},
        ],
    ]

    fake_processor = MagicMock()
    fake_processor.preprocess_batch.return_value = ["great day", "it is ok", "bad day"]

    with patch.object(inference.ort, "InferenceSession", return_value=fake_session), patch.object(
        inference, "TextPreprocessor", return_value=fake_processor
    ):
        service = inference.ONNXInferenceService(model_path="artifacts/onnx_pipeline.onnx")
        results = service.predict_batch(["raw1", "raw2", "raw3"])

    fake_processor.preprocess_batch.assert_called_once_with(["raw1", "raw2", "raw3"])
    fake_session.run.assert_called_once()
    assert results == [
        {
            "clean_text": "great day",
            "sentiment": "positive",
            "probabilities": {"positive": 0.9},
        },
        {
            "clean_text": "it is ok",
            "sentiment": "neutral",
            "probabilities": {"neutral": 0.8},
        },
        {
            "clean_text": "bad day",
            "sentiment": "negative",
            "probabilities": {"negative": 0.7},
        },
    ]

def test_predict_single_returns_first_batch_result():
    inference = load_inference_module()

    with patch.object(inference.ort, "InferenceSession", return_value=MagicMock()), patch.object(
        inference, "TextPreprocessor", return_value=MagicMock()
    ):
        service = inference.ONNXInferenceService(model_path="artifacts/onnx_pipeline.onnx")

    with patch.object(
        service,
        "predict_batch",
        return_value=[{"clean_text": "hello", "sentiment": "positive", "probabilities": {}}],
    ) as predict_batch:
        result = service.predict_single("Hello")

    predict_batch.assert_called_once_with(["Hello"])
    assert result["sentiment"] == "positive"
