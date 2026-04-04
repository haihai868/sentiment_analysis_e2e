import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.onnx
import mlflow.sklearn
import onnx
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.models.model_registry import register_model, transition_stage
from src.models.onnx_export import export_sklearn_to_onnx

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DEFAULT_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "sentiment analysis")

ARTIFACT_DIR = Path("artifacts")
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
REFERENCE_STATS_PATH = ARTIFACT_DIR / "reference_stats.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_info.json"
ONNX_PATH = ARTIFACT_DIR / "onnx_model.onnx"

DEFAULT_GATE = {
    "accuracy_min": 0.75,
    "f1_min": 0.75,
    "precision_min": 0.70,
    "recall_min": 0.70,
}

LABEL_MAP = {
    -1: "negative",
    0: "neutral",
    1: "positive",
}


def _ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _load_params() -> Dict[str, Any]:
    params_path = Path("params.yaml")
    if not params_path.exists():
        return {}

    import yaml

    with params_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_vectorizer(vectorizer_type: str, ngram_range: Tuple[int, int], max_features: int):
    if vectorizer_type == "bow":
        return CountVectorizer(ngram_range=ngram_range, max_features=max_features)
    if vectorizer_type == "tfidf":
        return TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    raise ValueError("invalid vectorizer_type")


def _build_model(model_type: str, model_params: Dict[str, Any]):
    if model_type == "logistic":
        return LogisticRegression(**model_params)
    if model_type == "nb":
        return MultinomialNB(**model_params)
    if model_type == "svm":
        return LinearSVC(**model_params)
    raise ValueError("invalid model_type")


def _build_reference_stats(df: pd.DataFrame) -> Dict[str, Any]:
    clean_text = df["clean_text"].fillna("").astype(str)
    text_lengths = clean_text.str.len()
    token_counts = clean_text.str.split().apply(len)
    label_distribution = (
        df["category"]
        .map(lambda value: LABEL_MAP.get(value, str(value)))
        .value_counts(normalize=True)
        .sort_index()
        .apply(float)
        .to_dict()
    )

    return {
        "dataset_size": int(len(df)),
        "text_length": {
            "mean": float(text_lengths.mean()),
            "std": float(text_lengths.std(ddof=0) or 0.0),
            "p50": float(text_lengths.quantile(0.50)),
            "p95": float(text_lengths.quantile(0.95)),
        },
        "token_count": {
            "mean": float(token_counts.mean()),
            "std": float(token_counts.std(ddof=0) or 0.0),
            "p50": float(token_counts.quantile(0.50)),
            "p95": float(token_counts.quantile(0.95)),
        },
        "label_distribution": label_distribution,
    }


def _baseline_gate(metrics: Dict[str, float], gate: Dict[str, float]) -> bool:
    return (
        metrics["accuracy"] >= gate["accuracy_min"]
        and metrics["f1_score"] >= gate["f1_min"]
        and metrics["precision_weighted"] >= gate["precision_min"]
        and metrics["recall_weighted"] >= gate["recall_min"]
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiment(
    df: pd.DataFrame,
    vectorizer_type: str = "tfidf",
    ngram_range: Tuple[int, int] = (1, 1),
    vectorizer_max_features: int = 5000,
    model_type: str = "logistic",
    model_params: Optional[Dict[str, Any]] = None,
    model_path: str = "model",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    default_params = {
        "logistic": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        "nb": {"alpha": 1.0},
        "svm": {"C": 1.0},
    }

    model_params = {**default_params[model_type], **(model_params or {})}

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["category"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["category"],
    )

    vectorizer = _build_vectorizer(vectorizer_type, ngram_range, vectorizer_max_features)
    model = _build_model(model_type, model_params)
    pipeline = Pipeline([("vectorizer", vectorizer), ("model", model)])

    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    mlflow.set_experiment(DEFAULT_EXPERIMENT)

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        }
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        for key, value in {
            "vectorizer_type": vectorizer_type,
            "ngram_range": str(ngram_range),
            "max_features": vectorizer_max_features,
            "model_type": model_type,
            "test_size": test_size,
            "random_state": random_state,
        }.items():
            mlflow.log_param(key, value)
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path=model_path)
        export_sklearn_to_onnx(model=pipeline, output_path=ONNX_PATH)
        mlflow.onnx.log_model(onnx_model=onnx.load(str(ONNX_PATH)), artifact_path=f"onnx_{model_path}")

        gate = DEFAULT_GATE | _load_params().get("gates", {})
        gate_passed = _baseline_gate(metrics, gate)

    sklearn_version = register_model(
        run_id=run.info.run_id,
        model_path=model_path,
        model_name="sklearn",
        description="Twitter Sentiment Analysis Model",
    )
    onnx_version = register_model(
        run_id=run.info.run_id,
        model_path=f"onnx_{model_path}",
        model_name="onnx_model",
        description="ONNX version of Twitter Sentiment Analysis Model",
    )

    if gate_passed:
        transition_stage(model_name="sklearn", version=sklearn_version, stage="Staging")
        transition_stage(model_name="onnx_model", version=onnx_version, stage="Staging")

    _write_json(
        METRICS_PATH,
        {
            "run_id": run.info.run_id,
            "registered_versions": {
                "sklearn": sklearn_version,
                "onnx_model": onnx_version,
            },
            "gate": gate,
            "gate_passed": gate_passed,
            "metrics": metrics,
            "classification_report": report,
            "test_samples": int(len(X_test)),
        },
    )
    _write_json(REFERENCE_STATS_PATH, _build_reference_stats(df))
    _write_json(
        MODEL_CARD_PATH,
        {
            "model_name": "onnx_model",
            "model_version": str(onnx_version),
            "run_id": run.info.run_id,
            "source": str(ONNX_PATH),
            "gate_passed": gate_passed,
        },
    )

    return {
        **metrics,
        "model_version": onnx_version,
        "gate_passed": gate_passed,
        "run_id": run.info.run_id,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _ensure_dirs()
    dataframe = pd.read_csv("data/processed/processed_twitter_data.csv")
    params = _load_params()
    training = params.get("training", {})
    model_cfg = params.get("model", {})

    results = run_experiment(
        dataframe,
        vectorizer_type=model_cfg.get("vectorizer_type", "tfidf"),
        ngram_range=tuple(model_cfg.get("ngram_range", [1, 1])),
        vectorizer_max_features=model_cfg.get("max_features", 5000),
        model_type=model_cfg.get("type", "logistic"),
        model_params=model_cfg.get("params"),
        test_size=training.get("test_size", 0.2),
        random_state=training.get("random_state", 42),
    )
    print(json.dumps(results, indent=2))
