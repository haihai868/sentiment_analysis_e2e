import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

REFERENCE_PATH = Path(os.environ.get("REFERENCE_STATS_PATH", "artifacts/reference_stats.json"))
PREDICTION_LOG_PATH = Path(os.environ.get("PREDICTION_LOG_PATH", "artifacts/monitoring/predictions.jsonl"))
DRIFT_REPORT_PATH = Path(os.environ.get("DRIFT_REPORT_PATH", "artifacts/monitoring/drift_report.json"))

MIN_LABEL_DIST_DRIFT_THRESHOLD = 0.30
MIN_CONFIDENCE_THRESHOLD = 0.55
MIN_LENGTH_MEAN_DELTA_THRESHOLD = 15.0

def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_distribution(values: Iterable[str]) -> Dict[str, float]:
    counter = Counter(values)
    total = sum(counter.values()) or 1
    return {label: count / total for label, count in counter.items()}


def detect_drift(reference_path: Path = REFERENCE_PATH, prediction_log_path: Path = PREDICTION_LOG_PATH) -> Dict:
    reference = _read_json(reference_path)
    records = _read_jsonl(prediction_log_path)

    if not records:
        report = {
            "drift_detected": False,
            "reason": "no_prediction_data",
            "sample_count": 0,
            "checks": {},
        }
        DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    clean_lengths = [record["clean_text_length"] for record in records]
    confidences = [record["confidence"] for record in records]
    predicted_distribution = _normalize_distribution(record["predicted_label"] for record in records)

    length_mean = sum(clean_lengths) / len(clean_lengths)
    confidence_mean = sum(confidences) / len(confidences)
    reference_length_mean = reference["text_length"]["mean"]
    reference_distribution = reference.get("label_distribution", {})

    distribution_shift = 0.0
    # all_labels = sorted(set(reference_distribution) | set(predicted_distribution))
    all_labels = ['negative', 'neutral', 'positive']
    for label in all_labels:
        distribution_shift += abs(predicted_distribution.get(label, 0.0) - reference_distribution.get(label, 0.0))

    checks = {
        "text_length_mean_delta": abs(length_mean - reference_length_mean),
        "text_length_std_reference": reference["text_length"]["std"],
        "prediction_distribution_l1": distribution_shift,
        "confidence_mean": confidence_mean,
    }
    drift_detected = (
        checks["text_length_mean_delta"] > max(MIN_LENGTH_MEAN_DELTA_THRESHOLD, reference["text_length"]["std"] * 2.0)
        or checks["prediction_distribution_l1"] > MIN_LABEL_DIST_DRIFT_THRESHOLD
        or checks["confidence_mean"] < MIN_CONFIDENCE_THRESHOLD
    )

    report = {
        "drift_detected": drift_detected,
        "sample_count": len(records),
        "checks": checks,
        "reference_summary": {
            "text_length_mean": reference_length_mean,
            "label_distribution": reference_distribution,
        },
        "current_summary": {
            "text_length_mean": length_mean,
            "prediction_distribution": predicted_distribution,
            "confidence_mean": confidence_mean,
        },
    }
    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(detect_drift(), indent=2))
