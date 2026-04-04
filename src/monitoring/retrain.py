import json
import os
from pathlib import Path

DRIFT_REPORT_PATH = Path(os.environ.get("DRIFT_REPORT_PATH", "artifacts/monitoring/drift_report.json"))
RETRAIN_SIGNAL_PATH = Path(os.environ.get("RETRAIN_SIGNAL_PATH", "artifacts/monitoring/retrain_signal.json"))


def build_retrain_signal() -> dict:
    if not DRIFT_REPORT_PATH.exists():
        payload = {
            "should_retrain": False,
            "reason": "missing_drift_report",
        }
    else:
        report = json.loads(DRIFT_REPORT_PATH.read_text(encoding="utf-8"))
        should_retrain = bool(report.get("drift_detected")) and int(report.get("sample_count", 0)) >= 100
        payload = {
            "should_retrain": should_retrain,
            "reason": "drift_detected" if should_retrain else "threshold_not_met",
            "drift_report": report,
        }

    RETRAIN_SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_SIGNAL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(build_retrain_signal(), indent=2))
