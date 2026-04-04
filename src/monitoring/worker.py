import json
import logging
import os
import time

from src.monitoring.drift import detect_drift
from src.monitoring.retrain import build_retrain_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = int(os.environ.get("MONITORING_INTERVAL_SECONDS", "300"))


def run_forever() -> None:
    while True:
        drift_report = detect_drift()
        retrain_signal = build_retrain_signal()
        logger.info(
            "Monitoring cycle complete: drift_detected=%s should_retrain=%s",
            drift_report.get("drift_detected"),
            retrain_signal.get("should_retrain"),
        )
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()
