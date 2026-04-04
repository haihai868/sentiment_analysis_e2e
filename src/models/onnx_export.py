
import logging
from pathlib import Path
from typing import Optional

import onnx
from sklearn.base import BaseEstimator
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

logger = logging.getLogger(__name__)


def validate_onnx_model(onnx_model) -> None:
    try:
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validated successfully")
    except Exception as exc:
        logger.error("ONNX validation failed: %s", exc)
        raise


def export_sklearn_to_onnx(
    model: BaseEstimator,
    output_path: Path,
    initial_types: Optional[list] = None,
) -> Path:
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if initial_types is None:
            initial_types = [("input", StringTensorType([None, 1]))]

        onnx_model = convert_sklearn(model, initial_types=initial_types)
        validate_onnx_model(onnx_model)

        with output_path.open("wb") as handle:
            handle.write(onnx_model.SerializeToString())

        logger.info("Exported sklearn model to ONNX at %s", output_path)
        return output_path
    except Exception as exc:
        logger.error("Error exporting sklearn to ONNX: %s", exc)
        raise
