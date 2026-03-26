
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
import joblib
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

def validate_onnx_model(onnx_model):
    """Validate ONNX model"""
    try:
        # Load and check ONNX model
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model validated successfully: {onnx_model}")
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        raise

def export_sklearn_to_onnx(model: BaseEstimator,
                            feature_names: list,
                            initial_types: Optional[list] = None) -> str:
    """
    Export scikit-learn model to ONNX using skl2onnx
    """
    try:
        
        if initial_types is None:
            initial_types = [("input", StringTensorType([None, 1]))]   # Input shape = (batch_size, 1)
        
        # Convert to ONNX
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_types
        )
        
        # Save ONNX model
        # onnx_path = onnx_path / f"{model_name}.onnx"
        # with open(onnx_path, 'wb') as f:
        #     f.write(onnx_model.SerializeToString())
        
        # logger.info(f"Exported sklearn model to ONNX: {onnx_path}")
        
        # Validate
        validate_onnx_model(onnx_model)
        
        # return str(onnx_path)
        return onnx_model
    
    except Exception as e:
        logger.error(f"Error exporting sklearn to ONNX: {e}")
        raise