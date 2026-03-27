import pandas as pd
import numpy as np
from typing import Tuple, Optional
import kaggle
import logging
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Handle data ingestion from ?"""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_path = Path(config['data']['raw_path'])
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def download_kaggle_dataset(self, dataset_name: str) -> str:
        """Download dataset from Kaggle"""
        try:
            logger.info(f"Downloading dataset: {dataset_name}")
            kaggle.api.dataset_download_files(
                dataset_name,
                path=self.data_path,
                unzip=True
            )
            return str(self.data_path)
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise