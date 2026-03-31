import os
import zipfile
import logging

from dotenv import load_dotenv
load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi


logger = logging.getLogger(__name__)

def download_kaggle_dataset(dataset: str, output_path: str):
    """
    Download dataset from Kaggle
    """

    api = KaggleApi()
    api.authenticate()

    os.makedirs(output_path, exist_ok=True)

    logger.info(f"Downloading dataset: {dataset}...")

    api.dataset_download_files(
        dataset,
        path=output_path,
        unzip=False  
    )

    logger.info("Complete downloading. Unzipping...")

    for file in os.listdir(output_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(output_path, file)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_path)

            # remove zip path
            os.remove(zip_path)

    logger.info("Done!")


if __name__ == "__main__":
    download_kaggle_dataset(
        dataset="saurabhshahane/twitter-sentiment-dataset",
        output_path="data/raw"
    )