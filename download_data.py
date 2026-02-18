"""Download the Fake News dataset from Kaggle."""

import os
import zipfile

def download_dataset():
    """Download and extract the WELFake dataset using the Kaggle API."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Check if data already exists
    if os.path.exists(os.path.join(data_dir, "WELFake_Dataset.csv")):
        print("Dataset already exists in data/. Skipping download.")
        return

    # Set Kaggle credentials path
    kaggle_dir = os.path.join(os.path.dirname(__file__), ".kaggle")
    os.environ["KAGGLE_CONFIG_DIR"] = kaggle_dir

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print("Downloading WELFake dataset (saurabhshahane/fake-news-classification)...")
    api.dataset_download_files(
        "saurabhshahane/fake-news-classification", path=data_dir
    )

    # Extract the zip file
    zip_path = os.path.join(data_dir, "fake-news-classification.zip")
    if os.path.exists(zip_path):
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        os.remove(zip_path)

    print("Dataset downloaded and extracted to data/")
    # List files
    for f in os.listdir(data_dir):
        size_mb = os.path.getsize(os.path.join(data_dir, f)) / (1024 * 1024)
        print(f"  {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download_dataset()
