"""
download_dataset.py
--------------------
Downloads and extracts the UCI Human Activity Recognition dataset
into the data/ folder.

Run this once before training:
    python download_dataset.py
"""

import os
import zipfile
import requests

DATASET_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "har_dataset.zip")
EXTRACTED_DIR = os.path.join(DATA_DIR, "UCI HAR Dataset")


def download_file(url, dest_path):
    """Download a file from a URL with a progress indicator."""
    print(f"Downloading dataset from:\n  {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

    print("\nDownload complete.")


def extract_zip(zip_path, extract_to):
    """Extract the outer ZIP, then extract any inner ZIP that contains the dataset."""
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as outer:
        outer.extractall(extract_to)

    # The outer ZIP contains another ZIP with the actual dataset files
    inner_zip = os.path.join(extract_to, "UCI HAR Dataset.zip")
    if os.path.exists(inner_zip):
        print("Extracting inner dataset ZIP ...")
        with zipfile.ZipFile(inner_zip, "r") as inner:
            inner.extractall(extract_to)
        os.remove(inner_zip)

    print("Extraction complete.")


def verify_dataset(base_dir):
    """Check that key dataset files are present."""
    required_files = [
        os.path.join(base_dir, "train", "X_train.txt"),
        os.path.join(base_dir, "train", "y_train.txt"),
        os.path.join(base_dir, "test", "X_test.txt"),
        os.path.join(base_dir, "test", "y_test.txt"),
        os.path.join(base_dir, "features.txt"),
        os.path.join(base_dir, "activity_labels.txt"),
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("WARNING: The following expected files are missing:")
        for f in missing:
            print(f"  {f}")
        return False
    print("Dataset verified successfully.")
    return True


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(EXTRACTED_DIR) and verify_dataset(EXTRACTED_DIR):
        print("\nDataset already exists. Nothing to do.")
        return

    if not os.path.exists(ZIP_PATH):
        download_file(DATASET_URL, ZIP_PATH)
    else:
        print("ZIP already downloaded, skipping download.")

    extract_zip(ZIP_PATH, DATA_DIR)

    if verify_dataset(EXTRACTED_DIR):
        print("\nDataset is ready at:", EXTRACTED_DIR)
    else:
        print("\nSetup may be incomplete. Check the data/ folder manually.")


if __name__ == "__main__":
    main()
