"""
generate_sample_csv.py
-----------------------
Generates a small sample CSV from the test set that can be uploaded
directly to the Streamlit prediction page.

Run:
    python generate_sample_csv.py

Output:
    sample_input.csv  (5 rows, 561 features each, no header)
"""

import os
import pandas as pd

DATA_DIR = os.path.join("data", "UCI HAR Dataset")

x_path = os.path.join(DATA_DIR, "test", "X_test.txt")
y_path = os.path.join(DATA_DIR, "test", "y_test.txt")

if not os.path.exists(x_path):
    print("Dataset not found. Run `python download_dataset.py` first.")
else:
    X = pd.read_csv(x_path, sep=r"\s+", header=None)
    y_raw = pd.read_csv(y_path, sep=r"\s+", header=None).squeeze()

    activity_map = {1: "Walking", 2: "Walking Upstairs", 3: "Walking Downstairs",
                    4: "Sitting", 5: "Standing", 6: "Laying"}
    y = y_raw.map(activity_map)

    # Take one sample per activity (up to 5)
    sample_indices = []
    for label in list(activity_map.values())[:5]:
        idx = y[y == label].index[0]
        sample_indices.append(idx)

    sample = X.loc[sample_indices]
    sample.to_csv("sample_input.csv", index=False, header=False)

    print(f"Saved sample_input.csv with {len(sample)} rows and {sample.shape[1]} features.")
    print("True labels for these rows:")
    for i, idx in enumerate(sample_indices):
        print(f"  Row {i+1}: {y.loc[idx]}")
    print("\nUpload sample_input.csv on the 'Activity Prediction' page.")
