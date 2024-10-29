# src/dataset.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import train_test_split
from features import extract_additional_features
from src.config import DATASET_PATH
from src.features import select_features


def load_data():
    """Load dataset dari path yang udah ditentukan di config."""
    data = pd.read_csv(DATASET_PATH)
    # Setelah memuat dataset
    data = extract_additional_features(data)
    print(f"Data loaded. Shape: {data.shape}")
    return data


def split_data(data, test_size=0.2, random_state=42):
    """Pisahkan fitur dan label, lalu bagi dataset jadi training dan testing."""
    X = data.drop(columns=["label"])  # Asumsi kolom 'label' sebagai target
    y = data["label"]

    # Seleksi fitur
    X_selected = select_features(X, y, k=40)  # Ambil top-10 fitur

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


