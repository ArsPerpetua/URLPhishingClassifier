# train.py
import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from dataset import load_data, split_data
from config import MODEL_PATH

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def train_model():
    """Fungsi untuk melatih model Random Forest."""
    # Load dan bagi data
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid, cv=5
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # Melakukan prediksi
    y_pred = model.predict(X_test)

    # Menghitung akurasi dan menampilkan laporan klasifikasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {scores}")

    # Simpan model dan fitur terpilih
    joblib.dump(model, MODEL_PATH)
    feature_names = X_train.columns.tolist()
    with open("selected_features.txt", "w") as f:
        f.write("\n".join(feature_names))


if __name__ == "__main__":
    train_model()
