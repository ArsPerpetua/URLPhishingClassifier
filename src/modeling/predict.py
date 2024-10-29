import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features import extract_features, extract_additional_features
from config import MODEL_PATH  # Asumsi ada path model
import joblib  # Untuk menyimpan dan memuat model


def load_model():
    """Load model yang sudah dilatih."""
    model = joblib.load(MODEL_PATH)
    return model


def load_selected_features():
    """Load nama fitur yang telah diseleksi."""
    with open("selected_features.txt", "r") as f:
        features = f.read().splitlines()
    return features


def predict_url(url):
    """Fungsi untuk memprediksi apakah URL adalah phishing atau tidak."""
    model = load_model()

    # Ekstrak fitur dari URL
    features = extract_features(url)

    print(features)

    # Tambahkan fitur tambahan langsung dalam proses ekstraksi
    additional_features = extract_additional_features(features)

    if "URL" in additional_features.columns:
        additional_features = additional_features.drop(columns=["URL"])

    print(additional_features)
    # Menggabungkan fitur dasar dan tambahan
    all_features = additional_features
    # Ambil fitur yang telah diseleksi
    selected_features = load_selected_features()

    # Pastikan fitur yang diekstrak sesuai dengan yang digunakan saat pelatihan
    all_features = all_features[selected_features]
    print(all_features)

    prediction = model.predict(all_features)

    return prediction


if __name__ == "__main__":
    # Minta input URL dari pengguna
    url = input("Masukkan URL yang ingin diprediksi: ")
    prediction = predict_url(url)
    print(f"Prediction for URL '{url}': {prediction[0]}")  # Tampilkan hasil prediksi
