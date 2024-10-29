import re
import pandas as pd
from urllib.parse import urlparse
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy
import numpy as np


def is_ip(domain):
    """Cek apakah domain adalah IP address."""
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))


def extract_features(url):
    """Extract semua fitur utama dari URL input sesuai dataset yang ada."""
    parsed_url = urlparse(url)
    features = {}

    """Mengambil fitur dasar dari URL."""
    features["URL"] = url

    # Basic URL features
    features["URLLength"] = len(url)
    features["Domain"] = parsed_url.netloc
    features["DomainLength"] = len(parsed_url.netloc)
    features["IsDomainIP"] = is_ip(parsed_url.netloc)
    features["TLD"] = parsed_url.netloc.split(".")[-1]
    features["NoOfSubDomain"] = parsed_url.netloc.count(".")

    # Placeholder for similarity index - update as per algorithm requirements
    features["URLSimilarityIndex"] = 0.5  # Contoh default

    # Character continuation rate (contoh: proporsi karakter yang berlanjut tanpa pemisah)
    features["CharContinuationRate"] = len(parsed_url.path) / (len(url) + 1)

    # Placeholder TLD legitimate probability - akan diperbarui
    features["TLDLegitimateProb"] = 0.05  # Placeholder

    # URL Character Probability
    features["URLCharProb"] = len(re.findall(r"[a-zA-Z]", url)) / (len(url) + 1)

    # TLD Length
    features["TLDLength"] = len(features["TLD"])

    # Obfuscation features
    features["HasObfuscation"] = "@" in url
    features["NoOfObfuscatedChar"] = url.count("@")
    features["ObfuscationRatio"] = features["NoOfObfuscatedChar"] / (len(url) + 1)

    # Letters and digits in URL
    features["NoOfLettersInURL"] = len(re.findall(r"[a-zA-Z]", url))
    features["LetterRatioInURL"] = features["NoOfLettersInURL"] / (len(url) + 1)
    features["NoOfDegitsInURL"] = len(re.findall(r"\d", url))
    features["DegitRatioInURL"] = features["NoOfDegitsInURL"] / (len(url) + 1)

    # Special character counts
    features["NoOfEqualsInURL"] = url.count("=")
    features["NoOfQMarkInURL"] = url.count("?")
    features["NoOfAmpersandInURL"] = url.count("&")
    features["NoOfOtherSpecialCharsInURL"] = len(re.findall(r"[^a-zA-Z0-9]", url))
    features["SpacialCharRatioInURL"] = features["NoOfOtherSpecialCharsInURL"] / (
        len(url) + 1
    )

    # HTTPS check
    features["IsHTTPS"] = 1 if parsed_url.scheme == "https" else 0

    # Placeholder for line of code and title-related features (contoh values)
    features["LineOfCode"] = 1000  # Contoh default
    features["LargestLineLength"] = 100  # Contoh default
    features["HasTitle"] = 1  # Contoh default
    features["Title"] = "Example Title"
    features["DomainTitleMatchScore"] = 50.0  # Contoh default
    features["URLTitleMatchScore"] = 50.0  # Contoh default

    # Favicon, robots, and responsiveness (contoh values)
    features["HasFavicon"] = 1
    features["Robots"] = 1
    features["IsResponsive"] = 1

    # Redirects and pop-ups
    features["NoOfURLRedirect"] = 0
    features["NoOfSelfRedirect"] = 0
    features["HasDescription"] = 1
    features["NoOfPopup"] = 0
    features["NoOfiFrame"] = 0

    # Social network and form submit buttons
    features["HasExternalFormSubmit"] = 0
    features["HasSocialNet"] = 1
    features["HasSubmitButton"] = 1
    features["HasHiddenFields"] = 0
    features["HasPasswordField"] = 1

    # Financial keywords
    features["Bank"] = 0
    features["Pay"] = 0
    features["Crypto"] = 0

    # Copyright and external resources
    features["HasCopyrightInfo"] = 1
    features["NoOfImage"] = 5  # Placeholder
    features["NoOfCSS"] = 3  # Placeholder
    features["NoOfJS"] = 10  # Placeholder
    features["NoOfSelfRef"] = 5  # Placeholder
    features["NoOfEmptyRef"] = 0  # Placeholder
    features["NoOfExternalRef"] = 2  # Placeholder

    return pd.DataFrame([features])


def select_features(X, y, k=40):
    """
    Select top-k features yang paling relevan dengan target menggunakan chi-squared test.

    Parameters:
    X (DataFrame): Feature DataFrame.
    y (Series): Target label.
    k (int): Number of top features to select.

    Returns:
    DataFrame: DataFrame dengan top-k features terpilih.
    """
    # Menghapus fitur string dari DataFrame
    X_numeric = X.select_dtypes(include=["int64", "float64"])

    # Encode target labels jika belum dalam bentuk numerik
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # SelectKBest dengan chi2
    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(X_numeric, y)
    selected_features = X_numeric.columns[selector.get_support(indices=True)]

    print(f"Selected features: {list(selected_features)}")
    return pd.DataFrame(X_new, columns=selected_features)


# Fungsi untuk menghitung entropi URL
def calculate_entropy(url):
    char_counts = np.array([url.count(char) for char in set(url)])
    return entropy(char_counts)


# Fungsi untuk mendeteksi adanya kata kunci phishing
def has_keyword(url, keywords=["login", "bank", "secure", "account", "signin"]):
    return int(any(keyword in url.lower() for keyword in keywords))


# Fungsi untuk menghitung jumlah karakter numerik dalam URL
def count_numeric_chars(url):
    return sum(c.isdigit() for c in url)


# Fungsi untuk menghitung jumlah karakter spesial dalam URL
def count_special_chars(url, special_chars="!@#$%^&*()"):
    return sum(c in special_chars for c in url)


# Fungsi untuk mendeteksi panjang subdomain
def has_long_subdomain(domain):
    parts = domain.split(".")
    if len(parts) > 2:
        return int(len(parts[-3]) > 5)  # contoh panjang subdomain > 5
    return 0


# Pipeline utama untuk menambah fitur-fitur baru ini ke dataset
def extract_additional_features(df):
    # Pastikan 'URL' ada dalam DataFrame
    if 'URL' not in df.columns:
        raise ValueError("Kolom 'URL' tidak ditemukan dalam DataFrame")
    df["entropy"] = df["URL"].apply(calculate_entropy)
    df["has_keyword"] = df["URL"].apply(has_keyword)
    df["numeric_char_count"] = df["URL"].apply(count_numeric_chars)
    df["special_char_count"] = df["URL"].apply(count_special_chars)
    df["long_subdomain"] = df["Domain"].apply(has_long_subdomain)
    return df
