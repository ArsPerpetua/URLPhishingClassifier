DATASET_PATH = 'data/processed/PhiUSIIL_Phishing_URL_Dataset.csv'

# Tambahan path buat penyimpanan model atau output lain nanti
MODEL_PATH = 'models/random_forest_model.pkl'
REPORT_PATH = 'reports/'
FIGURE_PATH = 'reports/figures/'

# Parameter untuk Random Forest (nanti bisa di-tweak kalau perlu)
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'random_state': 42
}