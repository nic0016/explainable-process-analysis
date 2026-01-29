from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATASETS_DIR = Path("../Datasets")

DATASETS = {
    'BankingData': DATASETS_DIR / 'BankingData.xes',
    'DomesticDeclarations': DATASETS_DIR / 'DomesticDeclarations.xes',
    'InternationalDeclarations': DATASETS_DIR / 'InternationalDeclarations.xes',
    'PermitLog': DATASETS_DIR / 'PermitLog.xes',
    'PrepaidTravelCost': DATASETS_DIR / 'PrepaidTravelCost.xes',
    'RequestForPayment': DATASETS_DIR / 'RequestForPayment.xes',
}

RESULTS_DIR = PROJECT_DIR / "results"

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
RANDOM_SEED = 42

MODEL_CONFIGS = {
    'XGBoost': {'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
    'ResNet': {'hidden_channels': 64, 'num_blocks': 3},
    'TCN': {'hidden_channels': 64, 'num_blocks': 4, 'kernel_size': 3},
    'BiLSTM': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.1},
    'BERT': {'d_model': 128, 'nhead': 4, 'num_layers': 2},
    'GPT': {'d_model': 128, 'nhead': 4, 'num_layers': 2},
    'TFT': {'d_model': 128, 'nhead': 4, 'num_layers': 2},
}

XAI_DEFAULTS = {
    'n_samples': 50,
    'lime_samples': 50,
    'kernel_shap_samples': 30,
    'ig_steps': 50,
}


def get_dataset_path(dataset_name: str) -> Path:
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASETS[dataset_name]


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR
