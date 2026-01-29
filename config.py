"""Configuration for the Explainable Process Analysis project."""

from pathlib import Path

# =============================================================================
# Paths Configuration
# =============================================================================

# Base directory (adjust this to your local setup)
PROJECT_DIR = Path(__file__).parent

# Dataset directory (relative to project or absolute path)
# Change this to point to your local datasets folder
DATASETS_DIR = Path("../Datasets")

# Available datasets (XES event log files)
DATASETS = {
    'BankingData': DATASETS_DIR / 'BankingData.xes',
    'DomesticDeclarations': DATASETS_DIR / 'DomesticDeclarations.xes',
    'InternationalDeclarations': DATASETS_DIR / 'InternationalDeclarations.xes',
    'PermitLog': DATASETS_DIR / 'PermitLog.xes',
    'PrepaidTravelCost': DATASETS_DIR / 'PrepaidTravelCost.xes',
    'RequestForPayment': DATASETS_DIR / 'RequestForPayment.xes',
}

# Results output directory
RESULTS_DIR = PROJECT_DIR / "results"


# =============================================================================
# Training Defaults
# =============================================================================

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
RANDOM_SEED = 42

# Model-specific defaults
MODEL_CONFIGS = {
    'XGBoost': {
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    },
    'ResNet': {
        'hidden_channels': 64,
        'num_blocks': 3,
    },
    'TCN': {
        'hidden_channels': 64,
        'num_blocks': 4,
        'kernel_size': 3,
    },
    'BiLSTM': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.1,
    },
    'BERT': {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
    },
    'GPT': {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
    },
    'TFT': {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
    },
}


# =============================================================================
# XAI Evaluation Defaults
# =============================================================================

XAI_DEFAULTS = {
    'n_samples': 50,  # Number of samples for XAI evaluation
    'lime_samples': 50,  # Number of samples for LIME
    'kernel_shap_samples': 30,  # Number of samples for Kernel SHAP
    'ig_steps': 50,  # Number of steps for Integrated Gradients
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_dataset_path(dataset_name: str) -> Path:
    """Get the path to a dataset by name."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    return DATASETS[dataset_name]


def ensure_results_dir() -> Path:
    """Create and return the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR
