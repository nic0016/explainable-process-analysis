# Explainable Process Analysis

A comprehensive framework for **Explainable AI (XAI)** in **Process Mining**, comparing 17 explanation methods across 7 model architectures on process duration prediction tasks.

## Overview

This project provides:

- **Data Pipeline**: Load and encode XES event logs for ML models
- **7 Model Architectures**: XGBoost, ResNet, TCN, BiLSTM, BERT, GPT, TFT
- **17 XAI Methods**: Occlusion, LIME, SHAP, Integrated Gradients, Attention methods, and more
- **12 Evaluation Metrics**: Faithfulness, Robustness, and Complexity measures
- **Benchmark Scripts**: Systematic evaluation across datasets and models

## Project Structure

```
explainable-process-analysis/
├── data/                    # Data loading and encoding
│   ├── loader.py            # XES event log loading
│   ├── static_features.py   # Static attribute extraction
│   └── ENCODING.md          # Encoding documentation
├── models/                  # Neural network architectures
│   ├── resnet1d.py          # 1D ResNet
│   ├── tcn.py               # Temporal Convolutional Network
│   ├── bilstm.py            # Bidirectional LSTM
│   ├── bert_encoder.py      # BERT-style Transformer
│   ├── gpt_causal.py        # GPT-style Transformer
│   └── tft_light.py         # Temporal Fusion Transformer
├── xai/                     # Explainability methods
│   ├── methods.py           # 17 attribution methods
│   └── metrics.py           # Evaluation metrics
├── training/                # Training utilities
│   └── trainers.py          # Model training functions
├── scripts/                 # Executable scripts
│   ├── run_benchmark.py     # Basic benchmark
│   └── run_benchmark_static.py  # With static features
├── config.py                # Configuration
├── requirements.txt         # Dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/explainable-process-analysis.git
cd explainable-process-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Format

The framework uses **XES** (eXtensible Event Stream) format for event logs. Each event log contains:

- **Traces**: Sequences of events representing process executions
- **Events**: Individual activities with attributes:
  - `concept:name`: Event/activity name
  - `time:timestamp`: Event timestamp
  - Optional: `org:resource`, `org:role`, etc.

### Supported Datasets

Configure dataset paths in `config.py`:

```python
DATASETS = {
    'BankingData': Path('../Datasets/BankingData.xes'),
    'DomesticDeclarations': Path('../Datasets/DomesticDeclarations.xes'),
    # ... add your datasets
}
```

## Quick Start

### Load and Encode Data

```python
from data.loader import create_eventlog, build_dataframe

# Load XES event log
log = create_eventlog('path/to/dataset.xes')

# Encode to 3D tensor [N, K, M]
# N = traces, K = max sequence length, M = event types
encoded, durations, trace_ids, encoder = build_dataframe(log)
```

### Train a Model

```python
from training.trainers import train_resnet

# Train ResNet model
history = train_resnet(
    encoded_sequences=encoded,
    y=durations,
    max_epochs=50,
    csv_path='results/metrics.csv',
    dataset_name='MyDataset',
    in_channels=encoded.shape[2]
)
```

### Explain Predictions

```python
from xai.methods import get_attributions, METHODS_BY_TYPE

# Get available methods for model type
methods = METHODS_BY_TYPE['cnn']  # ['OC', 'LIM', 'KS', 'VG', 'IG', ...]

# Compute attributions
attributions = get_attributions(
    model=model,
    x=sample,
    method='IG',  # Integrated Gradients
    model_type='cnn'
)
```

### Evaluate Explanations

```python
from xai.metrics import compute_all_metrics

metrics = compute_all_metrics(
    model=model,
    x=sample,
    attributions=attributions,
    attr_func=lambda x: get_attributions(model, x, 'IG', 'cnn')
)
# Returns: faithfulness, robustness, complexity metrics
```

## Models

| Model | Type | Input Format | Description |
|-------|------|--------------|-------------|
| XGBoost | Tree | Flattened [N, K×M] | Gradient boosted trees |
| ResNet | CNN | [N, M, K] | Residual convolutional network |
| TCN | CNN | [N, M, K] | Dilated temporal convolutions |
| BiLSTM | RNN | [N, K, M] | Bidirectional LSTM |
| BERT | Transformer | Token IDs [N, K] | Bidirectional encoder |
| GPT | Transformer | Token IDs [N, K] | Causal decoder |
| TFT | Hybrid | [N, K, M] | LSTM + Attention + GRN |

## XAI Methods

### Universal (Black-Box)
- **OC**: Occlusion
- **LIM**: LIME
- **KS**: Kernel SHAP
- **TS**: TreeSHAP (XGBoost only)

### Gradient-Based (Neural Networks)
- **VG**: Vanilla Gradient
- **IxG**: Input × Gradient
- **GB**: Guided Backprop
- **IG**: Integrated Gradients
- **EG**: Expected Gradients
- **DL**: DeepLIFT
- **DLS**: DeepLIFT SHAP
- **LRP**: Layer-wise Relevance Propagation

### CAM (CNNs only)
- **GC**: GradCAM
- **SC**: ScoreCAM
- **GC++**: GradCAM++

### Attention (Transformers only)
- **RA**: Raw Attention
- **RoA**: Rollout Attention
- **LA**: LRP Attention

## Evaluation Metrics

### Faithfulness
- **Deletion**: Prediction drop when removing important features
- **Insertion**: Prediction recovery when adding important features
- **Infidelity**: How well attributions predict model behavior
- **Faithfulness Correlation**: Correlation with actual prediction changes
- **Pixel Flipping**: Sequential feature removal impact

### Robustness
- **Max Sensitivity**: Maximum attribution change for small perturbations
- **Local Lipschitz**: Lipschitz constant estimate
- **Continuity**: Cosine similarity under perturbations
- **Relative Stability**: Attribution vs prediction change ratio

### Complexity
- **Sparseness**: Fraction of near-zero attributions
- **Effective Complexity**: Number of significant features
- **Gini Coefficient**: Attribution inequality measure

## Running Benchmarks

```bash
# Basic benchmark (all models, all datasets)
python scripts/run_benchmark.py

# Benchmark with static features
python scripts/run_benchmark_static.py
```

## Citation

If you use this framework, please cite:

```bibtex
@thesis{wormann2026xai,
  title={Explainable AI for Process Mining: A Comparative Study},
  author={Wormann, Nico},
  year={2026},
  school={Your University}
}
```

## License

MIT License - see LICENSE file for details.
