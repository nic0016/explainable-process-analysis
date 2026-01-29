#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split

from data.loader import create_eventlog, build_dataframe_with_static
from training.trainers import train_xgboost_per_epoch, train_resnet, train_bilstm, train_tcn, train_bert, train_gpt, train_tft
from config import DATASETS, DEFAULT_EPOCHS, RANDOM_SEED, ensure_results_dir

np.random.seed(RANDOM_SEED)
max_epochs = DEFAULT_EPOCHS
csv_path = 'benchmark_static_metrics.csv'


def train_all_models_static(dataset_name, xes_path):
    print(f"\n{'='*80}\nDataset: {dataset_name} (WITH STATIC FEATURES)\n{'='*80}")
    results = {'dataset': dataset_name}
    
    print(f"[1/8] Loading Event Log...")
    log = create_eventlog(str(xes_path))
    n_total = len(log)
    indices = np.arange(n_total)
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
    
    print(f"[2/8] Loading data with static features...")
    encoded_static, y, trace_ids, encoder, static_encoders, static_attrs = build_dataframe_with_static(log, train_indices=train_indices)
    
    if encoded_static is None:
        print(f"   Error loading {dataset_name}")
        return None
    
    feature_dim = encoded_static.shape[2]
    N, K, MS = encoded_static.shape
    X_flat = encoded_static.reshape(N, K * MS)
    X_train, X_val = X_flat[train_indices], X_flat[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    print(f"[3/8] XGBoost+Static...")
    hist = train_xgboost_per_epoch(X_train, y_train, X_val, y_val, max_epochs, csv_path, dataset_name, 'XGBoost+Static')
    results['xgb_static'] = hist
    print(f"   R²={hist['r2_val'][-1]:.4f}")
    
    print(f"[4/8] ResNet+Static...")
    try:
        hist = train_resnet(encoded_static, y, max_epochs, csv_path, dataset_name + '+Static', in_channels=feature_dim)
        results['resnet_static'] = hist
        print(f"   R²={hist['r2_val'][-1]:.4f}")
    except Exception as e:
        print(f"   Failed: {e}")
        results['resnet_static'] = None
    
    print(f"[5/8] BiLSTM+Static...")
    hist = train_bilstm(encoded_static, y, max_epochs, csv_path, dataset_name + '+Static', input_size=feature_dim)
    results['bilstm_static'] = hist
    print(f"   R²={hist['r2_val'][-1]:.4f}")
    
    print(f"[6/8] TCN+Static...")
    hist = train_tcn(encoded_static, y, max_epochs, csv_path, dataset_name + '+Static', in_channels=feature_dim)
    results['tcn_static'] = hist
    print(f"   R²={hist['r2_val'][-1]:.4f}")
    
    print(f"[7/8] BERT+Static...")
    try:
        hist = train_bert(encoded_static, y, max_epochs, csv_path, dataset_name + '+Static')
        results['bert_static'] = hist
        print(f"   R²={hist['r2_val'][-1]:.4f}")
    except Exception as e:
        print(f"   Failed: {e}")
        results['bert_static'] = None
    
    print(f"[8/8] GPT+Static...")
    try:
        hist = train_gpt(encoded_static, y, max_epochs, csv_path, dataset_name + '+Static')
        results['gpt_static'] = hist
        print(f"   R²={hist['r2_val'][-1]:.4f}")
    except Exception as e:
        print(f"   Failed: {e}")
        results['gpt_static'] = None
    
    print(f"[Bonus] TFT+Static...")
    try:
        hist = train_tft(encoded_static, y, max_epochs, csv_path, dataset_name + '+Static', input_size=feature_dim)
        results['tft_static'] = hist
        print(f"   R²={hist['r2_val'][-1]:.4f}")
    except Exception as e:
        print(f"   Failed: {e}")
        results['tft_static'] = None
    
    return results


def plot_model_comparison(all_results, model_key, title_prefix, filename):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs = np.arange(1, max_epochs + 1)
    
    for res in all_results:
        if res is None or res.get(model_key) is None:
            continue
        name = res['dataset']
        axes[0].plot(epochs, res[model_key]['r2_val'], label=name, linewidth=2)
        axes[1].plot(epochs, res[model_key]['mae_val'], label=name, linewidth=2)
    
    axes[0].set_title(f'{title_prefix} - Validation R²')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('R²')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_title(f'{title_prefix} - Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"   Saved: {filename}")


def create_summary_table(all_results):
    summary = []
    models = ['xgb_static', 'resnet_static', 'bilstm_static', 'tcn_static', 'bert_static', 'gpt_static', 'tft_static']
    model_names = ['XGBoost+Static', 'ResNet+Static', 'BiLSTM+Static', 'TCN+Static', 'BERT+Static', 'GPT+Static', 'TFT+Static']
    
    for res in all_results:
        if res is None:
            continue
        for model_key, model_name in zip(models, model_names):
            if res.get(model_key) is None:
                continue
            hist = res[model_key]
            summary.append({
                'Dataset': res['dataset'], 'Model': model_name,
                'R2_Val': hist['r2_val'][-1], 'MAE_Val': hist['mae_val'][-1],
            })
    
    df = pd.DataFrame(summary)
    df.to_csv('benchmark_static_summary.csv', index=False)
    print(df.to_string(index=False))
    return df


def main():
    start_time = datetime.now()
    print(f"BENCHMARK STATIC - {len(DATASETS)} datasets, 7 models, {max_epochs} epochs")
    
    ensure_results_dir()
    all_results = []
    
    for i, (name, xes_path) in enumerate(DATASETS.items(), 1):
        print(f"\n>>> Dataset {i}/{len(DATASETS)}: {name}")
        try:
            results = train_all_models_static(name, xes_path)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nCreating plots...")
    for key, name in [('xgb_static', 'XGBoost+Static'), ('resnet_static', 'ResNet+Static'), 
                      ('bilstm_static', 'BiLSTM+Static'), ('tcn_static', 'TCN+Static'),
                      ('bert_static', 'BERT+Static'), ('gpt_static', 'GPT+Static'), ('tft_static', 'TFT+Static')]:
        plot_model_comparison(all_results, key, name, f'benchmark_static_{key.replace("_static", "")}.png')
    
    create_summary_table(all_results)
    
    print(f"\nDone in {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
