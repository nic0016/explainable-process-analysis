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

from data.loader import create_eventlog, build_dataframe, convert_to_dataframe, load_encoded_with_static_attributes
from training.trainers import train_xgboost_per_epoch, train_resnet, train_bilstm, train_tcn, train_bert, train_gpt, train_tft
from config import DATASETS, DEFAULT_EPOCHS, RANDOM_SEED, ensure_results_dir

np.random.seed(RANDOM_SEED)
max_epochs = DEFAULT_EPOCHS
csv_path = 'benchmark_metrics.csv'


def load_encoded(xes_path):
    log = create_eventlog(str(xes_path))
    encoded_sequences, normalized_durations, trace_ids, encoder = build_dataframe(log)
    df = convert_to_dataframe(encoded_sequences, normalized_durations, trace_ids, encoder)
    X = df.drop(columns=['Trace_ID', 'Total_Duration_Normalized']).values.astype(np.float32)
    y = df['Total_Duration_Normalized'].values.astype(np.float32)
    return encoded_sequences.astype(np.float32), X, y


def load_encoded_with_static(xes_path):
    df_static, encoder, static_encoders, static_attributes = load_encoded_with_static_attributes(str(xes_path))
    Xs = df_static.drop(columns=['Trace_ID', 'Total_Duration_Normalized']).values.astype(np.float32)
    ys = df_static['Total_Duration_Normalized'].values.astype(np.float32)
    return Xs, ys


def train_all_models(dataset_name, xes_path):
    print(f"\n{'='*80}\nDataset: {dataset_name}\n{'='*80}")
    results = {'dataset': dataset_name}
    
    print(f"[1/8] Loading data...")
    encoded, X, y = load_encoded(xes_path)
    in_channels = encoded.shape[2]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    print(f"[2/8] XGBoost...")
    hist = train_xgboost_per_epoch(X_train, y_train, X_val, y_val, max_epochs, csv_path, dataset_name, 'XGBoost')
    results['xgb'] = hist
    print(f"   R²={hist['r2_val'][-1]:.4f}")
    
    print(f"[3/8] XGBoost+Static...")
    try:
        Xs, ys = load_encoded_with_static(xes_path)
        Xs_train, Xs_val, ys_train, ys_val = train_test_split(Xs, ys, test_size=0.2, random_state=RANDOM_SEED)
        hist = train_xgboost_per_epoch(Xs_train, ys_train, Xs_val, ys_val, max_epochs, csv_path, dataset_name, 'XGBoost+Static')
        results['xgb_static'] = hist
        print(f"   R²={hist['r2_val'][-1]:.4f}")
    except Exception as e:
        print(f"   Failed: {e}")
        results['xgb_static'] = None
    
    print(f"[4/8] ResNet...")
    hist = train_resnet(encoded, y, max_epochs, csv_path, dataset_name, in_channels)
    results['resnet'] = hist
    print(f"   R²={hist['r2_val'][-1]:.4f}")
    
    print(f"[5/8] BiLSTM...")
    hist = train_bilstm(encoded, y, max_epochs, csv_path, dataset_name, input_size=in_channels)
    results['bilstm'] = hist
    print(f"   R²={hist['r2_val'][-1]:.4f}")
    
    print(f"[6/8] TCN...")
    hist = train_tcn(encoded, y, max_epochs, csv_path, dataset_name, in_channels)
    results['tcn'] = hist
    print(f"   R²={hist['r2_val'][-1]:.4f}")
    
    print(f"[7/8] BERT...")
    hist = train_bert(encoded, y, max_epochs, csv_path, dataset_name)
    results['bert'] = hist
    print(f"   R²={hist['r2_val'][-1]:.4f}")
    
    print(f"[8/8] GPT...")
    try:
        hist = train_gpt(encoded, y, max_epochs, csv_path, dataset_name)
        results['gpt'] = hist
        print(f"   R²={hist['r2_val'][-1]:.4f}")
    except Exception as e:
        print(f"   Failed: {e}")
        results['gpt'] = None
    
    print(f"[Bonus] TFT...")
    try:
        hist = train_tft(encoded, y, max_epochs, csv_path, dataset_name, input_size=in_channels)
        results['tft'] = hist
        print(f"   R²={hist['r2_val'][-1]:.4f}")
    except Exception as e:
        print(f"   Failed: {e}")
        results['tft'] = None
    
    return results


def plot_model_comparison(all_results, model_key, title_prefix, filename):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs = np.arange(1, max_epochs + 1)
    
    for res in all_results:
        if res.get(model_key) is None:
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
    models = ['xgb', 'xgb_static', 'resnet', 'bilstm', 'tcn', 'bert', 'gpt', 'tft']
    model_names = ['XGBoost', 'XGBoost+Static', 'ResNet', 'BiLSTM', 'TCN', 'BERT', 'GPT', 'TFT']
    
    for res in all_results:
        for model_key, model_name in zip(models, model_names):
            if res.get(model_key) is None:
                continue
            hist = res[model_key]
            summary.append({
                'Dataset': res['dataset'], 'Model': model_name,
                'R2_Val': hist['r2_val'][-1], 'MAE_Val': hist['mae_val'][-1],
            })
    
    df = pd.DataFrame(summary)
    df.to_csv('benchmark_summary.csv', index=False)
    print(df.to_string(index=False))
    return df


def main():
    start_time = datetime.now()
    print(f"BENCHMARK - {len(DATASETS)} datasets, 8 models, {max_epochs} epochs")
    
    ensure_results_dir()
    all_results = []
    
    for i, (name, xes_path) in enumerate(DATASETS.items(), 1):
        print(f"\n>>> Dataset {i}/{len(DATASETS)}: {name}")
        try:
            results = train_all_models(name, xes_path)
            all_results.append(results)
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nCreating plots...")
    for key, name in [('xgb', 'XGBoost'), ('resnet', 'ResNet'), ('bilstm', 'BiLSTM'), 
                      ('tcn', 'TCN'), ('bert', 'BERT'), ('gpt', 'GPT'), ('tft', 'TFT')]:
        plot_model_comparison(all_results, key, name, f'benchmark_{key}.png')
    
    create_summary_table(all_results)
    
    print(f"\nDone in {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
