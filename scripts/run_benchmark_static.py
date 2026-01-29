#!/usr/bin/env python3
"""
Benchmark Static - Train all models with static event attributes.
Trains 7 models on 6 datasets with extended features (Event + org:resource + org:role + ...).
Uses split-aware normalization to prevent data leakage.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split

from data.loader import create_eventlog, build_dataframe_with_static, convert_to_dataframe
from training.trainers import (
    train_xgboost_per_epoch,
    train_resnet,
    train_bilstm,
    train_tcn,
    train_bert,
    train_gpt,
    train_tft,
)
from config import DATASETS, DEFAULT_EPOCHS, RANDOM_SEED, ensure_results_dir

np.random.seed(RANDOM_SEED)

max_epochs = DEFAULT_EPOCHS
csv_path = 'benchmark_static_metrics.csv'


def train_all_models_static(dataset_name, xes_path):
    """Train all models on a dataset WITH static features."""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name} (WITH STATIC FEATURES - NO DATA LEAKAGE)")
    print(f"{'='*80}")
    
    results = {'dataset': dataset_name}
    
    # 1. Load log and create split indices FIRST
    print(f"[1/8] Loading Event Log...")
    log = create_eventlog(str(xes_path))
    n_total = len(log)
    indices = np.arange(n_total)
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )
    
    print(f"  Created consistent splits:")
    print(f"    Train: {len(train_indices)} samples")
    print(f"    Val:   {len(val_indices)} samples")
    
    # 2. Load data WITH static features AND split-aware normalization
    print(f"[2/8] Loading data with static features (split-aware normalization)...")
    encoded_static, y, trace_ids, encoder, static_encoders, static_attrs = build_dataframe_with_static(
        log, train_indices=train_indices
    )
    
    if encoded_static is None:
        print(f"   Error loading {dataset_name}")
        return None
    
    feature_dim = encoded_static.shape[2]  # M+S (Event + Static)
    
    # 3. Create splits using pre-defined indices (consistent across all models!)
    N, K, MS = encoded_static.shape
    X_flat = encoded_static.reshape(N, K * MS)
    X_train, X_val = X_flat[train_indices], X_flat[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    # 1. XGBoost with Static
    print(f"[3/8] Training XGBoost+Static...")
    hist_xgb = train_xgboost_per_epoch(
        X_train, y_train, X_val, y_val, 
        max_epochs, csv_path, dataset_name, 'XGBoost+Static'
    )
    results['xgb_static'] = hist_xgb
    print(f"   ✓ XGBoost+Static: R²={hist_xgb['r2_val'][-1]:.4f}, MAE={hist_xgb['mae_val'][-1]:.4f}")
    
    # 2. ResNet with Static
    print(f"[4/8] Training ResNet+Static...")
    try:
        hist_resnet = train_resnet(
            encoded_static, y, max_epochs, csv_path, dataset_name + '+Static', 
            in_channels=feature_dim
        )
        results['resnet_static'] = hist_resnet
        print(f"   ✓ ResNet+Static: R²={hist_resnet['r2_val'][-1]:.4f}, MAE={hist_resnet['mae_val'][-1]:.4f}")
    except Exception as e:
        print(f"   ⚠ ResNet+Static failed: {e}")
        results['resnet_static'] = None
    
    # 3. BiLSTM with Static
    print(f"[5/8] Training BiLSTM+Static...")
    hist_bilstm = train_bilstm(
        encoded_static, y, max_epochs, csv_path, dataset_name + '+Static', 
        input_size=feature_dim
    )
    results['bilstm_static'] = hist_bilstm
    print(f"   ✓ BiLSTM+Static: R²={hist_bilstm['r2_val'][-1]:.4f}, MAE={hist_bilstm['mae_val'][-1]:.4f}")
    
    # 4. TCN with Static
    print(f"[6/8] Training TCN+Static...")
    hist_tcn = train_tcn(
        encoded_static, y, max_epochs, csv_path, dataset_name + '+Static',
        in_channels=feature_dim
    )
    results['tcn_static'] = hist_tcn
    print(f"   ✓ TCN+Static: R²={hist_tcn['r2_val'][-1]:.4f}, MAE={hist_tcn['mae_val'][-1]:.4f}")
    
    # 5. BERT with Static
    print(f"[7/8] Training BERT+Static...")
    try:
        hist_bert = train_bert(
            encoded_static, y, max_epochs, csv_path, dataset_name + '+Static'
        )
        results['bert_static'] = hist_bert
        print(f"   ✓ BERT+Static: R²={hist_bert['r2_val'][-1]:.4f}, MAE={hist_bert['mae_val'][-1]:.4f}")
    except Exception as e:
        print(f"   ⚠ BERT+Static failed: {e}")
        results['bert_static'] = None
    
    # 6. GPT with Static
    print(f"[8/8] Training GPT+Static...")
    try:
        hist_gpt = train_gpt(
            encoded_static, y, max_epochs, csv_path, dataset_name + '+Static'
        )
        results['gpt_static'] = hist_gpt
        print(f"   ✓ GPT+Static: R²={hist_gpt['r2_val'][-1]:.4f}, MAE={hist_gpt['mae_val'][-1]:.4f}")
    except Exception as e:
        print(f"   ⚠ GPT+Static failed: {e}")
        results['gpt_static'] = None
    
    # 7. TFT with Static (optional)
    print(f"[Optional] Training TFT+Static...")
    try:
        hist_tft = train_tft(
            encoded_static, y, max_epochs, csv_path, dataset_name + '+Static', 
            input_size=feature_dim
        )
        results['tft_static'] = hist_tft
        print(f"   ✓ TFT+Static: R²={hist_tft['r2_val'][-1]:.4f}, MAE={hist_tft['mae_val'][-1]:.4f}")
    except Exception as e:
        print(f"   ⚠ TFT+Static failed: {e}")
        results['tft_static'] = None
    
    return results


def plot_model_comparison(all_results, model_key, title_prefix, filename):
    """Create comparison plot for a model across all datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs = np.arange(1, max_epochs + 1)
    
    for res in all_results:
        if res is None or res.get(model_key) is None:
            continue
        name = res['dataset']
        axes[0].plot(epochs, res[model_key]['r2_val'], label=f'{name}', linewidth=2)
        axes[1].plot(epochs, res[model_key]['mae_val'], label=f'{name}', linewidth=2)
    
    axes[0].set_title(f'{title_prefix} – Validation R² (all datasets)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('R²', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_title(f'{title_prefix} – Validation MAE (all datasets)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE (z-score)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   Saved: {filename}")


def create_summary_table(all_results):
    """Create summary table with final metrics."""
    summary = []
    models = ['xgb_static', 'resnet_static', 'bilstm_static', 'tcn_static', 'bert_static', 'gpt_static', 'tft_static']
    model_names = ['XGBoost+Static', 'ResNet+Static', 'BiLSTM+Static', 'TCN+Static', 'BERT+Static', 'GPT+Static', 'TFT+Static']
    
    for res in all_results:
        if res is None:
            continue
        dataset = res['dataset']
        for model_key, model_name in zip(models, model_names):
            if res.get(model_key) is None:
                continue
            hist = res[model_key]
            summary.append({
                'Dataset': dataset,
                'Model': model_name,
                'R2_Train': hist['r2_train'][-1],
                'R2_Val': hist['r2_val'][-1],
                'MAE_Train': hist['mae_train'][-1],
                'MAE_Val': hist['mae_val'][-1],
            })
    
    df = pd.DataFrame(summary)
    df.to_csv('benchmark_static_summary.csv', index=False)
    print("\n" + "="*80)
    print("SUMMARY TABLE (STATIC FEATURES)")
    print("="*80)
    print(df.to_string(index=False))
    return df


def main():
    start_time = datetime.now()
    print("="*80)
    print("BENCHMARK - ALL MODELS WITH STATIC FEATURES")
    print("="*80)
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Models: 7 (XGBoost, ResNet, BiLSTM, TCN, BERT, GPT, TFT) - all with Static Features")
    print(f"Epochs: {max_epochs}")
    print(f"Total trainings: {len(DATASETS) * 7}")
    print("="*80)
    
    # Ensure results directory exists
    ensure_results_dir()
    
    all_results = []
    
    # Train on all datasets
    for i, (name, xes_path) in enumerate(DATASETS.items(), 1):
        print(f"\n>>> Dataset {i}/{len(DATASETS)}: {name}")
        try:
            results = train_all_models_static(name, xes_path)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create plots
    print("\n" + "="*80)
    print("CREATING COMPARISON PLOTS")
    print("="*80)
    
    plot_model_comparison(all_results, 'xgb_static', 'XGBoost+Static', 'benchmark_static_xgboost.png')
    plot_model_comparison(all_results, 'resnet_static', 'ResNet+Static', 'benchmark_static_resnet.png')
    plot_model_comparison(all_results, 'bilstm_static', 'BiLSTM+Static', 'benchmark_static_bilstm.png')
    plot_model_comparison(all_results, 'tcn_static', 'TCN+Static', 'benchmark_static_tcn.png')
    plot_model_comparison(all_results, 'bert_static', 'BERT+Static', 'benchmark_static_bert.png')
    plot_model_comparison(all_results, 'gpt_static', 'GPT+Static', 'benchmark_static_gpt.png')
    plot_model_comparison(all_results, 'tft_static', 'TFT+Static', 'benchmark_static_tft.png')
    
    # Summary Table
    create_summary_table(all_results)
    
    # End
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("BENCHMARK STATIC FEATURES COMPLETE!")
    print("="*80)
    print(f"Start:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Results:")
    print(f"  - benchmark_static_metrics.csv (all epochs)")
    print(f"  - benchmark_static_summary.csv (final metrics)")
    print(f"  - benchmark_static_*.png (7 plots)")
    print("="*80)


if __name__ == "__main__":
    main()
