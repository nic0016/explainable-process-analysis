#!/usr/bin/env python3
"""
Benchmark - Train all models on all datasets.
Trains 8 models on 6 datasets (48 total trainings).
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

from data.loader import (
    create_eventlog, 
    build_dataframe, 
    convert_to_dataframe,
    load_encoded_with_static_attributes
)
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
csv_path = 'benchmark_metrics.csv'


def load_encoded(xes_path):
    """Load and encode XES dataset."""
    log = create_eventlog(str(xes_path))
    encoded_sequences, normalized_durations, trace_ids, encoder = build_dataframe(log)
    df = convert_to_dataframe(encoded_sequences, normalized_durations, trace_ids, encoder)
    X = df.drop(columns=['Trace_ID', 'Total_Duration_Normalized']).values.astype(np.float32)
    y = df['Total_Duration_Normalized'].values.astype(np.float32)
    return encoded_sequences.astype(np.float32), X, y


def load_encoded_with_static(xes_path):
    """Load dataset with static event attributes."""
    df_static, encoder, static_encoders, static_attributes = load_encoded_with_static_attributes(str(xes_path))
    Xs = df_static.drop(columns=['Trace_ID', 'Total_Duration_Normalized']).values.astype(np.float32)
    ys = df_static['Total_Duration_Normalized'].values.astype(np.float32)
    return Xs, ys


def train_all_models(dataset_name, xes_path):
    """Train all models on a dataset."""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    
    results = {'dataset': dataset_name}
    
    # Load data
    print(f"[1/8] Loading data...")
    encoded, X, y = load_encoded(xes_path)
    in_channels = encoded.shape[2]
    
    # Split for tabular models
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    # 1. XGBoost
    print(f"[2/8] Training XGBoost...")
    hist_xgb = train_xgboost_per_epoch(
        X_train, y_train, X_val, y_val, 
        max_epochs, csv_path, dataset_name, 'XGBoost'
    )
    results['xgb'] = hist_xgb
    print(f"   ✓ XGBoost: R²={hist_xgb['r2_val'][-1]:.4f}, MAE={hist_xgb['mae_val'][-1]:.4f}")
    
    # 2. XGBoost with static attributes
    print(f"[3/8] Training XGBoost+Static...")
    try:
        Xs, ys = load_encoded_with_static(xes_path)
        Xs_train, Xs_val, ys_train, ys_val = train_test_split(Xs, ys, test_size=0.2, random_state=RANDOM_SEED)
        hist_xgb_static = train_xgboost_per_epoch(
            Xs_train, ys_train, Xs_val, ys_val,
            max_epochs, csv_path, dataset_name, 'XGBoost+Static'
        )
        results['xgb_static'] = hist_xgb_static
        print(f"   ✓ XGBoost+Static: R²={hist_xgb_static['r2_val'][-1]:.4f}, MAE={hist_xgb_static['mae_val'][-1]:.4f}")
    except Exception as e:
        print(f"   ⚠ XGBoost+Static failed: {e}")
        results['xgb_static'] = None
    
    # 3. ResNet
    print(f"[4/8] Training ResNet...")
    hist_resnet = train_resnet(
        encoded, y, max_epochs, csv_path, dataset_name, in_channels
    )
    results['resnet'] = hist_resnet
    print(f"   ✓ ResNet: R²={hist_resnet['r2_val'][-1]:.4f}, MAE={hist_resnet['mae_val'][-1]:.4f}")
    
    # 4. BiLSTM
    print(f"[5/8] Training BiLSTM...")
    hist_bilstm = train_bilstm(
        encoded, y, max_epochs, csv_path, dataset_name, input_size=in_channels
    )
    results['bilstm'] = hist_bilstm
    print(f"   ✓ BiLSTM: R²={hist_bilstm['r2_val'][-1]:.4f}, MAE={hist_bilstm['mae_val'][-1]:.4f}")
    
    # 5. TCN
    print(f"[6/8] Training TCN...")
    hist_tcn = train_tcn(
        encoded, y, max_epochs, csv_path, dataset_name, in_channels
    )
    results['tcn'] = hist_tcn
    print(f"   ✓ TCN: R²={hist_tcn['r2_val'][-1]:.4f}, MAE={hist_tcn['mae_val'][-1]:.4f}")
    
    # 6. BERT
    print(f"[7/8] Training BERT...")
    hist_bert = train_bert(
        encoded, y, max_epochs, csv_path, dataset_name
    )
    results['bert'] = hist_bert
    print(f"   ✓ BERT: R²={hist_bert['r2_val'][-1]:.4f}, MAE={hist_bert['mae_val'][-1]:.4f}")
    
    # 7. GPT
    print(f"[8/8] Training GPT...")
    try:
        hist_gpt = train_gpt(
            encoded, y, max_epochs, csv_path, dataset_name
        )
        results['gpt'] = hist_gpt
        print(f"   ✓ GPT: R²={hist_gpt['r2_val'][-1]:.4f}, MAE={hist_gpt['mae_val'][-1]:.4f}")
    except Exception as e:
        print(f"   ⚠ GPT failed: {e}")
        results['gpt'] = None
    
    # 8. TFT (optional)
    print(f"[Bonus] Training TFT...")
    try:
        hist_tft = train_tft(
            encoded, y, max_epochs, csv_path, dataset_name, input_size=in_channels
        )
        results['tft'] = hist_tft
        print(f"   ✓ TFT: R²={hist_tft['r2_val'][-1]:.4f}, MAE={hist_tft['mae_val'][-1]:.4f}")
    except Exception as e:
        print(f"   ⚠ TFT failed: {e}")
        results['tft'] = None
    
    return results


def plot_model_comparison(all_results, model_key, title_prefix, filename):
    """Create comparison plot for a model across all datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs = np.arange(1, max_epochs + 1)
    
    for res in all_results:
        if res.get(model_key) is None:
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
    models = ['xgb', 'xgb_static', 'resnet', 'bilstm', 'tcn', 'bert', 'gpt', 'tft']
    model_names = ['XGBoost', 'XGBoost+Static', 'ResNet', 'BiLSTM', 'TCN', 'BERT', 'GPT', 'TFT']
    
    for res in all_results:
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
    df.to_csv('benchmark_summary.csv', index=False)
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    return df


def main():
    start_time = datetime.now()
    print("="*80)
    print("BENCHMARK - ALL MODELS ON ALL DATASETS")
    print("="*80)
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Models: 8 (XGBoost, XGBoost+Static, ResNet, BiLSTM, TCN, BERT, GPT, TFT)")
    print(f"Epochs: {max_epochs}")
    print(f"Total trainings: ~{len(DATASETS) * 8}")
    print("="*80)
    
    # Ensure results directory exists
    ensure_results_dir()
    
    all_results = []
    
    # Train on all datasets
    for i, (name, xes_path) in enumerate(DATASETS.items(), 1):
        print(f"\n>>> Dataset {i}/{len(DATASETS)}: {name}")
        try:
            results = train_all_models(name, xes_path)
            all_results.append(results)
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create plots
    print("\n" + "="*80)
    print("CREATING COMPARISON PLOTS")
    print("="*80)
    
    plot_model_comparison(all_results, 'xgb', 'XGBoost', 'benchmark_xgboost.png')
    plot_model_comparison(all_results, 'xgb_static', 'XGBoost+Static', 'benchmark_xgboost_static.png')
    plot_model_comparison(all_results, 'resnet', 'ResNet', 'benchmark_resnet.png')
    plot_model_comparison(all_results, 'bilstm', 'BiLSTM', 'benchmark_bilstm.png')
    plot_model_comparison(all_results, 'tcn', 'TCN', 'benchmark_tcn.png')
    plot_model_comparison(all_results, 'bert', 'BERT', 'benchmark_bert.png')
    plot_model_comparison(all_results, 'gpt', 'GPT', 'benchmark_gpt.png')
    plot_model_comparison(all_results, 'tft', 'TFT', 'benchmark_tft.png')
    
    # Summary Table
    create_summary_table(all_results)
    
    # End
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"Start:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Results:")
    print(f"  - benchmark_metrics.csv (all epochs)")
    print(f"  - benchmark_summary.csv (final metrics)")
    print(f"  - benchmark_*.png (8 plots)")
    print("="*80)


if __name__ == "__main__":
    main()
