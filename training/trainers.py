import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from typing import Dict, List, Tuple
import csv
import os
import time
import psutil


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_resource_usage() -> Dict[str, float]:
    resources = {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'ram_percent': psutil.virtual_memory().percent,
        'ram_used_gb': psutil.virtual_memory().used / (1024**3),
    }
    
    if torch.cuda.is_available():
        try:
            resources['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            resources['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
            resources['gpu_memory_total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            resources['gpu_utilization_percent'] = 0
        except:
            resources['gpu_memory_allocated_mb'] = 0
            resources['gpu_memory_reserved_mb'] = 0
            resources['gpu_memory_total_mb'] = 0
            resources['gpu_utilization_percent'] = 0
    else:
        resources['gpu_memory_allocated_mb'] = 0
        resources['gpu_memory_reserved_mb'] = 0
        resources['gpu_memory_total_mb'] = 0
        resources['gpu_utilization_percent'] = 0
    
    return resources


def log_csv_row(csv_path: str, row: Dict) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'model', 'epoch', 
            'r2_train', 'r2_val', 'mae_train', 'mae_val',
            'loss_train', 'loss_val',
            'n_train', 'n_val', 'training_time_sec', 
            'cpu_percent', 'ram_percent', 'ram_used_gb',
            'gpu_memory_allocated_mb', 'gpu_memory_reserved_mb', 'gpu_memory_total_mb', 'gpu_utilization_percent'
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_xgboost_per_epoch(X_train, y_train, X_val, y_val, max_epochs, csv_path=None, dataset_name=None, model_name='XGBoost'):
    xgb = XGBRegressor(
        n_estimators=0, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective='reg:squarederror', random_state=42, tree_method='hist',
    )
    r2_train, r2_val, mae_train, mae_val = [], [], [], []
    
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        xgb.n_estimators = epoch
        xgb.fit(X_train, y_train, xgb_model=xgb.get_booster() if epoch > 1 else None, verbose=False)
        y_tr = xgb.predict(X_train)
        y_va = xgb.predict(X_val)
        r2_tr = r2_score(y_train, y_tr)
        r2_va = r2_score(y_val, y_va)
        mae_tr = mean_absolute_error(y_train, y_tr)
        mae_va = mean_absolute_error(y_val, y_va)
        loss_tr = np.mean((y_train - y_tr) ** 2)
        loss_va = np.mean((y_val - y_va) ** 2)
        epoch_time = time.time() - epoch_start
        resources = get_resource_usage()
        r2_train.append(r2_tr)
        r2_val.append(r2_va)
        mae_train.append(mae_tr)
        mae_val.append(mae_va)
        if csv_path and dataset_name:
            log_csv_row(csv_path, {
                'dataset': dataset_name, 'model': model_name, 'epoch': epoch,
                'r2_train': r2_tr, 'r2_val': r2_va, 'mae_train': mae_tr, 'mae_val': mae_va,
                'loss_train': float(loss_tr), 'loss_val': float(loss_va),
                'n_train': len(X_train), 'n_val': len(X_val), 'training_time_sec': epoch_time,
                **resources
            })
    return {'r2_train': r2_train, 'r2_val': r2_val, 'mae_train': mae_train, 'mae_val': mae_val}


def _train_pytorch_model(model, train_loader, val_loader, max_epochs, device, csv_path, dataset_name, model_name):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    r2_train, r2_val, mae_train, mae_val = [], [], [], []
    
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        
        model.train()
        tr_preds, tr_targets = [], []
        train_loss_sum = 0.0
        train_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1
            tr_preds.append(pred.detach().cpu().numpy())
            tr_targets.append(yb.detach().cpu().numpy())
        y_tr_pred = np.concatenate(tr_preds).ravel()
        y_tr_true = np.concatenate(tr_targets).ravel()
        loss_tr = train_loss_sum / train_batches

        model.eval()
        va_preds, va_targets = [], []
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item()
                val_batches += 1
                va_preds.append(pred.detach().cpu().numpy())
                va_targets.append(yb.detach().cpu().numpy())
        y_va_pred = np.concatenate(va_preds).ravel()
        y_va_true = np.concatenate(va_targets).ravel()
        loss_va = val_loss_sum / val_batches

        r2_tr = r2_score(y_tr_true, y_tr_pred)
        r2_va = r2_score(y_va_true, y_va_pred)
        mae_tr = mean_absolute_error(y_tr_true, y_tr_pred)
        mae_va = mean_absolute_error(y_va_true, y_va_pred)
        epoch_time = time.time() - epoch_start
        resources = get_resource_usage()
        r2_train.append(r2_tr)
        r2_val.append(r2_va)
        mae_train.append(mae_tr)
        mae_val.append(mae_va)
        if csv_path and dataset_name:
            log_csv_row(csv_path, {
                'dataset': dataset_name, 'model': model_name, 'epoch': epoch,
                'r2_train': r2_tr, 'r2_val': r2_va, 'mae_train': mae_tr, 'mae_val': mae_va,
                'loss_train': loss_tr, 'loss_val': loss_va,
                'n_train': n_train, 'n_val': n_val, 'training_time_sec': epoch_time,
                **resources
            })
    return {'r2_train': r2_train, 'r2_val': r2_val, 'mae_train': mae_train, 'mae_val': mae_val}


def build_cnn_loaders(encoded_sequences, y, batch_size=64, random_state=42):
    E = encoded_sequences.astype(np.float32)
    N, K, M = E.shape
    E = np.transpose(E, (0, 2, 1))
    
    X_train, X_val, y_train, y_val = train_test_split(E, y, test_size=0.2, random_state=random_state)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.reshape(-1, 1)))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.reshape(-1, 1)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return train_loader, val_loader, device


def build_token_loaders(encoded_sequences, y, batch_size=64, random_state=42):
    E = encoded_sequences.astype(np.int32)
    N, K, M = E.shape
    token_ids = np.zeros((N, K), dtype=np.int64)
    for i in range(N):
        for t in range(K):
            row = E[i, t, :]
            if row.sum() > 0:
                token_ids[i, t] = int(row.argmax() + 1)
    
    X_train, X_val, y_train, y_val = train_test_split(token_ids, y, test_size=0.2, random_state=random_state)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.reshape(-1, 1)))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.reshape(-1, 1)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = M + 1
    return train_loader, val_loader, device, vocab_size


def train_resnet(encoded_sequences, y, max_epochs, csv_path, dataset_name, in_channels):
    from models.resnet1d import SmallResNet1D
    train_loader, val_loader, device = build_cnn_loaders(encoded_sequences, y)
    model = SmallResNet1D(in_channels=in_channels, hidden_channels=64, num_blocks=3).to(device)
    return _train_pytorch_model(model, train_loader, val_loader, max_epochs, device, csv_path, dataset_name, 'ResNet')


def train_tcn(encoded_sequences, y, max_epochs, csv_path, dataset_name, in_channels):
    from models.tcn import TemporalConvNet
    train_loader, val_loader, device = build_cnn_loaders(encoded_sequences, y)
    model = TemporalConvNet(in_channels=in_channels, hidden_channels=64, num_blocks=4, kernel_size=3).to(device)
    return _train_pytorch_model(model, train_loader, val_loader, max_epochs, device, csv_path, dataset_name, 'TCN')


def train_bilstm(encoded_sequences, y, max_epochs, csv_path, dataset_name, input_size):
    from models.bilstm import BiLSTMRegressor
    E = encoded_sequences.astype(np.float32)
    
    X_train, X_val, y_train, y_val = train_test_split(E, y, test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.reshape(-1, 1)))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.reshape(-1, 1)))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMRegressor(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.1).to(device)
    return _train_pytorch_model(model, train_loader, val_loader, max_epochs, device, csv_path, dataset_name, 'BiLSTM')


def train_bert(encoded_sequences, y, max_epochs, csv_path, dataset_name):
    from models.bert_encoder import BertStyleRegressor
    train_loader, val_loader, device, vocab_size = build_token_loaders(encoded_sequences, y)
    model = BertStyleRegressor(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2).to(device)
    
    def to_loader(dl):
        xs, ys = [], []
        for xb, yb in dl:
            xs.append(xb)
            ys.append(yb)
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        masks = (xs == 0)
        return DataLoader(TensorDataset(xs, ys, masks), batch_size=64, shuffle=(dl is train_loader))
    
    train_loader2 = to_loader(train_loader)
    val_loader2 = to_loader(val_loader)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    r2_train, r2_val, mae_train, mae_val = [], [], [], []
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_preds, tr_targets = [], []
        for tokens, yb, mask in train_loader2:
            tokens, yb, mask = tokens.to(device), yb.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(tokens, padding_mask=mask)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_preds.append(pred.detach().cpu().numpy())
            tr_targets.append(yb.detach().cpu().numpy())
        y_tr_pred = np.concatenate(tr_preds).ravel()
        y_tr_true = np.concatenate(tr_targets).ravel()

        model.eval()
        va_preds, va_targets = [], []
        with torch.no_grad():
            for tokens, yb, mask in val_loader2:
                tokens, yb, mask = tokens.to(device), yb.to(device), mask.to(device)
                pred = model(tokens, padding_mask=mask)
                va_preds.append(pred.detach().cpu().numpy())
                va_targets.append(yb.detach().cpu().numpy())
        y_va_pred = np.concatenate(va_preds).ravel()
        y_va_true = np.concatenate(va_targets).ravel()

        r2_train.append(r2_score(y_tr_true, y_tr_pred))
        r2_val.append(r2_score(y_va_true, y_va_pred))
        mae_train.append(mean_absolute_error(y_tr_true, y_tr_pred))
        mae_val.append(mean_absolute_error(y_va_true, y_va_pred))

    return {'r2_train': r2_train, 'r2_val': r2_val, 'mae_train': mae_train, 'mae_val': mae_val}


def train_gpt(encoded_sequences, y, max_epochs, csv_path, dataset_name):
    from models.gpt_causal import GPTStyleRegressor
    train_loader, val_loader, device, vocab_size = build_token_loaders(encoded_sequences, y)
    model = GPTStyleRegressor(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2).to(device)

    def to_loader(dl):
        xs, ys = [], []
        for xb, yb in dl:
            xs.append(xb)
            ys.append(yb)
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        masks = (xs == 0)
        return DataLoader(TensorDataset(xs, ys, masks), batch_size=64, shuffle=(dl is train_loader))
    
    train_loader2 = to_loader(train_loader)
    val_loader2 = to_loader(val_loader)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    r2_train, r2_val, mae_train, mae_val = [], [], [], []
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_preds, tr_targets = [], []
        for tokens, yb, mask in train_loader2:
            tokens, yb, mask = tokens.to(device), yb.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(tokens, padding_mask=mask)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_preds.append(pred.detach().cpu().numpy())
            tr_targets.append(yb.detach().cpu().numpy())
        y_tr_pred = np.concatenate(tr_preds).ravel()
        y_tr_true = np.concatenate(tr_targets).ravel()

        model.eval()
        va_preds, va_targets = [], []
        with torch.no_grad():
            for tokens, yb, mask in val_loader2:
                tokens, yb, mask = tokens.to(device), yb.to(device), mask.to(device)
                pred = model(tokens, padding_mask=mask)
                va_preds.append(pred.detach().cpu().numpy())
                va_targets.append(yb.detach().cpu().numpy())
        y_va_pred = np.concatenate(va_preds).ravel()
        y_va_true = np.concatenate(va_targets).ravel()

        r2_train.append(r2_score(y_tr_true, y_tr_pred))
        r2_val.append(r2_score(y_va_true, y_va_pred))
        mae_train.append(mean_absolute_error(y_tr_true, y_tr_pred))
        mae_val.append(mean_absolute_error(y_va_true, y_va_pred))

    return {'r2_train': r2_train, 'r2_val': r2_val, 'mae_train': mae_train, 'mae_val': mae_val}


def train_tft(encoded_sequences, y, max_epochs, csv_path, dataset_name, input_size):
    from models.tft_light import LightTFTRegressor
    E = encoded_sequences.astype(np.float32)
    
    X_train, X_val, y_train, y_val = train_test_split(E, y, test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.reshape(-1, 1)))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.reshape(-1, 1)))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightTFTRegressor(input_size=input_size, d_model=128, nhead=4, num_layers=2).to(device)
    return _train_pytorch_model(model, train_loader, val_loader, max_epochs, device, csv_path, dataset_name, 'TFT')
