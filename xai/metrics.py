"""Evaluation metrics for XAI attributions: Faithfulness, Robustness, Complexity."""

import numpy as np
from scipy.stats import spearmanr
from typing import Callable, Dict
import torch
import torch.nn as nn


# =============================================================================
# FAITHFULNESS METRICS
# =============================================================================

def deletion_metric(model, x: np.ndarray, attributions: np.ndarray, 
                    baseline: np.ndarray = None, steps: int = 10) -> float:
    """Remove features in order of importance, measure prediction drop.
    
    For regression: Returns normalized AUC where higher = better explanation.
    The metric measures how much the prediction changes when important features are removed.
    Range: [0, 1] where 1 means removing top features causes maximum prediction change.
    """
    if baseline is None:
        baseline = np.zeros_like(x)
    
    x_flat = x.flatten()
    attr_flat = attributions.flatten()
    baseline_flat = baseline.flatten()
    
    sorted_idx = np.argsort(-np.abs(attr_flat))
    predictions = []
    original_pred = _predict(model, x)
    
    for step in range(steps + 1):
        n_remove = int(len(x_flat) * step / steps)
        x_modified = x_flat.copy()
        x_modified[sorted_idx[:n_remove]] = baseline_flat[sorted_idx[:n_remove]]
        predictions.append(_predict(model, x_modified.reshape(x.shape)))
    
    pred_range = max(abs(original_pred), abs(predictions[-1] - original_pred), 1e-10)
    
    total_drop = sum(abs(original_pred - p) for p in predictions) / (len(predictions) * pred_range)
    
    return min(1.0, max(0.0, total_drop))


def insertion_metric(model, x: np.ndarray, attributions: np.ndarray,
                     baseline: np.ndarray = None, steps: int = 10) -> float:
    """Add features in order of importance, measure prediction gain.
    
    For regression: Returns normalized AUC where higher = better explanation.
    Measures how quickly the prediction approaches the original when adding important features.
    Range: [0, 1] where 1 means adding top features quickly recovers the original prediction.
    """
    if baseline is None:
        baseline = np.zeros_like(x)
    
    x_flat = x.flatten()
    attr_flat = attributions.flatten()
    baseline_flat = baseline.flatten()
    
    sorted_idx = np.argsort(-np.abs(attr_flat))
    predictions = []
    original_pred = _predict(model, x)
    baseline_pred = _predict(model, baseline)
    
    for step in range(steps + 1):
        n_add = int(len(x_flat) * step / steps)
        x_modified = baseline_flat.copy()
        x_modified[sorted_idx[:n_add]] = x_flat[sorted_idx[:n_add]]
        predictions.append(_predict(model, x_modified.reshape(x.shape)))
    
    pred_range = max(abs(original_pred - baseline_pred), 1e-10)
    
    total_recovery = sum(1 - abs(original_pred - p) / pred_range for p in predictions) / len(predictions)
    
    return min(1.0, max(0.0, total_recovery))


def infidelity_metric(model, x: np.ndarray, attributions: np.ndarray,
                      n_perturbations: int = 50, sigma: float = 0.1) -> float:
    """Measure if attributions explain prediction changes.
    
    Lower values = better (attributions accurately predict model behavior).
    Normalized to typical range [0, 1].
    """
    x_flat = x.flatten()
    attr_flat = attributions.flatten()
    
    attr_norm = attr_flat / (np.linalg.norm(attr_flat) + 1e-10)
    
    original_pred = _predict(model, x)
    infidelity_vals = []
    
    for _ in range(n_perturbations):
        noise = np.random.randn(len(x_flat)) * sigma
        perturbed = (x_flat + noise).reshape(x.shape)
        perturbed_pred = _predict(model, perturbed)
        
        expected_change = np.dot(attr_norm, noise)
        actual_change = perturbed_pred - original_pred
        
        scale = max(abs(actual_change), abs(expected_change), 1e-10)
        infidelity_vals.append(abs(expected_change - actual_change) / scale)
    
    return min(1.0, np.mean(infidelity_vals))


def faithfulness_correlation(model, x: np.ndarray, attributions: np.ndarray,
                             n_perturbations: int = 50) -> float:
    """Correlation between attribution and actual prediction change."""
    x_flat = x.flatten()
    attr_flat = attributions.flatten()
    original_pred = _predict(model, x)
    changes = []
    attr_values = []
    
    for _ in range(n_perturbations):
        idx = np.random.randint(len(x_flat))
        x_perturbed = x_flat.copy()
        x_perturbed[idx] = 0
        perturbed_pred = _predict(model, x_perturbed.reshape(x.shape))
        changes.append(abs(original_pred - perturbed_pred))
        attr_values.append(abs(attr_flat[idx]))
    
    if len(set(changes)) < 2 or len(set(attr_values)) < 2:
        return 0.0
    corr, _ = spearmanr(attr_values, changes)
    return corr if not np.isnan(corr) else 0.0


def pixel_flipping(model, x: np.ndarray, attributions: np.ndarray, steps: int = 10) -> float:
    """Flip top features and measure prediction change."""
    x_flat = x.flatten()
    attr_flat = attributions.flatten()
    sorted_idx = np.argsort(-np.abs(attr_flat))
    original_pred = _predict(model, x)
    changes = []
    
    x_modified = x_flat.copy()
    for step in range(steps):
        n_flip = max(1, len(x_flat) // steps)
        start = step * n_flip
        end = min(start + n_flip, len(x_flat))
        x_modified[sorted_idx[start:end]] = 0
        new_pred = _predict(model, x_modified.reshape(x.shape))
        change = abs(original_pred - new_pred)
        if not np.isnan(change) and not np.isinf(change):
            changes.append(change)
    
    return np.mean(changes) if changes else 0.0


# =============================================================================
# ROBUSTNESS METRICS
# =============================================================================

def max_sensitivity(attr_func: Callable, x: np.ndarray, 
                    n_perturbations: int = 20, sigma: float = 0.01) -> float:
    """Maximum change in attributions for small input perturbations."""
    x_flat = x.flatten()
    original_attr = attr_func(x).flatten()
    max_diff = 0.0
    
    for _ in range(n_perturbations):
        noise = np.random.randn(len(x_flat)) * sigma
        perturbed = (x_flat + noise).reshape(x.shape)
        perturbed_attr = attr_func(perturbed).flatten()
        diff = np.linalg.norm(original_attr - perturbed_attr)
        max_diff = max(max_diff, diff / (np.linalg.norm(noise) + 1e-10))
    
    return max_diff


def local_lipschitz(attr_func: Callable, x: np.ndarray,
                    n_perturbations: int = 20, sigma: float = 0.01) -> float:
    """Estimate local Lipschitz constant of attribution function."""
    x_flat = x.flatten()
    original_attr = attr_func(x).flatten()
    lipschitz_vals = []
    
    for _ in range(n_perturbations):
        noise = np.random.randn(len(x_flat)) * sigma
        perturbed = (x_flat + noise).reshape(x.shape)
        perturbed_attr = attr_func(perturbed).flatten()
        
        attr_diff = np.linalg.norm(original_attr - perturbed_attr)
        input_diff = np.linalg.norm(noise)
        
        if input_diff > 1e-10:
            lipschitz_vals.append(attr_diff / input_diff)
    
    return np.max(lipschitz_vals) if lipschitz_vals else 0.0


def continuity_metric(attr_func: Callable, x: np.ndarray,
                      n_perturbations: int = 20, sigma: float = 0.01) -> float:
    """Measure continuity of attributions with respect to input changes."""
    x_flat = x.flatten()
    original_attr = attr_func(x).flatten()
    attr_changes = []
    
    for _ in range(n_perturbations):
        noise = np.random.randn(len(x_flat)) * sigma
        perturbed = (x_flat + noise).reshape(x.shape)
        perturbed_attr = attr_func(perturbed).flatten()
        cosine_sim = np.dot(original_attr, perturbed_attr) / (
            np.linalg.norm(original_attr) * np.linalg.norm(perturbed_attr) + 1e-10
        )
        attr_changes.append(cosine_sim)
    
    return np.mean(attr_changes)


def relative_stability(attr_func: Callable, model, x: np.ndarray,
                       n_perturbations: int = 20, sigma: float = 0.01) -> float:
    """Relative stability: attribution change vs prediction change ratio.
    
    Lower values = more stable (attributions don't change much relative to predictions).
    Normalized to typical range [0, 1].
    """
    x_flat = x.flatten()
    original_attr = attr_func(x).flatten()
    original_pred = _predict(model, x)
    ratios = []
    
    preds = [original_pred]
    
    for _ in range(n_perturbations):
        noise = np.random.randn(len(x_flat)) * sigma
        perturbed = (x_flat + noise).reshape(x.shape)
        perturbed_attr = attr_func(perturbed).flatten()
        perturbed_pred = _predict(model, perturbed)
        preds.append(perturbed_pred)
        
        attr_change = np.linalg.norm(original_attr - perturbed_attr) / (np.linalg.norm(original_attr) + 1e-10)
        
        pred_diff = abs(original_pred - perturbed_pred)
        
        if pred_diff > 1e-10:
            ratios.append(attr_change / (pred_diff / (sigma + 1e-10)))
    
    if not ratios:
        return 0.0
    
    median_ratio = np.median(ratios)
    return min(1.0, median_ratio / 10.0)


# =============================================================================
# COMPLEXITY METRICS
# =============================================================================

def sparseness(attributions: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of attributions below threshold (higher = sparser)."""
    abs_attr = np.abs(attributions.flatten())
    max_attr = np.max(abs_attr) + 1e-10
    normalized = abs_attr / max_attr
    return np.mean(normalized < threshold)


def effective_complexity(attributions: np.ndarray, threshold: float = 0.01) -> float:
    """Number of features with significant attributions."""
    abs_attr = np.abs(attributions.flatten())
    max_attr = np.max(abs_attr) + 1e-10
    normalized = abs_attr / max_attr
    return np.sum(normalized >= threshold)


def gini_coefficient(attributions: np.ndarray) -> float:
    """Gini coefficient of attributions (inequality measure)."""
    abs_attr = np.abs(attributions.flatten())
    if np.sum(abs_attr) == 0:
        return 0.0
    
    sorted_attr = np.sort(abs_attr)
    n = len(sorted_attr)
    cumulative = np.cumsum(sorted_attr)
    return (2 * np.sum((np.arange(1, n+1) * sorted_attr)) / (n * np.sum(sorted_attr))) - (n + 1) / n


# =============================================================================
# HELPER AND AGGREGATION
# =============================================================================

def _predict(model, x: np.ndarray) -> float:
    """Predict with any model type. Handles GPU models, BERT/GPT token inputs, and CNNs."""
    try:
        if hasattr(model, 'predict'):
            x_input = x.flatten().reshape(1, -1) if x.ndim > 1 else x.reshape(1, -1)
            result = float(model.predict(x_input)[0])
        elif isinstance(model, nn.Module):
            model.eval()
            device = next(model.parameters()).device
            
            if hasattr(model, 'embedding') and isinstance(model.embedding, nn.Embedding):
                x_tensor = torch.tensor(x, dtype=torch.float32)
                if x_tensor.dim() == 2:
                    token_ids = x_tensor.argmax(dim=-1) + 1
                    token_ids[x_tensor.sum(dim=-1) == 0] = 0
                    token_ids = token_ids.unsqueeze(0).long().to(device)
                else:
                    token_ids = x_tensor.unsqueeze(0).long().to(device)
                padding_mask = (token_ids == 0)
                with torch.no_grad():
                    out = model(token_ids, padding_mask=padding_mask)
            else:
                is_cnn = hasattr(model, 'stem') or hasattr(model, 'blocks')
                
                x_tensor = torch.tensor(x, dtype=torch.float32)
                
                if is_cnn:
                    if x_tensor.dim() == 2:
                        x_tensor = x_tensor.T.unsqueeze(0).to(device)
                    else:
                        x_tensor = x_tensor.to(device)
                else:
                    x_tensor = x_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = model(x_tensor)
            
            result = float(out.squeeze().cpu().numpy())
        else:
            return 0.0
        
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return result
    except Exception:
        return 0.0


def compute_all_metrics(model, x: np.ndarray, attributions: np.ndarray,
                        attr_func: Callable = None) -> Dict[str, float]:
    """Compute all metrics for given attributions. Each metric is computed in try-except."""
    results = {}
    
    def safe_compute(name: str, func: Callable, *args, **kwargs) -> float:
        """Safely compute a metric, returning NaN on error."""
        try:
            val = func(*args, **kwargs)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                return np.nan
            return float(val)
        except Exception:
            return np.nan
    
    # Faithfulness
    results['deletion'] = safe_compute('deletion', deletion_metric, model, x, attributions)
    results['insertion'] = safe_compute('insertion', insertion_metric, model, x, attributions)
    results['infidelity'] = safe_compute('infidelity', infidelity_metric, model, x, attributions)
    results['faith_corr'] = safe_compute('faith_corr', faithfulness_correlation, model, x, attributions)
    results['pixel_flip'] = safe_compute('pixel_flip', pixel_flipping, model, x, attributions)
    
    # Robustness (requires attr_func)
    if attr_func is not None:
        results['max_sens'] = safe_compute('max_sens', max_sensitivity, attr_func, x)
        results['lipschitz'] = safe_compute('lipschitz', local_lipschitz, attr_func, x)
        results['continuity'] = safe_compute('continuity', continuity_metric, attr_func, x)
        results['rel_stability'] = safe_compute('rel_stability', relative_stability, attr_func, model, x)
    
    # Complexity
    results['sparseness'] = safe_compute('sparseness', sparseness, attributions)
    results['eff_complexity'] = safe_compute('eff_complexity', effective_complexity, attributions)
    results['gini'] = safe_compute('gini', gini_coefficient, attributions)
    
    return results


FAITHFULNESS_METRICS = ['deletion', 'insertion', 'infidelity', 'faith_corr', 'pixel_flip']
ROBUSTNESS_METRICS = ['max_sens', 'lipschitz', 'continuity', 'rel_stability']
COMPLEXITY_METRICS = ['sparseness', 'eff_complexity', 'gini']
