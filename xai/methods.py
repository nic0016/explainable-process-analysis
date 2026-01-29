import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from captum.attr import (
    IntegratedGradients, Saliency, InputXGradient, GuidedBackprop,
    DeepLift, DeepLiftShap, GradientShap, LayerGradCam, LRP as CaptumLRP
)
import shap


def occlusion(model, x: np.ndarray, baseline: np.ndarray = None, window_size: int = None) -> np.ndarray:
    if baseline is None:
        baseline = np.zeros_like(x)
    
    if window_size is None:
        window_size = x.shape[-1] if x.ndim > 1 else 1
    
    x_flat = x.flatten()
    baseline_flat = baseline.flatten()
    original_pred = _predict(model, x)
    attributions = np.zeros_like(x_flat)
    
    occluded_batch = []
    window_ranges = []
    
    for i in range(0, len(x_flat), window_size):
        end = min(i + window_size, len(x_flat))
        x_occluded = x_flat.copy()
        x_occluded[i:end] = baseline_flat[i:end]
        occluded_batch.append(x_occluded.reshape(x.shape))
        window_ranges.append((i, end))
    
    occluded_preds = _predict_batch(model, np.stack(occluded_batch), x.shape)
    
    for idx, (i, end) in enumerate(window_ranges):
        attributions[i:end] = original_pred - occluded_preds[idx]
    
    return attributions.reshape(x.shape)


def _predict_batch(model, X_batch: np.ndarray, original_shape: tuple, batch_size: int = 64) -> np.ndarray:
    preds = []
    is_torch = hasattr(model, 'parameters')
    has_embedding = is_torch and hasattr(model, 'embedding') and isinstance(model.embedding, nn.Embedding)
    
    for i in range(0, len(X_batch), batch_size):
        batch = X_batch[i:i+batch_size]
        
        if is_torch:
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                if has_embedding:
                    batch_t = torch.tensor(batch, dtype=torch.float32)
                    if batch_t.dim() == 3:
                        token_ids = batch_t.argmax(dim=-1) + 1
                        token_ids[batch_t.sum(dim=-1) == 0] = 0
                        token_ids = token_ids.long().to(device)
                    else:
                        token_ids = batch_t.long().to(device)
                    padding_mask = (token_ids == 0)
                    pred = model(token_ids, padding_mask=padding_mask).cpu().numpy().flatten()
                else:
                    batch_t = torch.tensor(batch, dtype=torch.float32).to(device)
                    pred = model(batch_t).cpu().numpy().flatten()
        else:
            batch_2d = batch.reshape(len(batch), -1)
            pred = model.predict(batch_2d).flatten()
        
        preds.extend(pred)
    
    return np.array(preds)


def lime_attributions(model, x: np.ndarray, n_samples: int = 50) -> np.ndarray:
    try:
        from lime import lime_tabular
    except ImportError:
        return np.zeros_like(x).flatten()
    
    x_flat = x.flatten()
    
    def predict_fn(X):
        return _predict_batch(model, X.reshape(-1, *x.shape), x.shape)
    
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.zeros((10, len(x_flat))),
        mode='regression',
        verbose=False
    )
    
    max_features = min(len(x_flat), 100)
    exp = explainer.explain_instance(x_flat, predict_fn, num_features=max_features, num_samples=n_samples)
    attr = np.zeros(len(x_flat))
    for idx, val in exp.local_exp[0]:
        attr[idx] = val
    
    return attr.reshape(x.shape)


def kernel_shap(model, x: np.ndarray, background: np.ndarray = None, n_samples: int = 30) -> np.ndarray:
    if x.ndim == 1:
        x_flat = x.reshape(1, -1)
        if background is None:
            background = np.zeros_like(x_flat)
        
        def predict_fn(X):
            return np.array([_predict(model, xi) for xi in X])
        
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(x_flat, nsamples=n_samples)
        return shap_values[0]
    
    seq_len, num_features = x.shape
    if background is None:
        background = np.zeros_like(x)
    
    def grouped_predict(masks):
        preds = []
        for mask in masks:
            x_masked = x.copy()
            for t in range(seq_len):
                if mask[t] < 0.5:
                    x_masked[t, :] = background[t, :]
            preds.append(_predict(model, x_masked))
        return np.array(preds)
    
    bg = np.zeros((1, seq_len))
    explainer = shap.KernelExplainer(grouped_predict, bg)
    shap_values = explainer.shap_values(np.ones((1, seq_len)), nsamples=n_samples)
    
    timestep_attr = shap_values[0]
    expanded = np.zeros_like(x)
    for t in range(seq_len):
        expanded[t, :] = timestep_attr[t]
    
    return expanded


def vanilla_gradient(model: nn.Module, x: torch.Tensor, target_idx: int = 0) -> np.ndarray:
    saliency = Saliency(model)
    attr = saliency.attribute(x, target=target_idx if x.dim() > 1 else None)
    return attr.detach().cpu().numpy()


def input_x_gradient(model: nn.Module, x: torch.Tensor, target_idx: int = 0) -> np.ndarray:
    ixg = InputXGradient(model)
    attr = ixg.attribute(x, target=target_idx if x.dim() > 1 else None)
    return attr.detach().cpu().numpy()


def guided_backprop(model: nn.Module, x: torch.Tensor, target_idx: int = 0) -> np.ndarray:
    gbp = GuidedBackprop(model)
    attr = gbp.attribute(x, target=target_idx if x.dim() > 1 else None)
    return attr.detach().cpu().numpy()


def integrated_gradients(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor = None,
                         n_steps: int = 50, target_idx: int = 0) -> np.ndarray:
    if baseline is None:
        baseline = torch.zeros_like(x)
    
    ig = IntegratedGradients(model)
    attr = ig.attribute(x, baselines=baseline, n_steps=n_steps,
                        target=target_idx if x.dim() > 1 else None)
    return attr.detach().cpu().numpy()


def expected_gradients(model: nn.Module, x: torch.Tensor, background: torch.Tensor,
                       n_samples: int = 50, target_idx: int = 0) -> np.ndarray:
    gs = GradientShap(model)
    attr = gs.attribute(x, baselines=background, n_samples=n_samples,
                        target=target_idx if x.dim() > 1 else None)
    return attr.detach().cpu().numpy()


def deeplift(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor = None,
             target_idx: int = 0) -> np.ndarray:
    if baseline is None:
        baseline = torch.zeros_like(x)
    
    dl = DeepLift(model, multiply_by_inputs=False)
    
    try:
        attr = dl.attribute(x, baselines=baseline, target=target_idx if x.dim() > 1 else None)
        return attr.detach().cpu().numpy()
    except Exception:
        ig = IntegratedGradients(model)
        attr = ig.attribute(x, baselines=baseline, target=target_idx if x.dim() > 1 else None)
        return attr.detach().cpu().numpy()


def deeplift_shap(model: nn.Module, x: torch.Tensor, background: torch.Tensor,
                  target_idx: int = 0) -> np.ndarray:
    dls = DeepLiftShap(model)
    attr = dls.attribute(x, baselines=background, target=target_idx if x.dim() > 1 else None)
    return attr.detach().cpu().numpy()


def lrp(model: nn.Module, x: torch.Tensor, target_idx: int = 0) -> np.ndarray:
    try:
        lrp_attr = CaptumLRP(model)
        attr = lrp_attr.attribute(x, target=target_idx if x.dim() > 1 else None)
        return attr.detach().cpu().numpy()
    except Exception:
        return integrated_gradients(model, x, target_idx=target_idx)


def gradcam(model: nn.Module, x: torch.Tensor, target_layer: nn.Module,
            target_idx: int = 0) -> np.ndarray:
    gc = LayerGradCam(model, target_layer)
    attr = gc.attribute(x, target=target_idx if x.dim() > 1 else None)
    interpolate_mode = 'linear' if x.dim() == 3 else 'bilinear'
    attr = F.interpolate(attr, size=x.shape[2:], mode=interpolate_mode, align_corners=False)
    return attr.detach().cpu().numpy()


def scorecam(model: nn.Module, x: torch.Tensor, target_layer: nn.Module,
             target_idx: int = 0) -> np.ndarray:
    model.eval()
    activations = []
    interpolate_mode = 'linear' if x.dim() == 3 else 'bilinear'
    
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        output = model(x)
        baseline_score = output[0, target_idx] if output.dim() > 1 else output[0]
    handle.remove()
    
    if not activations:
        return np.zeros_like(x.detach().cpu().numpy())
    
    acts = activations[0]
    n_channels = acts.shape[1]
    weights = torch.zeros(n_channels, device=x.device)
    
    for c in range(n_channels):
        mask = F.interpolate(acts[:, c:c+1], size=x.shape[2:], mode=interpolate_mode, align_corners=False)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        masked_input = x * mask
        with torch.no_grad():
            masked_output = model(masked_input)
            score = masked_output[0, target_idx] if masked_output.dim() > 1 else masked_output[0]
        weights[c] = score - baseline_score
    
    weights = F.relu(weights)
    cam = torch.zeros(x.shape[2:], device=x.device)
    for c in range(n_channels):
        cam += weights[c] * F.interpolate(acts[:, c:c+1], size=x.shape[2:], 
                                          mode=interpolate_mode, align_corners=False)[0, 0]
    
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.unsqueeze(0).unsqueeze(0).detach().cpu().numpy()


def gradcam_pp(model: nn.Module, x: torch.Tensor, target_layer: nn.Module,
               target_idx: int = 0) -> np.ndarray:
    model.eval()
    activations = []
    gradients = []
    interpolate_mode = 'linear' if x.dim() == 3 else 'bilinear'
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)
    
    output = model(x)
    score = output[0, target_idx] if output.dim() > 1 else output[0]
    model.zero_grad()
    score.backward(retain_graph=True)
    
    fh.remove()
    bh.remove()
    
    if not activations or not gradients:
        return np.zeros_like(x.detach().cpu().numpy())
    
    acts = activations[0]
    grads = gradients[0]
    
    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3
    sum_acts = torch.sum(acts, dim=(2, 3) if acts.dim() == 4 else (2,), keepdim=True)
    alpha = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + 1e-8)
    weights = torch.sum(alpha * F.relu(grads), dim=(2, 3) if acts.dim() == 4 else (2,), keepdim=True)
    
    cam = torch.sum(weights * acts, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[2:], mode=interpolate_mode, align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam.detach().cpu().numpy()


def _prepare_transformer_input(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    has_embedding = hasattr(model, 'embedding') and isinstance(model.embedding, nn.Embedding)
    
    if has_embedding:
        if x.dim() == 2:
            token_ids = x.argmax(dim=-1) + 1
            token_ids[x.sum(dim=-1) == 0] = 0
            return token_ids.unsqueeze(0).long()
        elif x.dim() == 3:
            token_ids = x.argmax(dim=-1) + 1
            token_ids[x.sum(dim=-1) == 0] = 0
            return token_ids.long()
        else:
            return x.long()
    else:
        if x.dim() == 2:
            return x.unsqueeze(0).float()
        return x.float()


def raw_attention(model: nn.Module, x: torch.Tensor, layer_idx: int = -1) -> np.ndarray:
    model.eval()
    x_prepared = _prepare_transformer_input(model, x)
    
    with torch.no_grad():
        if hasattr(model, 'embedding'):
            padding_mask = (x_prepared == 0)
            model(x_prepared, padding_mask=padding_mask)
        else:
            model(x_prepared)
    
    if not hasattr(model, 'attention_weights') or not model.attention_weights:
        return np.zeros(x.shape[-1] if x.dim() > 1 else len(x))
    
    attn = model.attention_weights[layer_idx]
    attn_squeezed = attn.squeeze(0)
    
    if attn_squeezed.dim() > 1:
        attn_mean = attn_squeezed.mean(dim=0)
    else:
        attn_mean = attn_squeezed
    
    return attn_mean.cpu().numpy()


def rollout_attention(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    x_prepared = _prepare_transformer_input(model, x)
    
    with torch.no_grad():
        if hasattr(model, 'embedding'):
            padding_mask = (x_prepared == 0)
            model(x_prepared, padding_mask=padding_mask)
        else:
            model(x_prepared)
    
    if not hasattr(model, 'attention_weights') or not model.attention_weights:
        return np.zeros(x.shape[-1] if x.dim() > 1 else len(x))
    
    rollout = None
    for attn in model.attention_weights:
        attn_squeezed = attn.squeeze(0)
        eye = torch.eye(attn_squeezed.shape[-1], device=attn_squeezed.device)
        attn_with_residual = 0.5 * attn_squeezed + 0.5 * eye
        attn_normalized = attn_with_residual / (attn_with_residual.sum(dim=-1, keepdim=True) + 1e-8)
        
        if rollout is None:
            rollout = attn_normalized
        else:
            rollout = torch.matmul(rollout, attn_normalized)
    
    if rollout is None:
        return np.zeros(x.shape[-1] if x.dim() > 1 else len(x))
    
    result = rollout[0] if rollout.dim() > 1 else rollout
    return result.cpu().numpy()


def lrp_attention(model: nn.Module, x: torch.Tensor, layer_idx: int = -1) -> np.ndarray:
    attn = raw_attention(model, x, layer_idx)
    
    if np.all(attn == 0):
        return attn
    
    try:
        lrp_scores = lrp(model, x)
        lrp_flat = lrp_scores.flatten()
        
        if len(lrp_flat) == len(attn):
            combined = attn * np.abs(lrp_flat)
            return combined / (combined.sum() + 1e-8)
    except Exception:
        pass
    
    return attn


def _predict(model, x: np.ndarray) -> float:
    if hasattr(model, 'predict'):
        x_input = x.flatten().reshape(1, -1) if x.ndim > 1 else x.reshape(1, -1)
        return float(model.predict(x_input)[0])
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
        
        return float(out.squeeze().cpu().numpy())
    return 0.0


def get_attributions(model, x: Union[np.ndarray, torch.Tensor], method: str,
                     model_type: str = 'tree', **kwargs) -> np.ndarray:
    device = torch.device('cpu')
    if hasattr(model, 'parameters'):
        try:
            device = next(model.parameters()).device
        except StopIteration:
            pass
    
    has_embedding = hasattr(model, 'embedding') and isinstance(model.embedding, nn.Embedding)
    
    if method == 'OC':
        return occlusion(model, x if isinstance(x, np.ndarray) else x.detach().cpu().numpy(), **kwargs)
    elif method == 'LIM':
        x_np = x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
        return lime_attributions(model, x_np, **kwargs)
    elif method == 'KS':
        x_np = x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
        return kernel_shap(model, x_np, **kwargs)
    
    if model_type == 'transformer' and method in ['RA', 'RoA', 'LA']:
        original_shape = x.shape if isinstance(x, np.ndarray) else x.cpu().numpy().shape
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(device)
        elif x.device != device:
            x = x.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if method == 'RA':
            attn_result = raw_attention(model, x, **kwargs)
        elif method == 'RoA':
            attn_result = rollout_attention(model, x, **kwargs)
        elif method == 'LA':
            attn_result = lrp_attention(model, x, **kwargs)
        else:
            attn_result = np.zeros(original_shape[0] if len(original_shape) > 0 else 1)
        
        if len(original_shape) == 2 and len(attn_result) == original_shape[0]:
            expanded = np.zeros(original_shape)
            for t in range(original_shape[0]):
                expanded[t, :] = attn_result[t]
            return expanded
        return attn_result
    
    if model_type in ['transformer', 'rnn', 'cnn']:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(device)
        elif x.device != device:
            x = x.to(device)
        
        is_cnn = hasattr(model, 'stem') or hasattr(model, 'blocks')
        is_rnn = model_type == 'rnn' or hasattr(model, 'lstm')
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            if is_cnn:
                x = x.T.unsqueeze(0)
            else:
                x = x.unsqueeze(0)
        x = x.clone().detach().requires_grad_(True)
        
        was_training = model.training
        if is_rnn:
            model.train()
        
        try:
            if method == 'VG':
                result = vanilla_gradient(model, x)
            elif method == 'IxG':
                result = input_x_gradient(model, x)
            elif method == 'GB':
                result = guided_backprop(model, x)
            elif method == 'IG':
                result = integrated_gradients(model, x, **kwargs)
            elif method == 'EG':
                background = kwargs.get('background', torch.zeros_like(x).expand(10, *x.shape[1:]))
                result = expected_gradients(model, x, background=background, **kwargs)
            elif method == 'DL':
                result = deeplift(model, x, **kwargs)
            elif method == 'DLS':
                background = kwargs.get('background', torch.zeros_like(x).expand(10, *x.shape[1:]))
                result = deeplift_shap(model, x, background=background)
            elif method == 'LRP':
                result = lrp(model, x)
            elif model_type == 'cnn' and 'target_layer' in kwargs:
                if method == 'GC':
                    result = gradcam(model, x, target_layer=kwargs['target_layer'])
                elif method == 'SC':
                    result = scorecam(model, x, target_layer=kwargs['target_layer'])
                elif method == 'GC++':
                    result = gradcam_pp(model, x, target_layer=kwargs['target_layer'])
                else:
                    result = np.zeros(x.detach().cpu().numpy().shape)
            else:
                result = np.zeros(x.detach().cpu().numpy().shape)
            
            if is_rnn and not was_training:
                model.eval()
            return result
        except Exception:
            if is_rnn and not was_training:
                model.eval()
            return np.zeros(x.shape if isinstance(x, np.ndarray) else x.detach().cpu().numpy().shape)
    
    if model_type == 'tree' and method == 'TS':
        try:
            import re
            
            def flatten_to_floats(obj):
                result = []
                if hasattr(obj, 'values'):
                    obj = obj.values
                if hasattr(obj, 'tolist'):
                    obj = obj.tolist()
                if isinstance(obj, (list, tuple)):
                    for item in obj:
                        result.extend(flatten_to_floats(item))
                else:
                    try:
                        result.append(float(obj))
                    except:
                        result.append(0.0)
                return result
            
            explainer = shap.TreeExplainer(model)
            x_input = x.flatten().reshape(1, -1) if isinstance(x, np.ndarray) else x.detach().cpu().numpy().flatten().reshape(1, -1)
            shap_output = explainer.shap_values(x_input)
            
            if isinstance(shap_output, list):
                shap_output = shap_output[0]
            
            float_list = flatten_to_floats(shap_output)
            result = np.array(float_list, dtype=np.float64)
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
            return result
        except Exception:
            return np.zeros(x.flatten().shape)
    
    return np.zeros_like(x if isinstance(x, np.ndarray) else x.detach().cpu().numpy())


METHODS_BY_TYPE = {
    'transformer': ['OC', 'LIM', 'KS', 'VG', 'IxG', 'GB', 'IG', 'EG', 'DL', 'DLS', 'LRP', 'RA', 'RoA', 'LA'],
    'cnn': ['OC', 'LIM', 'KS', 'VG', 'IxG', 'GB', 'IG', 'EG', 'DL', 'DLS', 'LRP', 'GC', 'SC', 'GC++'],
    'rnn': ['OC', 'LIM', 'KS', 'VG', 'IxG', 'GB', 'IG', 'EG', 'DL', 'DLS', 'LRP'],
    'tree': ['OC', 'LIM', 'KS', 'TS']
}

METHOD_NAMES = {
    'OC': 'Occlusion', 'LIM': 'LIME', 'KS': 'Kernel SHAP', 'TS': 'TreeSHAP',
    'VG': 'Vanilla Gradient', 'IxG': 'Input×Gradient', 'GB': 'Guided Backprop',
    'IG': 'Integrated Gradients', 'EG': 'Expected Gradients',
    'DL': 'DeepLIFT', 'DLS': 'DeepLIFT SHAP', 'LRP': 'LRP',
    'GC': 'GradCAM', 'SC': 'ScoreCAM', 'GC++': 'GradCAM++',
    'RA': 'Raw Attention', 'RoA': 'Rollout Attention', 'LA': 'LRP Attention'
}
