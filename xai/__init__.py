"""Explainability methods and evaluation metrics for XAI."""

from .methods import (
    get_attributions,
    METHODS_BY_TYPE,
    METHOD_NAMES,
    # Universal methods
    occlusion,
    lime_attributions,
    kernel_shap,
    # Gradient-based methods
    vanilla_gradient,
    input_x_gradient,
    integrated_gradients,
    deeplift,
    # Attention methods
    raw_attention,
    rollout_attention,
    # CAM methods
    gradcam,
)

from .metrics import (
    compute_all_metrics,
    FAITHFULNESS_METRICS,
    ROBUSTNESS_METRICS,
    COMPLEXITY_METRICS,
    # Individual metrics
    deletion_metric,
    insertion_metric,
    infidelity_metric,
    faithfulness_correlation,
    max_sensitivity,
    local_lipschitz,
    sparseness,
    gini_coefficient,
)

__all__ = [
    # Main interface
    "get_attributions",
    "compute_all_metrics",
    "METHODS_BY_TYPE",
    "METHOD_NAMES",
    "FAITHFULNESS_METRICS",
    "ROBUSTNESS_METRICS",
    "COMPLEXITY_METRICS",
    # Universal methods
    "occlusion",
    "lime_attributions",
    "kernel_shap",
    # Gradient methods
    "vanilla_gradient",
    "input_x_gradient",
    "integrated_gradients",
    "deeplift",
    # Attention methods
    "raw_attention",
    "rollout_attention",
    # CAM methods
    "gradcam",
    # Metrics
    "deletion_metric",
    "insertion_metric",
    "infidelity_metric",
    "faithfulness_correlation",
    "max_sensitivity",
    "local_lipschitz",
    "sparseness",
    "gini_coefficient",
]
