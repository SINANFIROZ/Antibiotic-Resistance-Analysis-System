from __future__ import annotations

from typing import Any


def compute_shap_summary(model: Any, features, feature_names: list[str]) -> list[dict]:
    try:
        import shap  # type: ignore
    except ImportError as error:
        raise RuntimeError('Install ml/requirements-advanced.txt to enable SHAP explanations.') from error

    explainer = shap.Explainer(model)
    shap_values = explainer(features)
    mean_abs_values = abs(shap_values.values).mean(axis=0)
    return [
        {'feature': feature_name, 'impact': float(mean_abs_values[index])}
        for index, feature_name in enumerate(feature_names)
    ]
