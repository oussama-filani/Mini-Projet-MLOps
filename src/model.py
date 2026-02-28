from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_model(cfg: dict) -> LogisticRegression:
    """Construit le modèle selon la configuration.

    Baseline: LogisticRegression uniquement.
    """

    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "logistic_regression")

    if name != "logistic_regression":
        raise ValueError(f"Model not supported in baseline: {name}")

    max_iter = int(model_cfg.get("max_iter", 2000))
    return LogisticRegression(max_iter=max_iter)
