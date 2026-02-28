from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris


def load_dataset(cfg: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Charge le dataset selon la configuration.

    Baseline: Iris (toujours disponible).
    Extension: CSV si cfg["data"]["kind"] == "csv".
    """

    data_cfg = cfg.get("data", {})
    kind = data_cfg.get("kind", "iris")

    if kind == "iris":
        X, y = load_iris(return_X_y=True, as_frame=True)
        # y est déjà un Series
        return X, y

    if kind != "csv":
        raise ValueError(f"Unsupported data.kind: {kind}. Use 'iris' or 'csv'.")

    path = data_cfg.get("path")
    target = data_cfg.get("target")
    if not path or not target:
        raise ValueError("For data.kind='csv', you must set data.path and data.target")

    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV: {path}")

    y = df[target]
    X = df.drop(columns=[target])
    return X, y
