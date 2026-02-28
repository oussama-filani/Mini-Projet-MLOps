from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer as Funct

def _clip(X):
    return X.clip(-3, 3)

def build_numeric_preprocess() -> Pipeline:
    """Prétraitement minimal (baseline):

    - imputation médiane
    - standardisation

    (Dans la partie Git feature/, vous pourrez enrichir ce pipeline.)
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clip", Funct(_clip)),
        ]
    )
