from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Dans src/features.py, vous pouvez enrichir le pipeline:
# (exemple: no-op pédagogique via FunctionTransformer)
from sklearn.preprocessing import FunctionTransformer

def _clip(X):
    return X.clip(-3, 3)

def build_numeric_preprocess():
    return Pipeline(steps=[
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler()),
("clip", FunctionTransformer(_clip)),
])

def infer_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    '''Détecte colonnes numériques vs catégorielles.'''
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, cat_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    '''
    Prétraitement "propre" (correspond à l'exercice feature/preprocessing) :
    - numériques: imputation médiane + standardisation
    - catégorielles: imputation + one-hot
    '''
    numeric_cols, cat_cols = infer_columns(X)

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )