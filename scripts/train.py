from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Permet "python scripts/train.py" depuis la racine du repo
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_dataset  # noqa: E402
from src.features import build_numeric_preprocess  # noqa: E402
from src.model import build_model  # noqa: E402


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    cfg_path = REPO_ROOT / "config" / "train.yaml"
    cfg = load_cfg(cfg_path)

    art_dir = REPO_ROOT / cfg.get("artifacts_dir", "artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(cfg)

    split_cfg = cfg.get("split", {})
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=float(split_cfg.get("test_size", 0.2)),
        random_state=int(split_cfg.get("random_state", 42)),
        stratify=y,
    )

    preprocess = build_numeric_preprocess()
    model = build_model(cfg)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, average="macro"))

    # Artefacts
    joblib.dump(pipe, art_dir / "model.joblib")
    (art_dir / "metrics.json").write_text(
        json.dumps({"accuracy": acc, "f1_macro": f1}, indent=2),
        encoding="utf-8",
    )
    save_confusion_matrix(yte, pred, art_dir / "confusion_matrix.png")

    print("Train OK:", {"accuracy": acc, "f1_macro": f1})


if __name__ == "__main__":
    main()
