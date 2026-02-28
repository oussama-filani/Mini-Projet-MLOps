from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import yaml
from sklearn.metrics import classification_report

# Permet "python scripts/evaluate.py" depuis la racine du repo
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_dataset  # noqa: E402


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    cfg_path = REPO_ROOT / "config" / "train.yaml"
    cfg = load_cfg(cfg_path)

    art_dir = REPO_ROOT / cfg.get("artifacts_dir", "artifacts")

    model_path = art_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python scripts/train.py"
        )

    model = joblib.load(model_path)
    X, y = load_dataset(cfg)

    pred = model.predict(X)
    report = classification_report(y, pred, output_dict=True)

    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Evaluate OK: artifacts/report.json")


if __name__ == "__main__":
    main()
