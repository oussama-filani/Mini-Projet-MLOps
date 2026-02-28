from pathlib import Path

import yaml


def test_train_yaml_exists_and_has_required_keys():
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "config" / "train.yaml"

    assert cfg_path.exists(), "config/train.yaml is missing"

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict)

    # Sections attendues
    assert "data" in cfg
    assert "split" in cfg
    assert "model" in cfg
    assert "artifacts_dir" in cfg

    # Champs minimum
    assert cfg["data"].get("kind") in {"iris", "csv"}
    assert isinstance(cfg["split"].get("test_size"), (int, float))
    assert isinstance(cfg["split"].get("random_state"), int)
    assert cfg["model"].get("name") == "logistic_regression"
