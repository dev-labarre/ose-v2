import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inference import predict


def test_load_inference_artifacts_reads_feature_list(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    inference_dir = project_root / "inference"
    models_dir = project_root / "models"
    inference_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    # Create dummy artifacts
    (inference_dir / "preprocess.joblib").write_text("prep")
    (models_dir / "final_calibrated.joblib").write_text("model")
    feature_list_path = inference_dir / "feature_list.json"
    feature_list_path.write_text(json.dumps({"features": ["a", "b"]}))

    monkeypatch.setattr(predict, "__file__", str(inference_dir / "predict.py"))
    monkeypatch.setattr(predict.joblib, "load", lambda path: Path(path).name)

    preprocessor, model, features = predict.load_inference_artifacts()
    assert preprocessor == "preprocess.joblib"
    assert model == "final_calibrated.joblib"
    assert features == ["a", "b"]


def test_predict_single_and_batch_use_stubbed_model(monkeypatch):
    class _ModelStub:
        def predict_proba(self, X):
            return np.tile([[0.2, 0.8]], (len(X), 1))

    df_features = pd.DataFrame({"siren": ["1", "2"], "feature_a": [1.0, 2.0]})

    monkeypatch.setattr(
        predict,
        "load_inference_artifacts",
        lambda: ("prep", _ModelStub(), ["feature_a"]),
    )

    single = predict.predict_single("1", df_features)
    batch = predict.predict_batch(["1", "2"], df_features)

    assert single["score"] == 0.8
    assert len(batch) == 2 and all("score" in item for item in batch)


@pytest.mark.smoke
def test_predict_all_and_rank_fills_missing_columns(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    predict_root = project_root / "inference"
    predict_root.mkdir(parents=True, exist_ok=True)
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "features").mkdir(parents=True, exist_ok=True)
    (data_dir / "labels").mkdir(parents=True, exist_ok=True)

    df_features = pd.DataFrame({"siren": ["1", "2"], "feature_a": [0.5, 1.5]})
    df_features.to_parquet(data_dir / "features" / "features.parquet", index=False)
    pd.DataFrame(
        {
            "siren": ["1", "2"],
            "article_count_label": [1, 2],
            "positive_count": [1, 0],
            "negative_count": [0, 1],
            "is_good_opportunity": [1, 0],
        }
    ).to_parquet(data_dir / "labels" / "labels.parquet", index=False)

    class _ModelStub:
        def predict_proba(self, X):
            # Expect missing feature to be added as zero; output varies with rows
            return np.array([[0.1, 0.9], [0.6, 0.4]])

        def predict(self, X):
            return np.array([1, 0])

    monkeypatch.setattr(predict, "__file__", str(predict_root / "predict.py"))
    monkeypatch.setattr(
        predict,
        "load_inference_artifacts",
        lambda: ("prep", _ModelStub(), ["feature_a", "feature_missing"]),
    )

    outputs = predict.predict_all_and_rank(output_dir=project_root / "reports", top_n=1)
    assert outputs["top_path"].exists()
    top_json = json.loads(Path(outputs["top_path"]).read_text())
    # Missing feature should be filled with zeros, predictions still produced
    assert len(top_json) == 1
    assert "opportunity_score" in top_json[0]
