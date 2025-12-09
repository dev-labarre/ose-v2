import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import builder
import src.train as train_mod


def test_build_pipeline_handles_numeric_categorical_text():
    feature_names = ["num_a", "cat_b", "text_pca_0"]
    numeric_features = ["num_a"]
    categorical_features = ["cat_b"]
    text_features = ["text_pca_0"]

    pipeline = builder.build_pipeline(feature_names, numeric_features, categorical_features, text_features)

    # ColumnTransformer includes all three groups
    preprocessor = pipeline.named_steps["preprocessor"]
    names = [name for name, _, _ in preprocessor.transformers]
    assert set(names) == {"num", "cat", "text"}
    # Feature mask retains ordering
    assert pipeline.named_steps["feature_mask"].feature_names == feature_names


@pytest.mark.smoke
def test_train_smoke_writes_artifacts(tmp_path, monkeypatch):
    # Stub project paths to temp
    monkeypatch.setattr(
        train_mod,
        "get_project_paths",
        lambda: {
            "root": tmp_path,
            "data_raw": tmp_path / "data" / "raw_json",
            "data_features": tmp_path / "data" / "features",
            "data_labels": tmp_path / "data" / "labels",
            "data_external": tmp_path / "data" / "external",
            "models": tmp_path / "models",
            "reports": tmp_path / "reports",
            "inference": tmp_path / "inference",
        },
    )

    # Minimal features/labels
    features_df = pd.DataFrame(
        {
            "siren": ["1", "2", "3", "4"],
            "feature_a": [0.1, 0.2, 0.3, 0.4],
            "feature_b": [1, 0, 1, 0],
            "year": [2022, 2022, 2023, 2023],
        }
    )
    labels_df = pd.DataFrame(
        {
            "siren": ["1", "2", "3", "4"],
            "is_good_opportunity": [1, 0, 1, 0],
        }
    )
    validation = {"validation_checks": {}, "sample_counts": {}}

    # Bypass heavy steps with lightweight stubs
    monkeypatch.setattr(train_mod, "build_temporal_windows", lambda: (features_df, labels_df, validation))

    class _DQT(train_mod.DataQualityTransformer):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    monkeypatch.setattr(train_mod, "DataQualityTransformer", _DQT)
    monkeypatch.setattr(train_mod, "process_inpi_ratios", lambda df, ratios_path, output_path: df)
    monkeypatch.setattr(train_mod, "process_text_features", lambda df, articles, output_path: df)
    monkeypatch.setattr(
        train_mod,
        "select_features",
        lambda X, y, policy, selection, **kwargs: (
            X[["feature_a", "feature_b"]],
            ["feature_a", "feature_b"],
            {"selected_features": 2},
        ),
    )

    class _DummyClassifier:
        def __init__(self):
            self.calibrated_classifiers_ = []
            self.estimator = None

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([[0.4, 0.6]], (len(X), 1))

    class _DummyPipeline:
        def __init__(self):
            self.named_steps = {"preprocessor": "prep", "classifier": _DummyClassifier()}

        def fit(self, X, y):
            return self

        def predict(self, X):
            return self.named_steps["classifier"].predict(X)

        def predict_proba(self, X):
            return self.named_steps["classifier"].predict_proba(X)

    monkeypatch.setattr(train_mod, "build_pipeline", lambda **kwargs: _DummyPipeline())
    monkeypatch.setattr(train_mod, "joblib", type("J", (), {"dump": lambda *args, **kwargs: Path(args[1]).write_text("stub")}))  # type: ignore[arg-type]
    # run_ablation_study is imported from evaluator, not in train module
    from src.models import evaluator as evaluator_mod
    monkeypatch.setattr(evaluator_mod, "run_ablation_study", lambda **kwargs: {})

    # Run main
    train_mod.main()

    # Artifacts expected
    assert (tmp_path / "models" / "final_calibrated.joblib").exists()
    assert (tmp_path / "inference" / "preprocess.joblib").exists()
    assert (tmp_path / "inference" / "feature_list.json").exists()
    assert (tmp_path / "reports" / "metrics.json").exists()
