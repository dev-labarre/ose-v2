from pathlib import Path

import numpy as np
import pandas as pd

from src.features import text as text_mod
from src.features.financials import process_inpi_ratios
from src.features import selection


def test_process_text_features_adds_pca_columns(monkeypatch, tmp_path):
    # Use a small PCA stub to avoid requiring 10 components with tiny data
    class _PCAStub:
        def __init__(self, n_components, random_state=None):
            self.n_components = n_components  # Use actual n_components
            self.explained_variance_ratio_ = np.array([0.6, 0.4] + [0.0] * (n_components - 2))

        def fit_transform(self, X):
            # Return n_components columns
            result = np.zeros((len(X), self.n_components))
            result[:, 0] = np.arange(len(X))
            result[:, 1] = np.arange(len(X))
            return result

    monkeypatch.setattr(text_mod, "PCA", _PCAStub)

    df_features = pd.DataFrame({"siren": ["100", "200"]})
    df_articles = pd.DataFrame(
        {
            "siren": ["100", "100", "200"],
            "publishedAt": ["2022-12-31", "2023-02-01", "2022-11-01"],
            "title": ["alpha pre", "alpha post", "beta pre"],
        }
    )

    output_path = tmp_path / "text_report.json"
    result = text_mod.process_text_features(df_features.copy(), df_articles, output_path)

    # Only pre-t0 titles should be used for combined_titles
    assert result.loc[result["siren"] == "100", "combined_titles"].iloc[0] == "alpha pre"
    # PCA features present from stub
    assert {"text_pca_0", "text_pca_1"}.issubset(result.columns)
    # Report written
    assert output_path.exists()


def test_process_inpi_ratios_merges_and_flags(tmp_path):
    fixture_root = Path(__file__).resolve().parent / "fixtures"
    ratios_path = fixture_root / "data_external" / "inpi_ratios.parquet"
    output_path = tmp_path / "ratio_summary.json"

    df_features = pd.DataFrame({"siren": ["100", "200"]})
    merged = process_inpi_ratios(df_features, ratios_path, output_path)

    added_cols = [c for c in merged.columns if c.startswith("ca_") or c.startswith("resultat_net_")]
    assert added_cols, "Ratio features should be added"
    # Flags are only created if there are missing values in the ratio features
    # Check that either flags exist (if missing values) or no flags (if no missing values)
    has_flags = any(c.endswith("_was_nan") for c in merged.columns)
    has_missing = merged[added_cols].isna().any().any()
    # If there are missing values, flags should exist; if no missing values, flags may not exist
    assert not has_missing or has_flags, "If ratio features have missing values, _was_nan flags should be created"
    assert output_path.exists()


def test_select_features_uses_deterministic_feature_list(monkeypatch, tmp_path):
    # Stub model-based selection to avoid xgboost cost and ensure determinism
    monkeypatch.setattr(selection, "model_based_selection", lambda X, y, text_features, **kwargs: ["num_a", "text_pca_0"])

    df = pd.DataFrame(
        {
            "num_a": [1, 2, 3, 4],
            "num_b": [0, 1, 0, 1],
            "text_pca_0": [0.1, 0.2, 0.3, 0.4],
            "positive_count": [1, 0, 0, 1],  # label-cousin to drop
        }
    )
    y = pd.Series([1, 0, 1, 0])

    policy_path = tmp_path / "policy.json"
    selection_path = tmp_path / "selection.json"

    df_selected, selected, report = selection.select_features(
        df, y, policy_path, selection_path, max_features=5, text_max_pct=0.5, random_state=0
    )

    assert selected == ["num_a", "text_pca_0"]
    assert list(df_selected.columns) == selected
    assert report["selected_features"] == 2
    assert policy_path.exists() and selection_path.exists()
