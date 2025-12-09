from pathlib import Path

import numpy as np
import pandas as pd

from src.features import text as text_mod
from src.features.financials import aggregate_financial_and_signal_scores
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


def test_aggregate_financial_and_signal_scores(tmp_path):
    signals = [
        {"type": {"code": "B"}, "publishedAt": "2022-01-01T00:00:00Z"},
        {"type": {"code": "O"}, "publishedAt": "2022-06-01T00:00:00Z"},
    ]
    df_features = pd.DataFrame(
        {
            "siren": ["111", "111"],
            "year": [2020, 2022],
            "ca_final": [100.0, 120.0],
            "resultat_final": [10.0, 20.0],
            "effectif": [10.0, 11.0],
            "capital_social": [50.0, 50.0],
            "signals": [signals, signals],
            "nbEtabSecondaire": [1, 1],
            "dateCreationUniteLegale": ["2010-01-01", "2010-01-01"],
        }
    )

    report_path = tmp_path / "financial_signal_summary.json"
    enriched = aggregate_financial_and_signal_scores(df_features, report_path)

    expected_cols = {
        "financial_score_last",
        "growth_score_last",
        "profit_score_last",
        "signal_score",
        "decidento_score",
        "OSE_score",
    }
    assert expected_cols.issubset(enriched.columns)
    assert enriched.loc[0, "financial_score_last"] != 0
    assert report_path.exists()
