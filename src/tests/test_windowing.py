import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.data import windowing


@pytest.mark.smoke
def test_build_temporal_windows_respects_cutoffs(tmp_path, monkeypatch):
    """
    Ensure build_temporal_windows outputs features/labels in correct windows and
    prevents leakage (features strictly < t0, labels within [t0, t1]).
    """
    # Redirect project_root used inside the module to a temporary tree
    project_root = tmp_path / "proj"
    raw_json_dir = project_root / "data" / "raw_json"
    raw_json_dir.mkdir(parents=True, exist_ok=True)

    # Copy fixture raw JSON inputs
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "raw_json"
    for fname in ["01_company_basic_info.json", "02_financial_data.json", "07_kpi_data.json", "08_signals.json", "09_articles.json"]:
        shutil.copy(fixture_root / fname, raw_json_dir / fname)

    # Avoid parquet dependency issues for this smoke test
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: self.to_csv(Path(path).with_suffix(".csv"), index=index))

    # Force the module to treat the temp tree as its project root
    monkeypatch.setattr(windowing, "__file__", str(project_root / "src" / "data" / "windowing.py"))

    features, labels, validation = windowing.build_temporal_windows()

    # Features use only pre-t0 articles/signals
    assert all(ts < windowing.T0 for ts in pd.to_datetime(
        [art["publishedAt"] for arts in features["articles"] for art in arts]
    ))

    # Labels only from [t0, t1)
    assert validation["validation_checks"]["min_label_date"].startswith(str(windowing.T0.date()))
    assert validation["validation_checks"]["labels_in_window"]

    # Required columns present
    assert {"siren", "combined_titles_label", "is_good_opportunity"}.issubset(labels.columns)
    assert "signals" in features.columns
