from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.reporting import generate_classifications, generate_report


@pytest.mark.smoke
def test_generate_classifications_smoke(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    data_dir = project_root / "data"
    (data_dir / "features").mkdir(parents=True, exist_ok=True)
    (data_dir / "labels").mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"siren": ["1", "2"], "feature_a": [0.1, 0.2]}).to_parquet(
        data_dir / "features" / "features.parquet", index=False
    )
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
            return np.array([[0.2, 0.8], [0.6, 0.4]])

        def predict(self, X):
            return np.array([1, 0])

    monkeypatch.setattr(generate_classifications.predict, "__file__", str(project_root / "inference" / "predict.py"))
    monkeypatch.setattr(
        generate_classifications.predict,
        "load_inference_artifacts",
        lambda: ("prep", _ModelStub(), ["feature_a"]),
    )

    output_path = generate_classifications.extract_all_classifications(output_dir=project_root / "reports", top_n=1)
    assert output_path.exists()
    data = json.loads(Path(output_path).read_text())
    assert len(data) == 2
    assert "opportunity_score" in data[0]


def test_generate_report_placeholder_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_report, "__file__", str(tmp_path / "src" / "reporting" / "generate_report.py"))
    # Should run without raising even when reports dir does not yet exist
    generate_report.generate_full_report()
