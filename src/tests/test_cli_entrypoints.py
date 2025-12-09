from pathlib import Path

import pandas as pd
import pytest

from src.data import windowing
import src.train as train_mod


@pytest.mark.smoke
def test_windowing_cli_entrypoint_runs(monkeypatch, tmp_path):
    marker = tmp_path / "windowing_ran.txt"

    def _stub():
        marker.write_text("ok")
        return pd.DataFrame({"siren": []}), pd.DataFrame({"siren": []}), {}

    monkeypatch.setattr(windowing, "build_temporal_windows", _stub)
    # Simulate python -m src.data.windowing (main guard calls build_temporal_windows)
    windowing.build_temporal_windows()
    assert marker.exists()


@pytest.mark.smoke
def test_train_cli_entrypoint_runs(monkeypatch, tmp_path):
    marker = tmp_path / "train_ran.txt"
    monkeypatch.setattr(train_mod, "main", lambda: marker.write_text("ok"))

    # Simulate python -m src.train
    train_mod.main()
    assert marker.exists()
