import pandas as pd

from src.data.quality import DataQualityTransformer


def test_data_quality_transformer_drops_dupes_and_missing_and_flags():
    df = pd.DataFrame(
        {
            "siren": ["1", "1", "2"],
            "effectif": [10, 10, None],
            "noisy_col": [None, None, None],  # >50% missing and non-critical
            "processedAt": ["2023-01-01", "2023-01-02", "2023-01-03"],
        }
    )

    transformer = DataQualityTransformer()
    transformer.fit(df)
    cleaned = transformer.transform(df)

    # Duplicate row removed (keep last)
    assert len(cleaned) == 2
    # Critical feature effectif preserved and imputed
    assert "effectif" in cleaned.columns
    assert cleaned["effectif"].isna().sum() == 0
    # High-missing noisy column dropped
    assert "noisy_col" not in cleaned.columns
    # Metadata column dropped
    assert "processedAt" not in cleaned.columns
    # _was_nan flag added for effectif
    assert "effectif_was_nan" in cleaned.columns
