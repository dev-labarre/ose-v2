"""
Generate all company classifications report.
Extracts classifications for ALL companies (not just top/bottom 10).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from typing import Optional, Union

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import from inference module
from inference.predict import load_inference_artifacts


def extract_all_classifications(
    output_dir: Optional[Union[str, Path]] = None,
    base_rate: float = 0.1,
    shrink_threshold: int = 5
) -> Path:
    """
    Extract classifications for ALL companies and save to JSON.
    
    Args:
        output_dir: Directory to save the report (default: project_root/reports)
        base_rate: Base rate for low-evidence shrinkage (default: 0.1)
        shrink_threshold: Article count threshold for shrinkage (default: 5)
        
    Returns:
        Path to the saved JSON file
        
    Raises:
        FileNotFoundError: If required data files are missing
    """
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = Path(output_dir) if output_dir else project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    features_path = project_root / "data" / "features" / "features.parquet"
    labels_path = project_root / "data" / "labels" / "labels.parquet"
    company_basic_path = project_root / "data" / "raw_json" / "01_company_basic_info.json"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    print("Loading inference artifacts...")
    # Load artifacts and data
    preprocessor, model, feature_names = load_inference_artifacts()
    df_features = pd.read_parquet(features_path)

    # Ensure SIREN is string
    if "siren" in df_features.columns:
        df_features["siren"] = df_features["siren"].astype(str)

    # Add any missing expected columns as zeros to avoid KeyErrors
    missing_cols = [c for c in feature_names if c not in df_features.columns]
    for col in missing_cols:
        df_features[col] = 0

    print("Generating predictions for all companies...")
    X = df_features[feature_names]
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    results_df = pd.DataFrame({
        "siren": df_features["siren"],
        "opportunity_score_raw": y_pred_proba,
        "predicted_opportunity": y_pred.astype(int)
    })

    # Merge labels to get article/positive/negative counts and actuals
    if labels_path.exists():
        print("Merging label data...")
        df_labels = pd.read_parquet(labels_path)
        df_labels["siren"] = df_labels["siren"].astype(str)
        df_labels = df_labels.rename(columns={
            "article_count_label": "article_count"
        })
        results_df = results_df.merge(
            df_labels[["siren", "article_count", "positive_count", "negative_count", "is_good_opportunity"]],
            on="siren",
            how="left"
        )
    else:
        results_df["article_count"] = 0
        results_df["positive_count"] = 0
        results_df["negative_count"] = 0
        results_df["is_good_opportunity"] = np.nan

    # Apply shrinkage on low-evidence companies
    print("Applying low-evidence shrinkage...")
    def apply_shrinkage(row):
        score = row["opportunity_score_raw"]
        article_count = row["article_count"]
        if pd.isna(score):
            return np.nan
        article_count = 0 if pd.isna(article_count) else article_count
        if article_count < shrink_threshold:
            shrinkage_factor = article_count / shrink_threshold
            score = shrinkage_factor * score + (1 - shrinkage_factor) * base_rate
        return float(score)

    results_df["opportunity_score"] = results_df.apply(apply_shrinkage, axis=1)

    # Attach basic company metadata (name, siret)
    if company_basic_path.exists():
        print("Merging company basic info...")
        with open(company_basic_path, "r", encoding="utf-8") as f:
            basic_data = json.load(f)
        df_basic = pd.DataFrame(basic_data)
        df_basic["siren"] = df_basic["siren"].astype(str)
        results_df = results_df.merge(
            df_basic[["siren", "company_name", "siret"]],
            on="siren",
            how="left"
        )
    else:
        results_df["company_name"] = None
        results_df["siret"] = None

    # Add contextual numeric fields if available
    contextual_fields = ["effectif", "effectifConsolide", "nbFilialesDirectes"]
    for optional_field in contextual_fields:
        if optional_field in df_features.columns:
            results_df[optional_field] = df_features[optional_field]

    # Aggregate by SIREN to deduplicate panel rows
    print("Aggregating by SIREN...")
    aggregation = {
        "opportunity_score": "mean",
        "predicted_opportunity": "max",
        "article_count": "sum",
        "positive_count": "sum",
        "negative_count": "sum",
        "is_good_opportunity": "max"
    }
    for optional_field in contextual_fields:
        if optional_field in results_df.columns:
            aggregation[optional_field] = "mean"

    results_df = results_df.groupby("siren").agg({
        **aggregation,
        "company_name": "first",
        "siret": "first"
    }).reset_index()
    results_df["predicted_opportunity"] = results_df["predicted_opportunity"].astype(int)

    # Reorder columns for readability
    preferred_order = [
        "siren", "company_name", "opportunity_score", "predicted_opportunity",
        "article_count", "positive_count", "negative_count", "is_good_opportunity",
        "effectif", "effectifConsolide", "nbFilialesDirectes", "siret"
    ]
    existing_order = [c for c in preferred_order if c in results_df.columns]
    remaining_cols = [c for c in results_df.columns if c not in existing_order and not c.endswith("_raw")]
    results_df = results_df[existing_order + remaining_cols]

    # Sort by opportunity score
    results_df = results_df.sort_values("opportunity_score", ascending=False)

    # Save ALL companies to JSON
    output_path = reports_dir / "all_company_classifications.json"
    results_df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    
    print(f"✓ Saved {len(results_df)} company classifications to {output_path}")
    print(f"  - Companies with predicted_opportunity=1: {results_df['predicted_opportunity'].sum()}")
    print(f"  - Companies with predicted_opportunity=0: {(results_df['predicted_opportunity'] == 0).sum()}")

    return output_path


if __name__ == "__main__":
    extract_all_classifications()
