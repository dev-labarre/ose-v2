"""
Inference script.
Handles single SIREN and batch predictions without rounding.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from typing import Union, List, Dict, Optional

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def load_inference_artifacts():
    """Load all inference artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    
    preprocessor_path = project_root / 'inference' / 'preprocess.joblib'
    model_path = project_root / 'models' / 'final_calibrated.joblib'
    feature_list_path = project_root / 'inference' / 'feature_list.json'
    
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    
    with open(feature_list_path, 'r') as f:
        feature_data = json.load(f)
        feature_names = feature_data['features']
    
    return preprocessor, model, feature_names


def predict_single(siren: str, df_features: pd.DataFrame) -> Dict:
    """
    Predict for a single SIREN.
    
    Args:
        siren: SIREN identifier
        df_features: Features DataFrame
        
    Returns:
        Dictionary with score and metadata
    """
    preprocessor, model, feature_names = load_inference_artifacts()
    
    # Get company features
    company_data = df_features[df_features['siren'] == siren]
    
    if len(company_data) == 0:
        return {
            'siren': siren,
            'score': None,
            'error': 'SIREN not found'
        }
    
    X = company_data[feature_names]
    
    # Predict
    proba = model.predict_proba(X)[0]
    score = proba[1]  # Positive class probability
    
    # Apply low-evidence shrinkage if needed
    if 'article_count' in company_data.columns:
        article_count = company_data['article_count'].iloc[0]
        if pd.isna(article_count):
            article_count = 0
    else:
        article_count = 0
    if article_count < 5:
        base_rate = 0.1  # Default base rate
        shrinkage_factor = article_count / 5
        score = shrinkage_factor * score + (1 - shrinkage_factor) * base_rate
    
    return {
        'siren': siren,
        'score': float(score),  # No rounding
        'article_count': int(article_count) if pd.notna(article_count) else 0,
        'confidence': 'high' if article_count >= 5 else 'low'
    }


def predict_batch(siren_list: List[str], df_features: pd.DataFrame) -> List[Dict]:
    """
    Predict for a batch of SIRENs.
    
    Args:
        siren_list: List of SIREN identifiers
        df_features: Features DataFrame
        
    Returns:
        List of prediction dictionaries
    """
    results = []
    for siren in siren_list:
        result = predict_single(siren, df_features)
        results.append(result)
    return results


def predict_all_and_rank(
    output_dir: Optional[Union[str, Path]] = None,
    top_n: int = 10,
    base_rate: float = 0.1,
    shrink_threshold: int = 5
) -> Dict[str, Path]:
    """
    Predict for all companies, apply low-evidence shrinkage, and save
    top/bottom rankings to JSON.
    """
    project_root = Path(__file__).resolve().parents[1]
    reports_dir = Path(output_dir) if output_dir else project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    features_path = project_root / "data" / "features" / "features.parquet"
    labels_path = project_root / "data" / "labels" / "labels.parquet"
    company_basic_path = project_root / "data" / "raw_json" / "01_company_basic_info.json"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

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

    # Apply shrinkage on low-evidence companies (same logic as predict_single)
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
        # Ensure columns exist even if merge didn't happen
        if "company_name" not in results_df.columns:
            results_df["company_name"] = None
        if "siret" not in results_df.columns:
            results_df["siret"] = None

    # Add a couple of contextual numeric fields if available
    contextual_fields = ["effectif", "effectifConsolide", "nbFilialesDirectes"]
    for optional_field in contextual_fields:
        if optional_field in df_features.columns:
            results_df[optional_field] = df_features[optional_field]

    # Aggregate by SIREN to deduplicate panel rows
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
    
    # Only include metadata columns in aggregation if they exist
    agg_dict = aggregation.copy()
    if "company_name" in results_df.columns:
        agg_dict["company_name"] = "first"
    if "siret" in results_df.columns:
        agg_dict["siret"] = "first"

    results_df = results_df.groupby("siren").agg(agg_dict).reset_index()
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

    # Sort and select
    results_df = results_df.sort_values("opportunity_score", ascending=False)
    top_df = results_df.head(top_n)
    bottom_df = results_df.sort_values("opportunity_score", ascending=True).head(top_n)

    # Save JSON outputs
    top_path = reports_dir / "top_10_companies.json"
    bottom_path = reports_dir / "bottom_10_companies.json"
    top_df.to_json(top_path, orient="records", indent=2, force_ascii=False)
    bottom_df.to_json(bottom_path, orient="records", indent=2, force_ascii=False)

    print(f"✓ Saved top {top_n} companies to {top_path}")
    print(f"✓ Saved bottom {top_n} companies to {bottom_path}")

    return {"top_path": top_path, "bottom_path": bottom_path}


def main():
    """Generate example single prediction and full rankings."""
    project_root = Path(__file__).resolve().parents[1]
    features_path = project_root / 'data' / 'features' / 'features.parquet'
    
    if not features_path.exists():
        print("⚠️  Features file not found")
        return
    
    df_features = pd.read_parquet(features_path)
    
    # Example: predict for first SIREN
    siren = df_features['siren'].iloc[0]
    result = predict_single(siren, df_features)
    print(f"Prediction for SIREN {siren}:")
    print(json.dumps(result, indent=2))

    # Full rankings
    predict_all_and_rank()


if __name__ == "__main__":
    main()
