"""
Feature selection with frozen policy.
Steps:
1. Drop label-cousins
2. Drop >50% missing columns
3. Add PCA(20) text features
4. Model-based selector (XGBoost + permutation)
5. Keep stable top-K, cap text at 30%
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def drop_label_cousins(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop label-cousin features (features derived from same window as labels).
    
    Args:
        df: Features DataFrame
        
    Returns:
        Tuple of (cleaned DataFrame, list of dropped columns)
    """
    label_cousins_prefix = (
        "articles_count", "article_count", "n_positive_signals",
        "n_negative_signals", "n_neutral_signals", "n_code_"
    )
    
    label_cousin_cols = ['positive_count', 'negative_count']
    
    to_drop = []
    
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in label_cousins_prefix):
            to_drop.append(col)
        elif col in label_cousin_cols:
            to_drop.append(col)
    
    to_drop = list(set(to_drop))
    
    if to_drop:
        df_cleaned = df.drop(columns=to_drop, errors='ignore')
        print(f"  ✓ Dropped {len(to_drop)} label-cousin features")
    else:
        df_cleaned = df.copy()
        print(f"  ✓ No label-cousin features found")
    
    return df_cleaned, to_drop


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.50) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with missing rate > threshold.
    
    Args:
        df: Features DataFrame
        threshold: Missing rate threshold (default 0.50 = 50%)
        
    Returns:
        Tuple of (cleaned DataFrame, list of dropped columns)
    """
    to_drop = []
    
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct_missing = (n_missing / len(df)) * 100 if len(df) > 0 else 0
        
        if pct_missing > (threshold * 100):
            to_drop.append(col)
    
    if to_drop:
        df_cleaned = df.drop(columns=to_drop, errors='ignore')
        print(f"  ✓ Dropped {len(to_drop)} columns with >{threshold*100}% missing")
    else:
        df_cleaned = df.copy()
        print(f"  ✓ No columns with >{threshold*100}% missing")
    
    return df_cleaned, to_drop


def identify_text_features(df: pd.DataFrame) -> List[str]:
    """Identify text features (PCA components)."""
    text_features = [col for col in df.columns if col.startswith('text_pca_')]
    return text_features


def model_based_selection(X: pd.DataFrame, y: pd.Series, 
                         text_features: List[str],
                         max_features: int = 200,
                         text_max_pct: float = 0.30,
                         random_state: int = 42) -> List[str]:
    """
    Model-based feature selection using XGBoost importance + permutation.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        text_features: List of text feature names
        max_features: Maximum number of features to select
        text_max_pct: Maximum percentage of text features (default 0.30 = 30%)
        random_state: Random seed
        
    Returns:
        List of selected feature names
    """
    print(f"\n📊 Model-based feature selection...")
    print(f"  Total features: {len(X.columns)}")
    print(f"  Text features: {len(text_features)}")
    
    # Filter to only numeric, bool, and category columns (XGBoost compatible)
    # Exclude object columns (strings, lists, dicts) which XGBoost can't handle
    numeric_cols = X.select_dtypes(include=[np.number, bool, 'category']).columns.tolist()
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Keep text features even if they're numeric (PCA components)
    # Exclude problematic object columns like 'signals', 'articles', 'combined_titles'
    exclude_object_cols = ['signals', 'articles', 'combined_titles']
    valid_object_cols = [col for col in object_cols if col not in exclude_object_cols]
    
    # For now, only use numeric/bool/category columns for feature selection
    # Object columns will be handled by the pipeline's OneHotEncoder later
    X_numeric = X[numeric_cols].copy()
    
    print(f"  Using {len(numeric_cols)} numeric/bool/category features for selection")
    if len(object_cols) > 0:
        print(f"  Excluding {len(object_cols)} object columns (will be encoded in pipeline)")
    
    # Prepare data (simple imputation for model)
    X_imputed = X_numeric.copy()
    for col in X_imputed.select_dtypes(include=[np.number]).columns:
        X_imputed[col] = X_imputed[col].fillna(X_imputed[col].median())
    
    for col in X_imputed.select_dtypes(include=['bool', 'category']).columns:
        X_imputed[col] = X_imputed[col].fillna(X_imputed[col].mode()[0] if len(X_imputed[col].mode()) > 0 else 0)
    
    # Train XGBoost to get feature importance
    print("  Training XGBoost for feature importance...")
    xgb = XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        random_state=random_state,
        n_estimators=100,
        max_depth=5,
        verbosity=0
    )
    
    # Simple train/test split for feature selection
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    xgb.fit(X_train, y_train)
    
    # Get feature importance (only for numeric columns we used)
    feature_importance = pd.DataFrame({
        'feature': numeric_cols,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Permutation importance
    print("  Computing permutation importance...")
    perm_importance = permutation_importance(
        xgb, X_test, y_test, n_repeats=5, random_state=random_state, n_jobs=-1
    )
    
    feature_importance['perm_importance'] = perm_importance.importances_mean
    
    # Combined score (weighted average)
    # Handle division by zero
    importance_max = feature_importance['importance'].max()
    perm_max = feature_importance['perm_importance'].max()
    
    if importance_max > 0:
        importance_normalized = feature_importance['importance'] / importance_max
    else:
        importance_normalized = feature_importance['importance']
    
    if perm_max > 0:
        perm_normalized = feature_importance['perm_importance'] / perm_max
    else:
        perm_normalized = feature_importance['perm_importance']
    
    feature_importance['combined_score'] = (
        0.7 * importance_normalized + 0.3 * perm_normalized
    )
    
    feature_importance = feature_importance.sort_values('combined_score', ascending=False)
    
    # Select top features, respecting text cap
    max_text_features = int(max_features * text_max_pct)
    max_tabular_features = max_features - max_text_features
    
    selected_features = []
    text_selected = 0
    tabular_selected = 0
    
    for _, row in feature_importance.iterrows():
        feat = row['feature']
        
        if feat in text_features:
            if text_selected < max_text_features:
                selected_features.append(feat)
                text_selected += 1
        else:
            if tabular_selected < max_tabular_features:
                selected_features.append(feat)
                tabular_selected += 1
        
        if len(selected_features) >= max_features:
            break
    
    print(f"  ✓ Selected {len(selected_features)} features")
    print(f"    Text: {text_selected} ({text_selected/len(selected_features)*100:.1f}%)")
    print(f"    Tabular: {tabular_selected} ({tabular_selected/len(selected_features)*100:.1f}%)")
    
    return selected_features


def select_features(df: pd.DataFrame, y: pd.Series,
                    output_policy_path: Path,
                    output_selection_path: Path,
                    max_features: int = 200,
                    text_max_pct: float = 0.30,
                    random_state: int = 42) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Feature selection with frozen policy.
    
    Args:
        df: Features DataFrame
        y: Target Series
        output_policy_path: Path to save feature policy JSON
        output_selection_path: Path to save feature selection JSON
        max_features: Maximum number of features to select
        text_max_pct: Maximum percentage of text features
        random_state: Random seed
        
    Returns:
        Tuple of (selected features DataFrame, selected feature names, selection report)
    """
    print("="*80)
    print("FEATURE SELECTION (FROZEN POLICY)")
    print("="*80)
    
    original_cols = len(df.columns)
    df_work = df.copy()
    
    # Step 1: Drop label-cousins
    print("\n1. Dropping label-cousin features...")
    df_work, dropped_cousins = drop_label_cousins(df_work)
    
    # Step 2: Drop >50% missing
    print("\n2. Dropping columns with >50% missing...")
    df_work, dropped_missing = drop_high_missing(df_work, threshold=0.50)
    
    # Step 3: Identify text features (PCA components should already be added)
    print("\n3. Identifying text features...")
    text_features = identify_text_features(df_work)
    print(f"  ✓ Found {len(text_features)} text features (PCA components)")
    
    # Step 4: Model-based selection
    print("\n4. Model-based feature selection...")
    selected_features = model_based_selection(
        df_work, y, text_features, max_features=max_features,
        text_max_pct=text_max_pct, random_state=random_state
    )
    
    # Step 5: Select final features
    # Ensure all selected features exist in df_work
    available_features = [f for f in selected_features if f in df_work.columns]
    if len(available_features) < len(selected_features):
        missing = set(selected_features) - set(available_features)
        print(f"  ⚠️  Warning: {len(missing)} selected features not found in dataframe: {missing}")
    
    df_selected = df_work[available_features].copy()
    selected_features = available_features  # Update to only available features
    
    # Generate reports
    policy = {
        'steps': [
            'drop_label_cousins',
            'drop_high_missing_50pct',
            'add_pca20_text',
            'model_based_selection',
            'cap_text_30pct'
        ],
        'parameters': {
            'max_features': max_features,
            'text_max_pct': text_max_pct,
            'missing_threshold': 0.50
        },
        'dropped_cousins': dropped_cousins,
        'dropped_missing': dropped_missing
    }
    
    selection_report = {
        'original_features': original_cols,
        'after_drop_cousins': len(df_work.columns) + len(dropped_cousins),
        'after_drop_missing': len(df_work.columns) + len(dropped_missing),
        'selected_features': len(selected_features),
        'text_features': len([f for f in selected_features if f in text_features]),
        'tabular_features': len([f for f in selected_features if f not in text_features]),
        'text_percentage': len([f for f in selected_features if f in text_features]) / len(selected_features) * 100,
        'selected_feature_names': selected_features
    }
    
    # Save reports
    with open(output_policy_path, 'w', encoding='utf-8') as f:
        json.dump(policy, f, indent=2, default=str)
    
    with open(output_selection_path, 'w', encoding='utf-8') as f:
        json.dump(selection_report, f, indent=2, default=str)
    
    print(f"\n✓ Feature policy saved to {output_policy_path}")
    print(f"✓ Feature selection saved to {output_selection_path}")
    
    print("\n" + "="*80)
    print("FEATURE SELECTION COMPLETE")
    print("="*80)
    print(f"  Original: {original_cols} features")
    print(f"  Selected: {len(selected_features)} features")
    print(f"    Text: {selection_report['text_features']} ({selection_report['text_percentage']:.1f}%)")
    print(f"    Tabular: {selection_report['tabular_features']}")
    
    return df_selected, selected_features, selection_report


def main():
    """Run feature selection CLI."""
    project_root = Path(__file__).resolve().parents[2]
    features_path = project_root / 'data' / 'features' / 'features.parquet'
    labels_path = project_root / 'data' / 'labels' / 'labels.parquet'
    output_policy_path = project_root / 'reports' / 'feature_policy.json'
    output_selection_path = project_root / 'reports' / 'feature_selection.json'
    
    if not features_path.exists() or not labels_path.exists():
        print("⚠️  Features or labels files not found")
        return
    
    df_features = pd.read_parquet(features_path)
    df_labels = pd.read_parquet(labels_path)
    
    # Merge features and labels
    df = df_features.merge(df_labels[['siren', 'is_good_opportunity']], on='siren', how='inner')
    
    y = df['is_good_opportunity']
    X = df.drop(columns=['is_good_opportunity', 'siren', 'company_name', 'siret'], errors='ignore')
    
    df_selected, selected_features, report = select_features(
        X, y, output_policy_path, output_selection_path
    )
    
    # Save feature list for inference
    inference_feature_path = project_root / 'inference' / 'feature_list.json'
    inference_feature_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inference_feature_path, 'w', encoding='utf-8') as f:
        json.dump({'features': selected_features}, f, indent=2)
    
    print(f"\n✓ Feature list saved to {inference_feature_path}")


if __name__ == "__main__":
    main()
