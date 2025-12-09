"""
Pipeline builder.
ColumnTransformer → FeatureMask → XGBoost → CalibratedClassifierCV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureMask(BaseEstimator, TransformerMixin):
    """Select specific features by name."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        else:
            # If X is array, assume columns match feature_names order
            return X


def build_pipeline(feature_names: List[str],
                  numeric_features: List[str],
                  categorical_features: List[str],
                  text_features: List[str] = None,
                  random_state: int = 42) -> Pipeline:
    """
    Build pipeline: ColumnTransformer → FeatureMask → XGBoost → Calibration.
    
    Args:
        feature_names: Final selected feature names
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        text_features: List of text feature names (PCA components)
        random_state: Random seed
        
    Returns:
        Fitted pipeline
    """
    # Prepare transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # ColumnTransformer
    transformers = []
    
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    if text_features:
        # Text features (PCA components) are already numeric, just scale
        text_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('text', text_transformer, text_features))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # Feature mask (select final features)
    feature_mask = FeatureMask(feature_names=feature_names)
    
    # XGBoost
    xgb = XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        random_state=random_state,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        verbosity=0
    )
    
    # Calibration
    calibrated = CalibratedClassifierCV(
        estimator=xgb,
        method='isotonic',
        cv=3
    )
    
    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_mask', feature_mask),
        ('classifier', calibrated)
    ])
    
    return pipeline


def apply_low_evidence_shrinkage(proba: np.ndarray, article_count: int, 
                                 base_rate: float, threshold: int = 5) -> np.ndarray:
    """
    Apply low-evidence shrinkage: shrink probabilities toward base rate if article_count < threshold.
    
    Args:
        proba: Probability array (shape: [n_samples, 2])
        article_count: Article count for the sample
        base_rate: Base rate (prior probability)
        threshold: Article count threshold (default 5)
        
    Returns:
        Shrunk probability array
    """
    if article_count >= threshold:
        return proba
    
    # Shrink toward base rate
    shrinkage_factor = article_count / threshold  # 0 to 1
    shrunk_proba = proba.copy()
    
    # Shrink positive class probability toward base_rate
    shrunk_proba[:, 1] = (
        shrinkage_factor * proba[:, 1] + 
        (1 - shrinkage_factor) * base_rate
    )
    shrunk_proba[:, 0] = 1 - shrunk_proba[:, 1]
    
    return shrunk_proba


def main():
    """Main function for testing."""
    project_root = Path(__file__).resolve().parents[2]
    feature_list_path = project_root / 'inference' / 'feature_list.json'
    
    if not feature_list_path.exists():
        print("⚠️  Feature list not found. Run feature selection first.")
        return
    
    with open(feature_list_path, 'r') as f:
        feature_data = json.load(f)
        feature_names = feature_data['features']
    
    # Categorize features
    numeric_features = [f for f in feature_names if f.startswith(('text_pca_', 'ca_', 'resultat_', 'fonds_', 'dettes_', 'effectif_'))]
    categorical_features = [f for f in feature_names if f not in numeric_features and not f.startswith('text_pca_')]
    text_features = [f for f in feature_names if f.startswith('text_pca_')]
    
    pipeline = build_pipeline(
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        text_features=text_features
    )
    
    print("✓ Pipeline built successfully")
    print(f"  Features: {len(feature_names)}")
    print(f"    Numeric: {len(numeric_features)}")
    print(f"    Categorical: {len(categorical_features)}")
    print(f"    Text: {len(text_features)}")


if __name__ == "__main__":
    main()
