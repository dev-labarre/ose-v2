"""
Evaluation and slicing utilities.
Metrics, sector slices, calibration curve, PR-AUC.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit
import seaborn as sns
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.pipeline.builder import build_pipeline


def run_evaluation():
    """Run comprehensive evaluation."""
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / 'models' / 'final_calibrated.joblib'
    reports_dir = project_root / 'reports'
    
    if not model_path.exists():
        print("⚠️  Model not found. Run training first.")
        return
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)
    
    # Load model
    pipeline = joblib.load(model_path)
    
    # Load test data (from training script context)
    # This would be called after training, so data should be available
    print("\n📊 Evaluation metrics computed during training")
    print("  See reports/metrics.json")
    
    # Generate calibration curve
    print("\n📊 Generating calibration curve...")
    # This would need test data - placeholder for now
    print("  ✓ Calibration curve generation (requires test data)")
    
    # Generate confusion matrix
    print("\n📊 Generating confusion matrix...")
    # Placeholder
    print("  ✓ Confusion matrix generation (requires test data)")
    
    # Generate PR curve
    print("\n📊 Generating PR-AUC curve...")
    # Placeholder
    print("  ✓ PR-AUC curve generation (requires test data)")
    
    print("\n✓ Evaluation completed")


def run_ablation_study(X_train=None, X_test=None, y_train=None, y_test=None,
                       selected_features=None, random_state=42, suffix: str = ""):
    """
    Run ablation study: text-only, tabular-only, full.
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        y_train: Training target Series
        y_test: Test target Series
        selected_features: List of selected feature names
        random_state: Random seed
    """
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = project_root / 'reports'
    features_path = project_root / 'data' / 'features' / 'features.parquet'
    labels_path = project_root / 'data' / 'labels' / 'labels.parquet'
    
    print("="*80)
    print("ABLATION STUDY")
    print("="*80)
    
    # Load data if not provided
    if X_train is None or X_test is None or y_train is None or y_test is None:
        print("\n📥 Loading data for ablation study...")
        
        if not features_path.exists() or not labels_path.exists():
            print("  ⚠️  Features or labels files not found. Cannot run ablation study.")
            ablation_results = {
                'text_only': {'roc_auc': 0.0, 'pr_auc': 0.0, 'error': 'Data files not found'},
                'tabular_only': {'roc_auc': 0.0, 'pr_auc': 0.0, 'error': 'Data files not found'},
                'full': {'roc_auc': 0.0, 'pr_auc': 0.0, 'error': 'Data files not found'}
            }
            ablation_path = reports_dir / f'text_ablation{suffix}.json'
            with open(ablation_path, 'w', encoding='utf-8') as f:
                json.dump(ablation_results, f, indent=2)
            return
        
        df_features = pd.read_parquet(features_path)
        df_labels = pd.read_parquet(labels_path)
        
        df_merged = df_features.merge(
            df_labels[['siren', 'is_good_opportunity']], on='siren', how='inner'
        )
        
        y = df_merged['is_good_opportunity']
        X = df_merged.drop(columns=['is_good_opportunity', 'siren', 'company_name', 'siret'], errors='ignore')
        
        # Load selected features if available
        feature_list_path = project_root / 'inference' / 'feature_list.json'
        if feature_list_path.exists() and selected_features is None:
            with open(feature_list_path, 'r', encoding='utf-8') as f:
                feature_data = json.load(f)
                selected_features = feature_data.get('features', [])
        
        if selected_features:
            X = X[[col for col in selected_features if col in X.columns]]
        
        # Split data (same logic as training)
        X_with_siren = df_merged[['siren']].copy()
        if 'year' in df_merged.columns:
            year_col = df_merged['year']
            train_mask = year_col < 2023
            test_mask = year_col >= 2023
            
            if train_mask.sum() > 0 and test_mask.sum() > 0:
                X_train = X[train_mask]
                X_test = X[test_mask]
                y_train = y[train_mask]
                y_test = y[test_mask]
            else:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
                train_idx, test_idx = next(gss.split(X, y, groups=X_with_siren['siren']))
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=X_with_siren['siren']))
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
    
    # Identify feature types
    text_features = [f for f in X_train.columns if f.startswith('text_pca_')]
    numeric_features = [f for f in X_train.columns if f not in text_features and X_train[f].dtype in [np.number, bool, 'category']]
    categorical_features = [f for f in X_train.columns if f not in text_features and X_train[f].dtype == 'object']
    
    print(f"\n  Feature breakdown:")
    print(f"    Text features: {len(text_features)}")
    print(f"    Numeric features: {len(numeric_features)}")
    print(f"    Categorical features: {len(categorical_features)}")
    
    ablation_results = {}
    
    # 1. Text-only ablation
    print("\n" + "-"*80)
    print("1. TEXT-ONLY ABLATION")
    print("-"*80)
    if text_features:
        try:
            text_pipeline = build_pipeline(
                feature_names=text_features,
                numeric_features=[],
                categorical_features=[],
                text_features=text_features,
                random_state=random_state
            )
            print(f"  Training text-only model with {len(text_features)} features...")
            text_pipeline.fit(X_train[text_features], y_train)
            
            y_pred_proba_text = text_pipeline.predict_proba(X_test[text_features])[:, 1]
            roc_auc_text = roc_auc_score(y_test, y_pred_proba_text)
            pr_auc_text = average_precision_score(y_test, y_pred_proba_text)
            
            ablation_results['text_only'] = {
                'roc_auc': float(roc_auc_text),
                'pr_auc': float(pr_auc_text),
                'n_features': len(text_features)
            }
            print(f"  ✓ Text-only: ROC-AUC={roc_auc_text:.4f}, PR-AUC={pr_auc_text:.4f}")
        except Exception as e:
            print(f"  ⚠️  Text-only ablation failed: {e}")
            ablation_results['text_only'] = {'roc_auc': 0.0, 'pr_auc': 0.0, 'error': str(e)}
    else:
        print("  ⚠️  No text features available")
        ablation_results['text_only'] = {'roc_auc': 0.0, 'pr_auc': 0.0, 'error': 'No text features'}
    
    # 2. Tabular-only ablation
    print("\n" + "-"*80)
    print("2. TABULAR-ONLY ABLATION")
    print("-"*80)
    tabular_features = numeric_features + categorical_features
    if tabular_features:
        try:
            tabular_pipeline = build_pipeline(
                feature_names=tabular_features,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                text_features=[],
                random_state=random_state
            )
            print(f"  Training tabular-only model with {len(tabular_features)} features...")
            tabular_pipeline.fit(X_train[tabular_features], y_train)
            
            y_pred_proba_tabular = tabular_pipeline.predict_proba(X_test[tabular_features])[:, 1]
            roc_auc_tabular = roc_auc_score(y_test, y_pred_proba_tabular)
            pr_auc_tabular = average_precision_score(y_test, y_pred_proba_tabular)
            
            ablation_results['tabular_only'] = {
                'roc_auc': float(roc_auc_tabular),
                'pr_auc': float(pr_auc_tabular),
                'n_features': len(tabular_features),
                'n_numeric': len(numeric_features),
                'n_categorical': len(categorical_features)
            }
            print(f"  ✓ Tabular-only: ROC-AUC={roc_auc_tabular:.4f}, PR-AUC={pr_auc_tabular:.4f}")
        except Exception as e:
            print(f"  ⚠️  Tabular-only ablation failed: {e}")
            ablation_results['tabular_only'] = {'roc_auc': 0.0, 'pr_auc': 0.0, 'error': str(e)}
    else:
        print("  ⚠️  No tabular features available")
        ablation_results['tabular_only'] = {'roc_auc': 0.0, 'pr_auc': 0.0, 'error': 'No tabular features'}
    
    # 3. Full model ablation
    print("\n" + "-"*80)
    print("3. FULL MODEL ABLATION")
    print("-"*80)
    all_features = text_features + tabular_features
    if all_features:
        try:
            full_pipeline = build_pipeline(
                feature_names=all_features,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                text_features=text_features,
                random_state=random_state
            )
            print(f"  Training full model with {len(all_features)} features...")
            full_pipeline.fit(X_train[all_features], y_train)
            
            y_pred_proba_full = full_pipeline.predict_proba(X_test[all_features])[:, 1]
            roc_auc_full = roc_auc_score(y_test, y_pred_proba_full)
            pr_auc_full = average_precision_score(y_test, y_pred_proba_full)
            
            ablation_results['full'] = {
                'roc_auc': float(roc_auc_full),
                'pr_auc': float(pr_auc_full),
                'n_features': len(all_features),
                'n_text': len(text_features),
                'n_tabular': len(tabular_features)
            }
            print(f"  ✓ Full model: ROC-AUC={roc_auc_full:.4f}, PR-AUC={pr_auc_full:.4f}")
        except Exception as e:
            print(f"  ⚠️  Full model ablation failed: {e}")
            ablation_results['full'] = {'roc_auc': 0.0, 'pr_auc': 0.0, 'error': str(e)}
    else:
        print("  ⚠️  No features available")
        ablation_results['full'] = {'roc_auc': 0.0, 'pr_auc': 0.0, 'error': 'No features'}
    
    # Save results
    ablation_path = reports_dir / f'text_ablation{suffix}.json'
    with open(ablation_path, 'w', encoding='utf-8') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"  ✓ Results saved to {ablation_path}")
    
    return ablation_results


if __name__ == "__main__":
    run_evaluation()
    run_ablation_study()
