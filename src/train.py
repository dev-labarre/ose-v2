"""
Main training script.
Orchestrates the complete pipeline:
1. Temporal windowing
2. Data quality
3. External ratios
4. Text processing
5. Feature selection
6. Model training & calibration
7. Evaluation
8. SHAP
9. Leak tests
10. Inference package
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
import hashlib
from pathlib import Path
from datetime import datetime
import joblib

# Set seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)

# Import modules
from src.data.windowing import build_temporal_windows
from src.data.quality import DataQualityTransformer, generate_missing_summary, generate_kept_features_report
from src.external.fetch_inpi_ratios import fetch_all_ratios
from src.features.financials import process_inpi_ratios
from src.features.text import process_text_features
from src.features.selection import select_features
from src.pipeline.builder import build_pipeline, apply_low_evidence_shrinkage
from sklearn.model_selection import GroupShuffleSplit
import json as json_module


def get_project_paths():
    """Get project directory paths."""
    project_root = Path(__file__).resolve().parents[1]
    return {
        'root': project_root,
        'data_raw': project_root / 'data' / 'raw_json',
        'data_features': project_root / 'data' / 'features',
        'data_labels': project_root / 'data' / 'labels',
        'data_external': project_root / 'data' / 'external',
        'models': project_root / 'models',
        'reports': project_root / 'reports',
        'inference': project_root / 'inference'
    }


def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute hash of data for reproducibility."""
    data_str = df.to_string()
    return hashlib.md5(data_str.encode()).hexdigest()[:16]


def save_run_manifest(paths: dict, validation: dict, selection_report: dict):
    """Save run manifest with version, seed, data hash, etc."""
    manifest = {
        'version': 'current',
        'seed': RANDOM_SEED,
        'timestamp': datetime.now().isoformat(),
        'temporal_windows': {
            't0': '2023-01-01',
            't1': '2024-01-01',
            'feature_window': '< 2023-01-01',
            'label_window': '[2023-01-01 → 2024-01-01]'
        },
        'split_policy': 'temporal (fallback: GroupShuffleSplit by SIREN)',
        'feature_policy': {
            'steps': [
                'drop_label_cousins',
                'drop_high_missing_50pct',
                'add_pca10_text',
                'model_based_selection',
                'cap_text_30pct'
            ],
            'max_features': 200,
            'text_max_pct': 0.30
        },
        'model': {
            'type': 'XGBoost + CalibratedClassifierCV',
            'calibration': 'isotonic (cv=3)',
            'random_state': RANDOM_SEED
        },
        'validation': validation,
        'feature_selection': selection_report
    }
    
    manifest_path = paths['reports'] / 'run_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print(f"\n✓ Run manifest saved to {manifest_path}")


def main():
    """Main training pipeline."""
    print("="*80)
    print("OSE - COMPLETE TRAINING PIPELINE")
    print("="*80)
    
    paths = get_project_paths()
    
    # Ensure directories exist
    for dir_path in paths.values():
        if isinstance(dir_path, Path):
            dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Temporal windowing (skip if features already exist)
        features_path = paths['data_features'] / 'features.parquet'
        labels_path = paths['data_labels'] / 'labels.parquet'
        
        if features_path.exists() and labels_path.exists():
            print("\n" + "="*80)
            print("STEP 1: LOADING EXISTING FEATURES")
            print("="*80)
            print("  ✓ Features file found, loading from disk...")
            df_features = pd.read_parquet(features_path)
            df_labels = pd.read_parquet(labels_path)
            
            # Load validation if available
            validation_path = paths['reports'] / 'window_validation.json'
            if validation_path.exists():
                with open(validation_path, 'r', encoding='utf-8') as f:
                    validation = json.load(f)
            else:
                validation = {}
            
            print(f"  ✓ Loaded {len(df_features)} feature records")
            print(f"  ✓ Loaded {len(df_labels)} label records")
        else:
            print("\n" + "="*80)
            print("STEP 1: TEMPORAL WINDOWING")
            print("="*80)
            df_features, df_labels, validation = build_temporal_windows()
        
        # Step 2: Data quality
        print("\n" + "="*80)
        print("STEP 2: DATA QUALITY")
        print("="*80)
        quality_transformer = DataQualityTransformer(
            remove_duplicates=True,
            duplicate_keep='last',
            drop_high_missing=True,
            missing_threshold=0.50
        )
        quality_transformer.fit(df_features)
        df_features_cleaned = quality_transformer.transform(df_features)
        
        # Generate quality reports
        generate_missing_summary(
            df_features, df_features_cleaned, quality_transformer,
            paths['reports'] / 'missing_summary.json'
        )
        generate_kept_features_report(
            df_features_cleaned,
            paths['reports'] / 'kept_features.json'
        )
        
        # Step 3: External ratios (if available)
        print("\n" + "="*80)
        print("STEP 3: EXTERNAL INPI RATIOS")
        print("="*80)
        ratios_path = paths['data_external'] / 'inpi_ratios.parquet'
        if ratios_path.exists():
            df_features_cleaned = process_inpi_ratios(
                df_features_cleaned, ratios_path,
                paths['reports'] / 'ratio_summary.json'
            )
        else:
            print("  ⚠️  Ratios file not found. Run 'make fetch_ratios' first.")
        
        # Step 4: Text processing
        print("\n" + "="*80)
        print("STEP 4: TEXT PROCESSING (FastText → PCA(10))")
        print("="*80)
        articles_path = paths['data_raw'] / '09_articles.json'
        if articles_path.exists():
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles_data = json_module.load(f)
            df_articles = pd.DataFrame(articles_data)
            df_features_cleaned = process_text_features(
                df_features_cleaned, df_articles,
                paths['reports'] / 'text_ablation.json'
            )
        else:
            print("  ⚠️  Articles file not found")
        
        # Save updated features
        features_path = paths['data_features'] / 'features.parquet'
        df_features_cleaned.to_parquet(features_path, index=False)
        
        # Step 5: Feature selection
        print("\n" + "="*80)
        print("STEP 5: FEATURE SELECTION")
        print("="*80)
        df_merged = df_features_cleaned.merge(
            df_labels[['siren', 'is_good_opportunity']], on='siren', how='inner'
        )
        
        y = df_merged['is_good_opportunity']
        X = df_merged.drop(columns=['is_good_opportunity', 'siren', 'company_name', 'siret'], errors='ignore')
        
        df_selected, selected_features, selection_report = select_features(
            X, y,
            paths['reports'] / 'feature_policy.json',
            paths['reports'] / 'feature_selection.json'
        )
        
        # Step 6: Model training
        print("\n" + "="*80)
        print("STEP 6: MODEL TRAINING & CALIBRATION")
        print("="*80)
        
        # Categorize features
        text_features = [f for f in selected_features if f.startswith('text_pca_')]
        numeric_features = [f for f in selected_features if f not in text_features and not X[f].dtype == 'object']
        categorical_features = [f for f in selected_features if f not in text_features and X[f].dtype == 'object']
        
        # Build pipeline
        pipeline = build_pipeline(
            feature_names=selected_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            text_features=text_features,
            random_state=RANDOM_SEED
        )
        
        # Temporal split
        X_with_siren = df_merged[['siren'] + selected_features].copy()
        X_features = X_with_siren[selected_features]
        
        # Try temporal split first
        if 'year' in df_merged.columns:
            year_col = df_merged['year']
            train_mask = year_col < 2023
            test_mask = year_col >= 2023
            
            if train_mask.sum() > 0 and test_mask.sum() > 0:
                X_train = X_features[train_mask]
                X_test = X_features[test_mask]
                y_train = y[train_mask]
                y_test = y[test_mask]
                print(f"  ✓ Using TEMPORAL split (train < 2023, test >= 2023)")
            else:
                # Fallback to grouped split
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
                train_idx, test_idx = next(gss.split(X_features, y, groups=X_with_siren['siren']))
                X_train = X_features.iloc[train_idx]
                X_test = X_features.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                print(f"  ✓ Using GROUPED split by SIREN")
        else:
            # Fallback to grouped split
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
            train_idx, test_idx = next(gss.split(X_features, y, groups=X_with_siren['siren']))
            X_train = X_features.iloc[train_idx]
            X_test = X_features.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            print(f"  ✓ Using GROUPED split by SIREN")
        
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        # Train
        print("\n  Training pipeline...")
        pipeline.fit(X_train, y_train)
        print("  ✓ Training completed")
        
        # Save models
        raw_model_path = paths['models'] / 'final_raw.joblib'
        calibrated_model_path = paths['models'] / 'final_calibrated.joblib'
        
        # Extract raw XGBoost (before calibration)
        # CalibratedClassifierCV stores base estimators in calibrated_classifiers_
        calibrated_classifier = pipeline.named_steps['classifier']
        
        # Access base estimator from calibrated classifier
        # In CalibratedClassifierCV, each calibrated classifier has an 'estimator' attribute
        raw_xgb = None
        if hasattr(calibrated_classifier, 'calibrated_classifiers_') and len(calibrated_classifier.calibrated_classifiers_) > 0:
            # Each _CalibratedClassifier has an 'estimator' attribute (not 'base_estimator')
            calibrated_item = calibrated_classifier.calibrated_classifiers_[0]
            if hasattr(calibrated_item, 'estimator'):
                raw_xgb = calibrated_item.estimator
            elif hasattr(calibrated_item, 'base_estimator'):
                raw_xgb = calibrated_item.base_estimator
        
        # Fallback: try estimator attribute on CalibratedClassifierCV itself
        if raw_xgb is None and hasattr(calibrated_classifier, 'estimator'):
            raw_xgb = calibrated_classifier.estimator
        
        if raw_xgb is not None:
            joblib.dump(raw_xgb, raw_model_path)
            print(f"  ✓ Saved raw model to {raw_model_path}")
        else:
            print(f"  ⚠️  Could not extract raw model, skipping raw model save")
        
        # Save calibrated pipeline
        joblib.dump(pipeline, calibrated_model_path)
        print(f"  ✓ Saved calibrated model to {calibrated_model_path}")
        
        # Save preprocessor for inference
        preprocessor = pipeline.named_steps['preprocessor']
        inference_preprocessor_path = paths['inference'] / 'preprocess.joblib'
        joblib.dump(preprocessor, inference_preprocessor_path)
        print(f"  ✓ Saved preprocessor to {inference_preprocessor_path}")
        
        # Save feature list
        feature_list_path = paths['inference'] / 'feature_list.json'
        with open(feature_list_path, 'w', encoding='utf-8') as f:
            json.dump({'features': selected_features}, f, indent=2)
        print(f"  ✓ Saved feature list to {feature_list_path}")
        
        # Step 7: Evaluation (basic metrics)
        print("\n" + "="*80)
        print("STEP 7: EVALUATION")
        print("="*80)
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, brier_score_loss
        )
        
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'pr_auc': float(average_precision_score(y_test, y_pred_proba)),
            'brier_score': float(brier_score_loss(y_test, y_pred_proba))
        }
        
        print("\n  Model Metrics:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
        
        # Save metrics
        metrics_path = paths['reports'] / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  ✓ Metrics saved to {metrics_path}")
        
        # Step 8: Ablation study
        print("\n" + "="*80)
        print("STEP 8: ABLATION STUDY")
        print("="*80)
        from src.models.evaluator import run_ablation_study
        try:
            ablation_results = run_ablation_study(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                selected_features=selected_features,
                random_state=RANDOM_SEED
            )
            print("\n  ✓ Ablation study completed")
        except Exception as e:
            print(f"  ⚠️  Ablation study failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Save run manifest
        save_run_manifest(paths, validation, selection_report)
        
        print("\n" + "="*80)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Output files:")
        print(f"  Models: {paths['models']}")
        print(f"  Reports: {paths['reports']}")
        print(f"  Inference: {paths['inference']}")
        
    except Exception as e:
        print(f"\n❌ Error during training pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
