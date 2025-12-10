"""
SHAP explainability utilities.
TreeExplainer on dense numeric arrays from preprocessor.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def generate_shap_explanations():
    """Generate SHAP explanations for the current model."""
    if not SHAP_AVAILABLE:
        print("="*80)
        print("SHAP EXPLAINABILITY")
        print("="*80)
        print("\n⚠️  SHAP library not installed.")
        print("  Install with: pip install shap")
        print("  Or run: make install")
        return
    
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / 'models' / 'final_calibrated.joblib'
    preprocessor_path = project_root / 'inference' / 'preprocess.joblib'
    feature_list_path = project_root / 'inference' / 'feature_list.json'
    reports_dir = project_root / 'reports'
    
    if not model_path.exists():
        print("⚠️  Model not found. Run training first.")
        return
    
    print("="*80)
    print("SHAP EXPLAINABILITY")
    print("="*80)
    
    # Load model and preprocessor
    pipeline = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    with open(feature_list_path, 'r') as f:
        import json
        feature_data = json.load(f)
        feature_names = feature_data['features']
    
    # Load test data
    features_path = project_root / 'data' / 'features' / 'features.parquet'
    labels_path = project_root / 'data' / 'labels' / 'labels.parquet'
    
    df_features = pd.read_parquet(features_path)
    df_labels = pd.read_parquet(labels_path)
    df_merged = df_features.merge(df_labels[['siren', 'is_good_opportunity']], on='siren', how='inner')
    
    # Prepare test set
    X_test = df_merged[feature_names].iloc[:100]  # Sample for SHAP
    
    # Transform to dense array
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()
    
    # Get actual feature names after transformation
    # The preprocessor may create more features (e.g., one-hot encoding)
    try:
        # Try to get feature names from preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            transformed_feature_names = preprocessor.get_feature_names_out()
        elif hasattr(preprocessor, 'get_feature_names'):
            transformed_feature_names = preprocessor.get_feature_names()
        else:
            # Fallback: use generic names
            n_features = X_test_transformed.shape[1]
            transformed_feature_names = [f'feature_{i}' for i in range(n_features)]
    except:
        # Fallback: use generic names
        n_features = X_test_transformed.shape[1]
        transformed_feature_names = [f'feature_{i}' for i in range(n_features)]
    
    print(f"  Transformed features: {X_test_transformed.shape[1]} features")
    
    # Get XGBoost model (before calibration)
    # CalibratedClassifierCV stores estimators in calibrated_classifiers_
    calibrated_classifier = pipeline.named_steps['classifier']
    
    # Access base estimator from calibrated classifier
    if hasattr(calibrated_classifier, 'calibrated_classifiers_') and len(calibrated_classifier.calibrated_classifiers_) > 0:
        # Each _CalibratedClassifier has an 'estimator' attribute
        xgb_model = calibrated_classifier.calibrated_classifiers_[0].estimator
    elif hasattr(calibrated_classifier, 'estimator'):
        xgb_model = calibrated_classifier.estimator
    else:
        raise ValueError("Could not extract XGBoost model from calibrated classifier")
    
    # SHAP TreeExplainer
    print("\n📊 Computing SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_transformed)
    
    # Handle case where shap_values might be a list (for binary classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class SHAP values
    
    # Get actual number of features from SHAP values
    if len(shap_values.shape) == 2:
        n_shap_features = shap_values.shape[1]
    else:
        n_shap_features = len(shap_values) if len(shap_values.shape) == 1 else shap_values.shape[0]
    
    # Ensure feature names match the number of features
    if len(transformed_feature_names) != n_shap_features:
        print(f"  ⚠️  Feature name count mismatch: {len(transformed_feature_names)} names vs {n_shap_features} SHAP features")
        if len(transformed_feature_names) > n_shap_features:
            transformed_feature_names = transformed_feature_names[:n_shap_features]
        else:
            # Add generic names for missing features
            transformed_feature_names = list(transformed_feature_names) + [f'feature_{i}' for i in range(len(transformed_feature_names), n_shap_features)]
    
    print(f"  Using {n_shap_features} features for SHAP plots")
    
    # Global summary
    if not MATPLOTLIB_AVAILABLE:
        print("  ⚠️  Matplotlib not available, skipping plot generation")
        print("  Install with: pip install matplotlib")
    else:
        print("  Generating global summary plot...")
        max_display = min(20, n_shap_features)
        # Ensure we have the right shape for summary plot
        if len(shap_values.shape) == 1:
            shap_values_2d = shap_values.reshape(-1, 1)
        else:
            shap_values_2d = shap_values
        
        shap.summary_plot(shap_values_2d, X_test_transformed, 
                         feature_names=transformed_feature_names, 
                         show=False, max_display=max_display)
        plt.savefig(reports_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved global summary to {reports_dir / 'shap_summary.png'}")
    
    # Local explanations (10 examples)
    if MATPLOTLIB_AVAILABLE:
        examples_dir = reports_dir / 'shap_examples'
        examples_dir.mkdir(parents=True, exist_ok=True)
        
        print("  Generating local explanations...")
        for i in range(min(10, len(X_test))):
            # Ensure we have the right shape
            if len(shap_values.shape) == 2:
                values = shap_values[i]
            else:
                values = shap_values
            
            n_features_used = min(15, n_shap_features)
            try:
                shap.waterfall_plot(shap.Explanation(values=values[:n_features_used], 
                                                    base_values=explainer.expected_value,
                                                    data=X_test_transformed[i][:n_features_used],
                                                    feature_names=transformed_feature_names[:n_features_used]),
                                   show=False, max_display=n_features_used)
                plt.savefig(examples_dir / f'example_{i+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  ⚠️  Could not generate example {i+1}: {e}")
                plt.close()
        
        print(f"  ✓ Saved 10 local explanations to {examples_dir}")
    else:
        print("  ⚠️  Matplotlib not available, skipping local explanation plots")
    
    # SHAP Force plots (PNG visualizations)
    if MATPLOTLIB_AVAILABLE:
        print("  Generating SHAP force plots...")
        force_plots_dir = reports_dir / 'shap_force_plots'
        force_plots_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(min(10, len(X_test))):
            try:
                # Ensure we have the right shape for SHAP values
                if len(shap_values.shape) == 2:
                    values = shap_values[i]
                else:
                    values = shap_values
                
                # Generate force plot as PNG using matplotlib
                shap.force_plot(
                    explainer.expected_value,
                    values,
                    X_test_transformed[i],
                    feature_names=transformed_feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.savefig(force_plots_dir / f'force_plot_{i+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"  ⚠️  Could not generate force plot {i+1}: {e}")
                plt.close()
        
        print(f"  ✓ Saved 10 force plots to {force_plots_dir}")
    else:
        print("  ⚠️  Matplotlib not available, skipping force plots")
    
    print("\n✓ SHAP explanations completed")


if __name__ == "__main__":
    generate_shap_explanations()
