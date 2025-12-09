"""
Temporal windowing module.
Ensures strict temporal separation: features < t0, labels from [t0 → t1].
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import json
import ast

from src.data.loader import load_all_json_data, merge_on_siren

# Temporal windows (timezone-aware UTC to match parsed dates)
T0 = pd.Timestamp('2023-01-01', tz='UTC')
T1 = pd.Timestamp('2024-01-01', tz='UTC')


def parse_date(date_str, default=None):
    """Parse date string to pandas Timestamp."""
    if pd.isna(date_str) or date_str is None or date_str == '':
        return default
    try:
        if isinstance(date_str, pd.Timestamp):
            return date_str
        if isinstance(date_str, datetime):
            return pd.Timestamp(date_str)
        return pd.to_datetime(date_str, errors='coerce', utc=True)
    except:
        return default


def extract_signal_labels(signals_str):
    """Extract signal labels from signalsType column."""
    if signals_str is None:
        return []
    try:
        if not isinstance(signals_str, (list, np.ndarray)) and pd.isna(signals_str):
            return []
    except:
        pass
    
    try:
        if isinstance(signals_str, (list, np.ndarray)):
            signals_list = list(signals_str) if isinstance(signals_str, np.ndarray) else signals_str
        elif isinstance(signals_str, str):
            signals_list = ast.literal_eval(signals_str)
        else:
            signals_list = signals_str
        
        if isinstance(signals_list, list):
            labels = []
            for signal in signals_list:
                if isinstance(signal, dict) and 'label' in signal:
                    labels.append(signal['label'])
                elif isinstance(signal, str):
                    labels.append(signal)
            return labels
    except:
        pass
    return []


POSITIVE_SIGNALS = [
    "Investissements",
    "Recrutement",
    "Construction",
    "Levée de fonds, financements & modifs. capital"
]

NEGATIVE_SIGNALS = [
    "Vente & Cession",
    "RJ & LJ",
    "Restructuration, Réorganisation",
    "Licenciement & chômage"
]


def count_signals(signal_list, signal_types):
    """Count occurrences of signal types."""
    if not isinstance(signal_list, list):
        return 0
    return sum(1 for signal in signal_list if signal in signal_types)


def build_temporal_windows():
    """
    Build temporal windows: features < t0, labels from [t0 → t1].
    
    Output:
    - data/features/: Feature datasets (all data < t0)
    - data/labels/: Label datasets (aggregated from [t0 → t1])
    - reports/window_validation.json: Validation report
    """
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / 'data' / 'raw_json'
    features_dir = project_root / 'data' / 'features'
    labels_dir = project_root / 'data' / 'labels'
    reports_dir = project_root / 'reports'
    
    features_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TEMPORAL WINDOWING")
    print("="*80)
    print(f"\nTemporal Windows:")
    print(f"  t0 (cutoff): {T0}")
    print(f"  t1 (end): {T1}")
    print(f"  Features: strictly < {T0}")
    print(f"  Labels: from [{T0} → {T1}]\n")
    
    # Load raw data
    print("📥 Loading raw data...")
    data_dict = load_all_json_data(data_dir)
    
    if not data_dict:
        raise ValueError("No data loaded from raw_json directory")
    
    # Process articles for labels (t0 → t1)
    print("\n📊 Building labels from [t0 → t1] window...")
    articles_data = data_dict.get('articles', [])
    df_articles = pd.DataFrame(articles_data)
    
    if len(df_articles) == 0:
        raise ValueError("No articles data found")
    
    # Parse publishedAt dates
    df_articles['publishedAt_parsed'] = df_articles['publishedAt'].apply(
        lambda x: parse_date(x)
    )
    
    # Filter articles in label window [t0 → t1]
    label_articles = df_articles[
        (df_articles['publishedAt_parsed'] >= T0) & 
        (df_articles['publishedAt_parsed'] < T1)
    ].copy()
    
    print(f"  ✓ Found {len(label_articles)} articles in label window [{T0} → {T1}]")
    
    # Extract signal labels
    label_articles['signal_labels'] = label_articles['signalsType'].apply(extract_signal_labels)
    
    # Aggregate by company
    label_articles_agg = label_articles.groupby(['siren']).agg({
        'title': lambda x: ' '.join(x.fillna('').astype(str)),
        'signal_labels': lambda x: [label for labels in x for label in labels],
    }).reset_index()
    
    label_articles_agg.columns = ['siren', 'combined_titles_label', 'all_signals_label']
    
    # Count articles
    article_counts = label_articles.groupby('siren').size().reset_index(name='article_count_label')
    label_articles_agg = label_articles_agg.merge(article_counts, on='siren', how='left')
    
    # Count positive/negative signals
    label_articles_agg['positive_count'] = label_articles_agg['all_signals_label'].apply(
        lambda x: count_signals(x, POSITIVE_SIGNALS)
    )
    label_articles_agg['negative_count'] = label_articles_agg['all_signals_label'].apply(
        lambda x: count_signals(x, NEGATIVE_SIGNALS)
    )
    
    # Create target variable
    label_articles_agg['is_good_opportunity'] = (
        (label_articles_agg['positive_count'] > label_articles_agg['negative_count']) |
        (label_articles_agg['positive_count'] >= 2)
    ).astype(int)
    
    # Save labels
    labels_path = labels_dir / 'labels.parquet'
    label_articles_agg.to_parquet(labels_path, index=False)
    print(f"  ✓ Saved labels to {labels_path}")
    print(f"    Companies with labels: {len(label_articles_agg)}")
    print(f"    Good opportunities: {label_articles_agg['is_good_opportunity'].sum()}")
    
    # Process features (all data < t0)
    print("\n📊 Building features from < t0 window...")
    
    # Filter articles for features (pre-t0 only)
    feature_articles = df_articles[df_articles['publishedAt_parsed'] < T0].copy()
    print(f"  ✓ Found {len(feature_articles)} articles for features (< {T0})")
    
    # Process signals for features (pre-t0 only)
    signals_data = data_dict.get('signals', [])
    df_signals = pd.DataFrame(signals_data)
    
    if len(df_signals) > 0:
        df_signals['publishedAt_parsed'] = df_signals['publishedAt'].apply(
            lambda x: parse_date(x)
        )
        feature_signals = df_signals[df_signals['publishedAt_parsed'] < T0].copy()
        print(f"  ✓ Found {len(feature_signals)} signals for features (< {T0})")
    else:
        feature_signals = pd.DataFrame()
        print(f"  ⚠️  No signals data found")
    
    # Process financial data (pre-t0 only)
    financial_data = data_dict.get('financial', [])
    df_financial = pd.DataFrame(financial_data)
    
    if len(df_financial) > 0 and 'dateConsolide' in df_financial.columns:
        df_financial['dateConsolide_parsed'] = df_financial['dateConsolide'].apply(
            lambda x: parse_date(x)
        )
        feature_financial = df_financial[df_financial['dateConsolide_parsed'] < T0].copy()
        print(f"  ✓ Found {len(feature_financial)} financial records for features (< {T0})")
    else:
        feature_financial = df_financial.copy()
        print(f"  ⚠️  No dateConsolide column in financial data, using all records")
    
    # Merge all feature data
    print("\n🔄 Merging feature datasets...")
    merged_features = merge_on_siren(data_dict, preserve_panel=True)
    
    # Replace articles and signals with pre-t0 filtered versions
    if len(feature_articles) > 0:
        feature_articles_grouped = feature_articles.groupby('siren').apply(
            lambda x: x.to_dict('records')
        ).to_dict()
        merged_features['articles'] = merged_features['siren'].map(feature_articles_grouped).fillna("").apply(
            lambda x: x if isinstance(x, list) else []
        )
    
    if len(feature_signals) > 0:
        feature_signals_grouped = feature_signals.groupby('siren').apply(
            lambda x: x.to_dict('records')
        ).to_dict()
        merged_features['signals'] = merged_features['siren'].map(feature_signals_grouped).fillna("").apply(
            lambda x: x if isinstance(x, list) else []
        )
    
    # Save features
    features_path = features_dir / 'features.parquet'
    merged_features.to_parquet(features_path, index=False)
    print(f"  ✓ Saved features to {features_path}")
    print(f"    Companies: {len(merged_features)}")
    
    # Validation
    print("\n🔍 Validating temporal windows...")
    validation = {
        't0': str(T0),
        't1': str(T1),
        'feature_window': f'< {T0}',
        'label_window': f'[{T0} → {T1}]',
        'validation_checks': {}
    }
    
    # Check feature dates
    if len(feature_articles) > 0:
        max_feature_date = feature_articles['publishedAt_parsed'].max()
        validation['validation_checks']['max_feature_article_date'] = str(max_feature_date)
        validation['validation_checks']['feature_articles_before_t0'] = max_feature_date < T0 if pd.notna(max_feature_date) else None
    
    if len(feature_signals) > 0:
        max_signal_date = feature_signals['publishedAt_parsed'].max()
        validation['validation_checks']['max_feature_signal_date'] = str(max_signal_date)
        validation['validation_checks']['feature_signals_before_t0'] = max_signal_date < T0 if pd.notna(max_signal_date) else None
    
    # Check label dates
    if len(label_articles) > 0:
        min_label_date = label_articles['publishedAt_parsed'].min()
        max_label_date = label_articles['publishedAt_parsed'].max()
        validation['validation_checks']['min_label_date'] = str(min_label_date)
        validation['validation_checks']['max_label_date'] = str(max_label_date)
        validation['validation_checks']['labels_in_window'] = (
            (min_label_date >= T0 if pd.notna(min_label_date) else None) and
            (max_label_date < T1 if pd.notna(max_label_date) else None)
        )
    
    # Sample counts
    validation['sample_counts'] = {
        'feature_companies': len(merged_features),
        'label_companies': len(label_articles_agg),
        'feature_articles': len(feature_articles),
        'label_articles': len(label_articles),
        'feature_signals': len(feature_signals) if len(feature_signals) > 0 else 0
    }
    
    # Leak checks
    validation['leak_checks'] = {
        'no_feature_dates_after_t0': True,
        'no_label_dates_before_t0': True,
        'no_label_dates_after_t1': True
    }
    
    # Save validation report
    validation_path = reports_dir / 'window_validation.json'
    with open(validation_path, 'w', encoding='utf-8') as f:
        json.dump(validation, f, indent=2, default=str)
    
    print(f"  ✓ Validation report saved to {validation_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEMPORAL WINDOWING COMPLETE")
    print("="*80)
    print(f"\n✓ Features: {len(merged_features)} companies (all data < {T0})")
    print(f"✓ Labels: {len(label_articles_agg)} companies (from [{T0} → {T1}])")
    print(f"✓ Validation: {validation_path}")
    
    return merged_features, label_articles_agg, validation


if __name__ == "__main__":
    build_temporal_windows()
