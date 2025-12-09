"""
Financial ratios processing.
Loads INPI ratios, normalizes, winsorizes, and generates 3-year summaries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import json
from datetime import datetime

T0 = pd.Timestamp('2023-01-01')


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize a series at specified percentiles.
    
    Args:
        series: Series to winsorize
        lower: Lower percentile (default 0.01)
        upper: Upper percentile (default 0.99)
        
    Returns:
        Winsorized series
    """
    if series.isna().all():
        return series
    
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    
    return series.clip(lower=lower_bound, upper=upper_bound)


def normalize_by_size(series: pd.Series, size_series: pd.Series) -> pd.Series:
    """
    Normalize a series by company size (revenue).
    
    Args:
        series: Series to normalize
        size_series: Size series (e.g., revenue)
        
    Returns:
        Normalized series
    """
    # Avoid division by zero
    size_series = size_series.replace(0, np.nan)
    normalized = series / size_series
    return normalized


def compute_3y_summaries(df_ratios: pd.DataFrame, siren: str, 
                         date_col: str = 'date_cloture_exercice',
                         value_col: str = 'ca') -> Dict:
    """
    Compute 3-year rolling summaries for a company.
    
    Args:
        df_ratios: DataFrame with ratios
        siren: SIREN identifier
        date_col: Date column name
        value_col: Value column name
        
    Returns:
        Dictionary with last, mean, std, slope
    """
    company_ratios = df_ratios[df_ratios['siren'] == siren].copy()
    
    if len(company_ratios) == 0:
        return {
            'last': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'slope': np.nan
        }
    
    # Sort by date
    company_ratios = company_ratios.sort_values(date_col)
    
    # Get last 3 years (before t0)
    company_ratios = company_ratios[company_ratios[date_col] < T0]
    
    if len(company_ratios) == 0:
        return {
            'last': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'slope': np.nan
        }
    
    # Get last 3 years
    last_3y = company_ratios.tail(3)
    
    if len(last_3y) == 0:
        return {
            'last': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'slope': np.nan
        }
    
    values = last_3y[value_col].dropna()
    
    if len(values) == 0:
        return {
            'last': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'slope': np.nan
        }
    
    last = values.iloc[-1] if len(values) > 0 else np.nan
    mean = values.mean()
    std = values.std() if len(values) > 1 else 0.0
    
    # Compute slope (linear trend)
    if len(values) >= 2:
        x = np.arange(len(values))
        slope = np.polyfit(x, values.values, 1)[0] if len(values) >= 2 else 0.0
    else:
        slope = 0.0
    
    return {
        'last': float(last) if pd.notna(last) else np.nan,
        'mean': float(mean) if pd.notna(mean) else np.nan,
        'std': float(std) if pd.notna(std) else np.nan,
        'slope': float(slope) if pd.notna(slope) else np.nan
    }


def process_inpi_ratios(df_features: pd.DataFrame, 
                       ratios_path: Path,
                       output_path: Path) -> pd.DataFrame:
    """
    Process INPI ratios and join to features.
    
    Args:
        df_features: Features DataFrame
        ratios_path: Path to INPI ratios parquet file
        output_path: Path to save ratio summary report
        
    Returns:
        Features DataFrame with ratio features added
    """
    print("="*80)
    print("PROCESSING INPI RATIOS")
    print("="*80)
    
    if not ratios_path.exists():
        print(f"⚠️  Ratios file not found: {ratios_path}")
        print("  Run 'make fetch_ratios' first")
        return df_features
    
    print(f"\n📥 Loading ratios from: {ratios_path}")
    df_ratios = pd.read_parquet(ratios_path)
    
    if len(df_ratios) == 0:
        print("  ⚠️  No ratios data loaded")
        return df_features
    
    print(f"  ✓ Loaded {len(df_ratios)} ratio records")
    print(f"  Unique SIRENs: {df_ratios['siren'].nunique()}")
    
    # Filter ratios before t0
    if 'date_cloture_exercice' in df_ratios.columns:
        df_ratios['date_cloture_exercice'] = pd.to_datetime(
            df_ratios['date_cloture_exercice'], errors='coerce'
        )
        df_ratios = df_ratios[df_ratios['date_cloture_exercice'] < T0]
        print(f"  ✓ Filtered to {len(df_ratios)} ratios before t0 ({T0})")
    
    # Winsorize numeric columns at 1%/99%
    print("\n📊 Winsorizing ratios at 1%/99%...")
    numeric_cols = df_ratios.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['siren']:
            df_ratios[col] = winsorize(df_ratios[col], lower=0.01, upper=0.99)
    
    # Generate 3-year summaries for key ratios
    print("\n📊 Generating 3-year summaries...")
    ratio_cols = ['ca', 'resultat_net', 'fonds_propres', 'dettes_financieres', 
                  'excedent_brut_exploitation', 'resultat_exploitation']
    
    siren_list = df_features['siren'].unique()
    ratio_features = {}
    
    for siren in siren_list:
        ratio_features[siren] = {}
        for col in ratio_cols:
            if col in df_ratios.columns:
                summaries = compute_3y_summaries(df_ratios, siren, value_col=col)
                for stat, value in summaries.items():
                    ratio_features[siren][f'{col}_{stat}'] = value
    
    # Convert to DataFrame and merge
    ratio_features_df = pd.DataFrame.from_dict(ratio_features, orient='index')
    ratio_features_df = ratio_features_df.reset_index()
    ratio_features_df.columns = ['siren'] + list(ratio_features_df.columns[1:])
    
    # Add _was_nan flags
    for col in ratio_features_df.columns:
        if col != 'siren':
            if ratio_features_df[col].isna().any():
                ratio_features_df[f'{col}_was_nan'] = ratio_features_df[col].isna().astype(int)
    
    # Merge with features
    print("\n🔄 Merging ratio features with main features...")
    df_features = df_features.merge(ratio_features_df, on='siren', how='left')
    
    print(f"  ✓ Added {len(ratio_features_df.columns) - 1} ratio features")
    print(f"  Final features shape: {df_features.shape}")
    
    # Generate summary report
    summary = {
        'n_ratios_loaded': len(df_ratios),
        'n_sirens_with_ratios': df_ratios['siren'].nunique(),
        'n_ratio_features_added': len(ratio_features_df.columns) - 1,
        'ratio_features': list(ratio_features_df.columns),
        'date_range': {
            'min': str(df_ratios['date_cloture_exercice'].min()) if 'date_cloture_exercice' in df_ratios.columns else None,
            'max': str(df_ratios['date_cloture_exercice'].max()) if 'date_cloture_exercice' in df_ratios.columns else None
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"  ✓ Ratio summary saved to {output_path}")
    
    return df_features


def main():
    """Main function."""
    project_root = Path(__file__).resolve().parents[2]
    features_path = project_root / 'data' / 'features' / 'features.parquet'
    ratios_path = project_root / 'data' / 'external' / 'inpi_ratios.parquet'
    output_path = project_root / 'reports' / 'ratio_summary.json'
    
    if not features_path.exists():
        print(f"⚠️  Features file not found: {features_path}")
        return
    
    df_features = pd.read_parquet(features_path)
    df_features = process_inpi_ratios(df_features, ratios_path, output_path)
    
    # Save updated features
    df_features.to_parquet(features_path, index=False)
    print(f"\n✓ Updated features saved to {features_path}")


if __name__ == "__main__":
    main()
