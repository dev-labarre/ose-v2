"""
Data quality module.
Extends DataQualityTransformer with strict requirements:
- Drop columns with >50% missing
- Add _was_nan flags for all columns with missing values
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
from sklearn.base import BaseEstimator, TransformerMixin


class DataQualityTransformer(BaseEstimator, TransformerMixin):
    """
    Data quality transformer.
    Removes duplicates, computes missing rates, adds _was_nan flags,
    and drops columns with missing_rate > 0.50 (strict enforcement).
    """
    
    def __init__(self, remove_duplicates: bool = True, duplicate_keep: str = 'last',
                 drop_high_missing: bool = True, missing_threshold: float = 0.50):
        """
        Args:
            remove_duplicates: Whether to remove duplicate rows
            duplicate_keep: Which duplicate to keep ('first', 'last')
            drop_high_missing: Whether to drop columns with >threshold missing
            missing_threshold: Threshold for dropping columns (default 0.50 = 50%)
        """
        self.remove_duplicates = remove_duplicates
        self.duplicate_keep = duplicate_keep
        self.drop_high_missing = drop_high_missing
        self.missing_threshold = missing_threshold
        self.quality_stats_ = {}
        self.original_shape_ = None
        self.transformed_shape_ = None
        self.dropped_columns_ = []
        self.missing_flag_columns_ = []
    
    def fit(self, X, y=None):
        """Fit the transformer."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("DataQualityTransformer requires pandas DataFrame")
        
        self.original_shape_ = X.shape
        self.quality_stats_ = self._compute_quality_stats(X)
        
        return self
    
    def transform(self, X):
        """Transform X by applying data quality improvements."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("DataQualityTransformer requires pandas DataFrame")
        
        X_cleaned = X.copy()
        
        # Remove metadata leakage columns (processedAt, updatedAt, last_modified)
        metadata_cols = ['processedAt', 'updatedAt', 'last_modified']
        metadata_cols_present = [col for col in metadata_cols if col in X_cleaned.columns]
        if metadata_cols_present:
            X_cleaned = X_cleaned.drop(columns=metadata_cols_present, errors='ignore')
            if 'metadata_columns_dropped' not in self.quality_stats_:
                self.quality_stats_['metadata_columns_dropped'] = []
            self.quality_stats_['metadata_columns_dropped'].extend(metadata_cols_present)
            print(f"  ✓ Dropped {len(metadata_cols_present)} metadata columns: {metadata_cols_present}")
        
        # Remove noisy categorical identifiers (company_name*, raison_sociale, siret)
        # Keep 'siren' for merging purposes
        noisy_categorical_cols = []
        for col in X_cleaned.columns:
            if col == 'siren':
                continue  # Keep siren for merging
            if 'company_name' in col.lower():
                noisy_categorical_cols.append(col)
            elif col in ['raison_sociale', 'siret']:
                noisy_categorical_cols.append(col)
            elif col.startswith('raison_sociale_'):
                noisy_categorical_cols.append(col)
        
        if noisy_categorical_cols:
            X_cleaned = X_cleaned.drop(columns=noisy_categorical_cols, errors='ignore')
            if 'noisy_categorical_columns_dropped' not in self.quality_stats_:
                self.quality_stats_['noisy_categorical_columns_dropped'] = []
            self.quality_stats_['noisy_categorical_columns_dropped'].extend(noisy_categorical_cols)
            print(f"  ✓ Dropped {len(noisy_categorical_cols)} noisy categorical columns: {noisy_categorical_cols[:5]}{'...' if len(noisy_categorical_cols) > 5 else ''}")
        
        # Remove duplicates
        if self.remove_duplicates:
            hashable_cols = self._get_hashable_columns(X_cleaned)
            if hashable_cols:
                try:
                    n_duplicates = X_cleaned[hashable_cols].duplicated().sum()
                    if n_duplicates > 0:
                        X_cleaned = X_cleaned.drop_duplicates(subset=hashable_cols, keep=self.duplicate_keep)
                        self.quality_stats_['duplicates_removed'] = n_duplicates
                except:
                    self.quality_stats_['duplicates_removed'] = 0
        
        # Define critical numeric features to preserve even if missingness > threshold
        critical_numeric_features = ['effectif', 'nbFilialesDirectes', 'nbMarques', 'contact_count']
        # Also preserve variants like effectifEstime, effectifConsolide, etc.
        critical_feature_patterns = ['effectif', 'nbFiliales', 'nbMarques', 'contact_count']
        preserved_critical_features = []
        
        # Compute missing rates and drop high-missing columns
        if self.drop_high_missing:
            cols_to_drop = []
            missing_rates = {}
            
            for col in X_cleaned.columns:
                n_missing = X_cleaned[col].isna().sum()
                pct_missing = (n_missing / len(X_cleaned)) * 100 if len(X_cleaned) > 0 else 0
                missing_rates[col] = pct_missing
                
                # Check if this is a critical feature to preserve
                is_critical = False
                if col in critical_numeric_features:
                    is_critical = True
                else:
                    # Check if column name matches critical patterns
                    for pattern in critical_feature_patterns:
                        if pattern.lower() in col.lower():
                            is_critical = True
                            break
                
                if pct_missing > (self.missing_threshold * 100) and not is_critical:
                    cols_to_drop.append(col)
                elif is_critical:
                    # Preserve ALL critical features regardless of missing rate
                    preserved_critical_features.append(col)
            
            if cols_to_drop:
                X_cleaned = X_cleaned.drop(columns=cols_to_drop)
                self.dropped_columns_ = cols_to_drop
                self.quality_stats_['columns_dropped_high_missing'] = len(cols_to_drop)
                self.quality_stats_['dropped_column_names'] = cols_to_drop
                self.quality_stats_['missing_rates'] = missing_rates
            else:
                self.quality_stats_['columns_dropped_high_missing'] = 0
                self.dropped_columns_ = []
            
            # Add _was_nan flags for preserved critical features BEFORE imputation
            if preserved_critical_features:
                for col in preserved_critical_features:
                    if col in X_cleaned.columns and X_cleaned[col].isna().any():
                        flag_col = f"{col}_was_nan"
                        X_cleaned[flag_col] = X_cleaned[col].isna().astype(int)
                        self.missing_flag_columns_.append(flag_col)
                
                # Apply median imputation to preserved critical features
                for col in preserved_critical_features:
                    if col in X_cleaned.columns and pd.api.types.is_numeric_dtype(X_cleaned[col]):
                        median_val = X_cleaned[col].median()
                        if pd.notna(median_val):
                            X_cleaned[col] = X_cleaned[col].fillna(median_val)
                        else:
                            # If all values are NaN, fill with 0
                            X_cleaned[col] = X_cleaned[col].fillna(0)
                self.quality_stats_['critical_features_preserved'] = preserved_critical_features
                print(f"  ✓ Preserved {len(preserved_critical_features)} critical numeric features with median imputation: {preserved_critical_features[:3]}{'...' if len(preserved_critical_features) > 3 else ''}")
        
        # Add _was_nan flags for all other columns with missing values
        for col in X_cleaned.columns:
            if col not in [f"{pc}_was_nan" for pc in preserved_critical_features] and col not in preserved_critical_features:
                if X_cleaned[col].isna().any():
                    flag_col = f"{col}_was_nan"
                    X_cleaned[flag_col] = X_cleaned[col].isna().astype(int)
                    if flag_col not in self.missing_flag_columns_:
                        self.missing_flag_columns_.append(flag_col)
        
        self.transformed_shape_ = X_cleaned.shape
        self.quality_stats_['missing_flags_created'] = len(self.missing_flag_columns_)
        
        return X_cleaned
    
    def _get_hashable_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that contain hashable types."""
        hashable_cols = []
        for col in df.columns:
            try:
                non_null = df[col].dropna()
                if non_null.empty:
                    hashable_cols.append(col)
                    continue
                
                sample_val = non_null.iloc[0]
                if sample_val is not None and isinstance(sample_val, (list, dict, set)):
                    continue
                hashable_cols.append(col)
            except:
                hashable_cols.append(col)
        return hashable_cols
    
    def _compute_quality_stats(self, df: pd.DataFrame) -> Dict:
        """Compute data quality statistics."""
        hashable_cols = self._get_hashable_columns(df)
        try:
            if hashable_cols:
                n_duplicates = df[hashable_cols].duplicated().sum()
            else:
                n_duplicates = 0
        except:
            n_duplicates = 0
        
        stats = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'n_duplicates': n_duplicates,
            'missing_values': {},
            'missing_percentage': {},
            'dtypes': df.dtypes.to_dict()
        }
        
        for col in df.columns:
            n_missing = df[col].isna().sum()
            pct_missing = (n_missing / len(df)) * 100 if len(df) > 0 else 0
            stats['missing_values'][col] = n_missing
            stats['missing_percentage'][col] = pct_missing
        
        return stats
    
    def get_quality_report(self) -> Dict:
        """Get data quality report."""
        report = {
            'original_shape': self.original_shape_,
            'transformed_shape': self.transformed_shape_,
            'statistics': self.quality_stats_,
            'dropped_columns': self.dropped_columns_,
            'missing_flag_columns': self.missing_flag_columns_
        }
        
        if self.transformed_shape_ and self.original_shape_:
            report['rows_removed'] = self.original_shape_[0] - self.transformed_shape_[0]
            report['columns_removed'] = self.original_shape_[1] - self.transformed_shape_[1]
            report['columns_added'] = len(self.missing_flag_columns_)
        
        return report


def generate_missing_summary(df_before: pd.DataFrame, df_after: pd.DataFrame,
                            transformer: DataQualityTransformer,
                            output_path: Path):
    """Generate missing summary report."""
    summary = {
        'before': {
            'n_rows': len(df_before),
            'n_cols': len(df_before.columns),
            'missing_by_column': {}
        },
        'after': {
            'n_rows': len(df_after),
            'n_cols': len(df_after.columns),
            'missing_by_column': {}
        },
        'dropped_columns': transformer.dropped_columns_,
        'missing_flags_created': transformer.missing_flag_columns_
    }
    
    for col in df_before.columns:
        n_missing = df_before[col].isna().sum()
        pct_missing = (n_missing / len(df_before)) * 100 if len(df_before) > 0 else 0
        summary['before']['missing_by_column'][col] = {
            'n_missing': int(n_missing),
            'pct_missing': float(pct_missing)
        }
    
    for col in df_after.columns:
        if not col.endswith('_was_nan'):
            n_missing = df_after[col].isna().sum()
            pct_missing = (n_missing / len(df_after)) * 100 if len(df_after) > 0 else 0
            summary['after']['missing_by_column'][col] = {
                'n_missing': int(n_missing),
                'pct_missing': float(pct_missing)
            }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"  ✓ Missing summary saved to {output_path}")


def generate_kept_features_report(df: pd.DataFrame, output_path: Path):
    """Generate report of kept features."""
    report = {
        'n_features': len(df.columns),
        'feature_names': list(df.columns),
        'feature_types': df.dtypes.astype(str).to_dict(),
        'feature_categories': {
            'numeric': list(df.select_dtypes(include=[np.number]).columns),
            'categorical': list(df.select_dtypes(include=['object', 'bool']).columns),
            'datetime': list(df.select_dtypes(include=['datetime']).columns),
            'missing_flags': [col for col in df.columns if col.endswith('_was_nan')]
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"  ✓ Kept features report saved to {output_path}")
