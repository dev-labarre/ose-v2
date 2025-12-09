"""
Signal processing utilities.
Uses stable parser, filters pre-t0 only.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast
from typing import Dict

T0 = pd.Timestamp('2023-01-01')


def prepare_signals(df_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw signals table (coworker's stable parser).
    
    Args:
        df_signals: DataFrame with signals data
        
    Returns:
        DataFrame with cleaned signals
    """
    df = df_signals.copy()
    
    # Convert the 'type' JSON-like string into a dictionary
    if 'type' in df.columns:
        df['type'] = df['type'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, dict) else {})
        )
        
        df['signal_code'] = df['type'].apply(lambda x: x.get('code') if isinstance(x, dict) else None)
        df['signal_label'] = df['type'].apply(lambda x: x.get('label') if isinstance(x, dict) else None)
    else:
        df['signal_code'] = None
        df['signal_label'] = None
    
    # Parse date
    if 'publishedAt' in df.columns:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], utc=True, errors='coerce')
    else:
        df['publishedAt'] = pd.NaT
    
    return df[['siren', 'signal_code', 'signal_label', 'publishedAt']].copy()


def filter_pre_t0_signals(df_signals: pd.DataFrame) -> pd.DataFrame:
    """Filter signals to pre-t0 only."""
    if 'publishedAt' in df_signals.columns:
        return df_signals[df_signals['publishedAt'] < T0].copy()
    return df_signals.copy()


def build_signal_features(df_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Build signal features from pre-t0 signals.
    
    Args:
        df_signals: DataFrame with cleaned signals (pre-t0 filtered)
        
    Returns:
        DataFrame with signal features per SIREN
    """
    if len(df_signals) == 0:
        return pd.DataFrame()
    
    # Group by SIREN and compute features
    signal_features = df_signals.groupby('siren').agg({
        'signal_code': 'count',
        'publishedAt': ['min', 'max', 'count']
    }).reset_index()
    
    signal_features.columns = ['siren', 'n_signals', 'signal_first_date', 'signal_last_date', 'signal_count']
    
    # Signal code counts
    if 'signal_code' in df_signals.columns:
        code_counts = df_signals.groupby(['siren', 'signal_code']).size().unstack(fill_value=0)
        code_counts.columns = [f'n_code_{col}' for col in code_counts.columns]
        signal_features = signal_features.merge(code_counts, left_on='siren', right_index=True, how='left')
    
    return signal_features
