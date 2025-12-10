"""Service for feature engineering from extracted data."""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.loader import merge_on_siren
from src.data.windowing import T0, T1
from src.features.financials import aggregate_financial_and_signal_scores
from src.features.text import process_text_features
from src.features.signals import build_signal_features
from inference.predict import load_inference_artifacts


class FeatureEngineer:
    """Engineer features from extracted data dictionary."""
    
    def __init__(self):
        self._feature_names = None
        self._load_feature_names()
    
    def _load_feature_names(self):
        """Load required feature names."""
        if self._feature_names is None:
            _, _, self._feature_names = load_inference_artifacts()
    
    def process_extracted_data(self, data_dict: Dict[str, List[dict]], 
                              target_siren: Optional[str] = None) -> pd.DataFrame:
        """
        Process extracted data through feature engineering pipeline.
        
        Args:
            data_dict: Dictionary with 9 datasets (from ExtractionService)
            target_siren: Optional SIREN to filter to (if None, processes all)
        
        Returns:
            DataFrame with engineered features
        """
        # Step 1: Merge all datasets on SIREN
        df = merge_on_siren(data_dict, preserve_panel=False)
        
        if len(df) == 0:
            raise ValueError("No data found after merging")
        
        # Filter to target SIREN if specified
        if target_siren:
            df = df[df['siren'] == str(target_siren)]
            if len(df) == 0:
                raise ValueError(f"SIREN {target_siren} not found in merged data")
        
        # Step 2: Prepare articles for temporal filtering
        articles_data = data_dict.get('articles', [])
        df_articles = pd.DataFrame(articles_data) if articles_data else pd.DataFrame()
        
        if len(df_articles) > 0 and 'publishedAt' in df_articles.columns:
            df_articles['publishedAt_parsed'] = (
                pd.to_datetime(df_articles['publishedAt'], errors='coerce', utc=True)
                .dt.tz_localize(None)
            )
            # Filter articles for features (pre-t0 only)
            df_articles_pre_t0 = df_articles[df_articles['publishedAt_parsed'] < T0].copy()
        else:
            df_articles_pre_t0 = pd.DataFrame()
        
        # Step 3: Build signal features
        signals_data = data_dict.get('signals', [])
        if signals_data:
            df_signals = pd.DataFrame(signals_data)
            signal_features = build_signal_features(df_signals)
            df = df.merge(signal_features, on='siren', how='left')
        else:
            # Add empty signal features
            signal_cols = [f'signal_{s}_count' for s in ['B', 'W', 'E', 'F', 'N', 'S', 'K1', 'I', 'M', 'O']]
            for col in signal_cols:
                if col not in df.columns:
                    df[col] = 0
        
        # Step 4: Financial & signal scoring
        # This requires the full feature set
        try:
            # Create a temporary path for report (function requires a Path)
            import tempfile
            with tempfile.NamedTemporaryFile(delete=True, suffix='.json') as tmp:
                tmp_path = Path(tmp.name)
                df = aggregate_financial_and_signal_scores(
                    df,
                    output_path=tmp_path  # Temporary file, will be deleted
                )
        except Exception as e:
            # If scoring fails, continue with basic features
            logger.warning(f"Financial scoring failed: {e}")
        
        # Step 5: Text processing (if articles available)
        if len(df_articles_pre_t0) > 0:
            try:
                # Create a temporary path for report (function requires a Path)
                import tempfile
                with tempfile.NamedTemporaryFile(delete=True, suffix='.json') as tmp:
                    tmp_path = Path(tmp.name)
                    df = process_text_features(
                        df,
                        df_articles_pre_t0,
                        output_path=tmp_path  # Temporary file, will be deleted
                    )
            except Exception as e:
                # If text processing fails, add empty text features
                logger.warning(f"Text processing failed: {e}")
                for i in range(10):
                    if f'text_pca_{i}' not in df.columns:
                        df[f'text_pca_{i}'] = 0.0
        else:
            # Add empty text features
            for i in range(10):
                if f'text_pca_{i}' not in df.columns:
                    df[f'text_pca_{i}'] = 0.0
        
        # Step 6: Ensure all required features exist
        for feature in self._feature_names:
            if feature not in df.columns:
                # Fill with appropriate defaults
                if 'was_nan' in feature:
                    df[feature] = 1  # Missing indicator
                elif feature.startswith('text_pca_'):
                    df[feature] = 0.0
                elif feature in ['year', 'annee']:
                    df[feature] = None
                else:
                    df[feature] = 0
        
        # Step 7: Select only required features
        df_features = df[self._feature_names].copy()
        
        return df_features
    
    def get_article_count(self, data_dict: Dict[str, List[dict]], siren: str) -> int:
        """
        Get article count for label window [t0 → t1].
        
        Args:
            data_dict: Dictionary with articles data
            siren: SIREN identifier
        
        Returns:
            Article count in label window
        """
        articles_data = data_dict.get('articles', [])
        if not articles_data:
            return 0
        
        df_articles = pd.DataFrame(articles_data)
        if 'publishedAt' not in df_articles.columns:
            return 0
        
        df_articles['publishedAt_parsed'] = (
            pd.to_datetime(df_articles['publishedAt'], errors='coerce', utc=True)
            .dt.tz_localize(None)
        )
        
        # Filter to label window [t0 → t1] and target SIREN
        label_articles = df_articles[
            (df_articles['publishedAt_parsed'] >= T0) & 
            (df_articles['publishedAt_parsed'] < T1) &
            (df_articles['siren'] == str(siren))
        ]
        
        return len(label_articles)

