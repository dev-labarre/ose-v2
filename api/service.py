"""Service layer for handling predictions and model management."""

import sys
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import joblib
import json
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from inference.predict import load_inference_artifacts, predict_single, predict_batch


class PredictionService:
    """Service for managing model and making predictions."""
    
    def __init__(self, model_suffix: Optional[str] = None):
        """
        Initialize the prediction service.
        
        Args:
            model_suffix: Optional model suffix (e.g., "_v2")
        """
        self.model_suffix = model_suffix or os.getenv("OSE_MODEL_SUFFIX", "")
        self.project_root = PROJECT_ROOT
        self.features_path = self.project_root / "data" / "features" / "features.parquet"
        
        # Lazy loading - will be loaded on first use
        self._preprocessor = None
        self._model = None
        self._feature_names = None
        self._df_features = None
        self._model_info = None
    
    def _load_model(self):
        """Lazy load model artifacts."""
        if self._model is None:
            try:
                self._preprocessor, self._model, self._feature_names = load_inference_artifacts(
                    suffix=self.model_suffix
                )
                self._model_info = {
                    "model_type": type(self._model).__name__,
                    "feature_count": len(self._feature_names),
                    "features": self._feature_names,
                    "model_suffix": self.model_suffix
                }
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _load_features(self):
        """Lazy load features DataFrame and merge labels for article_count."""
        if self._df_features is None:
            if not self.features_path.exists():
                raise FileNotFoundError(
                    f"Features file not found: {self.features_path}. "
                    "Please ensure features.parquet exists in data/features/"
                )
            try:
                self._df_features = pd.read_parquet(self.features_path)
                # Ensure SIREN is string
                if "siren" in self._df_features.columns:
                    self._df_features["siren"] = self._df_features["siren"].astype(str)
                
                # Merge labels to get article_count (needed for shrinkage logic)
                labels_path = self.project_root / "data" / "labels" / "labels.parquet"
                if labels_path.exists():
                    df_labels = pd.read_parquet(labels_path)
                    df_labels["siren"] = df_labels["siren"].astype(str)
                    df_labels = df_labels.rename(columns={
                        "article_count_label": "article_count"
                    })
                    self._df_features = self._df_features.merge(
                        df_labels[["siren", "article_count"]],
                        on="siren",
                        how="left"
                    )
                    # Fill NaN article_count with 0
                    self._df_features["article_count"] = self._df_features["article_count"].fillna(0)
                else:
                    # If labels don't exist, set article_count to 0
                    self._df_features["article_count"] = 0
                    
            except Exception as e:
                raise RuntimeError(f"Failed to load features: {str(e)}")
    
    def ensure_loaded(self):
        """Ensure both model and features are loaded."""
        self._load_model()
        self._load_features()
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def is_features_loaded(self) -> bool:
        """Check if features are loaded."""
        return self._df_features is not None
    
    def get_total_companies(self) -> Optional[int]:
        """Get total number of companies in features."""
        if self._df_features is not None:
            return len(self._df_features)
        return None
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if not self.is_model_loaded():
            self._load_model()
        return self._model_info
    
    def predict(self, siren: str) -> Dict:
        """
        Make a prediction for a single SIREN.
        
        Args:
            siren: SIREN identifier
            
        Returns:
            Prediction dictionary
        """
        self.ensure_loaded()
        return predict_single(siren, self._df_features)
    
    def predict_batch(self, sirens: List[str]) -> List[Dict]:
        """
        Make predictions for multiple SIRENs.
        
        Args:
            sirens: List of SIREN identifiers
            
        Returns:
            List of prediction dictionaries
        """
        self.ensure_loaded()
        return predict_batch(sirens, self._df_features)


# Global service instance (singleton pattern)
_service_instance: Optional[PredictionService] = None


def get_service() -> PredictionService:
    """Get or create the global prediction service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictionService()
    return _service_instance

