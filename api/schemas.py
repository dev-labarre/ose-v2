"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "siren": "123456789"
            }
        }
    )
    
    siren: str = Field(..., description="SIREN identifier (9 digits)", examples=["123456789"])


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sirens": ["123456789", "987654321"]
            }
        }
    )
    
    sirens: List[str] = Field(..., description="List of SIREN identifiers", min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "siren": "123456789",
                "score": 0.7523,
                "article_count": 12,
                "confidence": "high",
                "error": None
            }
        }
    )
    
    siren: str = Field(..., description="SIREN identifier")
    score: Optional[float] = Field(None, description="Opportunity score (0-1)", ge=0, le=1)
    article_count: int = Field(0, description="Number of articles for the company")
    confidence: Literal["high", "low"] = Field(..., description="Confidence level based on article count")
    error: Optional[str] = Field(None, description="Error message if prediction failed")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {
                        "siren": "123456789",
                        "score": 0.7523,
                        "article_count": 12,
                        "confidence": "high",
                        "error": None
                    }
                ],
                "total": 1,
                "successful": 1,
                "failed": 0
            }
        }
    )
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "features_loaded": True,
                "total_companies": 1500
            }
        }
    )
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    features_loaded: bool = Field(..., description="Whether features are loaded")
    total_companies: Optional[int] = Field(None, description="Total number of companies in features")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_type": "CalibratedClassifier",
                "feature_count": 84,
                "features": ["effectif_workforce_was_nan", "text_pca_2"],
                "model_suffix": ""
            }
        }
    )
    
    model_type: str = Field(..., description="Type of model")
    feature_count: int = Field(..., description="Number of features")
    features: List[str] = Field(..., description="List of feature names")
    model_suffix: str = Field(..., description="Model suffix identifier")


class SourceFormatRequest(BaseModel):
    """Request with source format JSON (companies, articles, projects)."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "companies": [{"siren": "123456789", "socialName": "ABC Corp", ...}],
                "articles": [{"title": "...", "publishedAt": "2022-06-01", ...}],
                "projects": [{"type": "...", "publishedAt": "2022-06-01", ...}]
            }
        }
    )
    
    companies: Optional[List[dict]] = Field(None, description="List of company objects")
    articles: Optional[List[dict]] = Field(None, description="List of article objects")
    projects: Optional[List[dict]] = Field(None, description="List of project objects")
    target_siren: Optional[str] = Field(None, description="Specific SIREN to predict (if not provided, uses first company's SIREN)")

