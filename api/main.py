"""FastAPI application for OSE v3 Business Opportunity Classifier."""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    SourceFormatRequest
)
from api.service import get_service
from api.extraction_service import ExtractionService
from api.feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting OSE v3 API...")
    service = get_service()
    try:
        # Pre-load model and features on startup
        logger.info("Loading model and features...")
        service.ensure_loaded()
        logger.info("✓ Model and features loaded successfully")
    except Exception as e:
        logger.error(f"⚠️  Failed to load model/features on startup: {e}")
        logger.warning("API will attempt lazy loading on first request")
    
    yield
    
    # Shutdown
    logger.info("Shutting down OSE v3 API...")


# Create FastAPI app
app = FastAPI(
    title="OSE v3 API",
    description="""
    **Business Opportunity Classifier API**
    
    This API provides predictions for business opportunities based on company SIREN identifiers.
    
    ## Features
    
    - **Single Predictions**: Get opportunity scores for individual companies
    - **Batch Predictions**: Process multiple companies at once (up to 100)
    - **Low-Evidence Shrinkage**: Automatically adjusts scores for companies with limited data
    - **Confidence Levels**: Provides confidence indicators based on article count
    
    ## Model Details
    
    - Uses calibrated XGBoost classifier
    - Features include financial metrics, text embeddings (PCA), workforce data, and signals
    - Scores range from 0.0 to 1.0 (higher = better opportunity)
    - Companies with < 5 articles have scores shrunk toward base rate (0.1)
    
    ## Usage
    
    1. Use `/predict` for single company predictions
    2. Use `/predict/batch` for multiple companies
    3. Check `/health` for service status
    4. View `/info` for model details
    """,
    version="3.0.0",
    contact={
        "name": "OSE Team",
        "email": "support@ose.example.com",
    },
    license_info={
        "name": "Proprietary",
    },
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "OSE v3 API",
        "version": "3.0.0",
        "description": "Business Opportunity Classifier API",
        "docs": "/docs",
        "health": "/health",
        "info": "/info"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API, model, and features.
    """
    service = get_service()
    
    try:
        model_loaded = service.is_model_loaded()
        features_loaded = service.is_features_loaded()
        total_companies = service.get_total_companies()
        
        if model_loaded and features_loaded:
            status_msg = "healthy"
        else:
            status_msg = "degraded"
        
        return HealthResponse(
            status=status_msg,
            model_loaded=model_loaded,
            features_loaded=features_loaded,
            total_companies=total_companies
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            features_loaded=False,
            total_companies=None
        )


@app.get("/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Get model information.
    
    Returns details about the loaded model including feature list.
    """
    service = get_service()
    
    try:
        info = service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Predict opportunity score for a single company.
    
    - **siren**: 9-digit SIREN identifier (French company ID)
    
    Returns:
    - **score**: Opportunity score (0.0 to 1.0)
    - **article_count**: Number of articles available for the company
    - **confidence**: "high" if article_count >= 5, "low" otherwise
    """
    service = get_service()
    
    try:
        result = service.predict(request.siren)
        
        # Convert to response model
        return PredictionResponse(
            siren=result["siren"],
            score=result.get("score"),
            article_count=result.get("article_count", 0),
            confidence=result.get("confidence", "low"),
            error=result.get("error")
        )
    except FileNotFoundError as e:
        logger.error(f"Features file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Features data not available. Please ensure features.parquet exists."
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict opportunity scores for multiple companies.
    
    - **sirens**: List of SIREN identifiers (1-100 companies)
    
    Returns:
    - **predictions**: List of prediction results
    - **total**: Total number of predictions requested
    - **successful**: Number of successful predictions
    - **failed**: Number of failed predictions
    """
    service = get_service()
    
    if len(request.sirens) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 SIRENs allowed per batch request"
        )
    
    try:
        results = service.predict_batch(request.sirens)
        
        # Convert to response models
        predictions = [
            PredictionResponse(
                siren=r["siren"],
                score=r.get("score"),
                article_count=r.get("article_count", 0),
                confidence=r.get("confidence", "low"),
                error=r.get("error")
            )
            for r in results
        ]
        
        successful = sum(1 for p in predictions if p.error is None)
        failed = len(predictions) - successful
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            successful=successful,
            failed=failed
        )
    except FileNotFoundError as e:
        logger.error(f"Features file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Features data not available. Please ensure features.parquet exists."
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/predict/from-source", response_model=PredictionResponse, tags=["Predictions"])
async def predict_from_source(request: SourceFormatRequest):
    """
    Predict from source format JSON (companies, articles, projects).
    
    Accepts data in the same format as your source files (agro_alim_companies.json,
    agro_alim_articles.json, agro_alim_projects.json) and computes features on-the-fly.
    
    **Note**: This is slower than `/predict` but works for new companies not in
    pre-computed features.
    
    - **companies**: List of company objects (required)
    - **articles**: List of article objects (optional)
    - **projects**: List of project objects (optional, for signals)
    - **target_siren**: Specific SIREN to predict (optional, uses first company if not provided)
    """
    from inference.predict import load_inference_artifacts
    
    try:
        # Validate input
        if not request.companies or len(request.companies) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one company is required"
            )
        
        # Determine target SIREN
        target_siren = request.target_siren
        if not target_siren:
            # Use first company's SIREN
            first_company = request.companies[0]
            target_siren = str(first_company.get('siren', ''))
            if not target_siren:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="SIREN not found in company data. Provide target_siren or ensure companies have 'siren' field."
                )
        
        # Step 1: Extract data into 9 datasets
        extraction_service = ExtractionService()
        data_dict = extraction_service.extract_from_source(
            companies=request.companies,
            articles=request.articles,
            projects=request.projects
        )
        
        # Step 2: Engineer features
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.process_extracted_data(
            data_dict,
            target_siren=target_siren
        )
        
        if len(df_features) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No features computed for SIREN {target_siren}"
            )
        
        # Step 3: Get article count for shrinkage
        article_count = feature_engineer.get_article_count(data_dict, target_siren)
        
        # Step 4: Make prediction
        preprocessor, model, feature_names = load_inference_artifacts()
        
        # Get company row (should be single row after filtering)
        company_data = df_features.iloc[0:1]  # Get first row as DataFrame
        
        X = company_data[feature_names]
        
        # Predict
        proba = model.predict_proba(X)[0]
        score = proba[1]  # Positive class probability
        
        # Apply shrinkage
        if article_count < 5:
            base_rate = 0.1
            shrinkage_factor = article_count / 5
            score = shrinkage_factor * score + (1 - shrinkage_factor) * base_rate
        
        return PredictionResponse(
            siren=str(target_siren),
            score=float(score),
            article_count=int(article_count),
            confidence='high' if article_count >= 5 else 'low',
            error=None
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature engineering/prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process data: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

