# OSE v3 API Documentation

FastAPI-based REST API for the OSE v3 Business Opportunity Engine.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the API

```bash
# Using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or using the provided script
python -m api.main
```

### Accessing the API

- **API Base URL**: `http://localhost:8000`
- **Interactive Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Alternative Documentation (ReDoc)**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## API Endpoints

### Root

- **GET `/`** - API information and available endpoints

### Health Check

- **GET `/health`** - Check API, model, and features status
  - Returns: `status`, `model_loaded`, `features_loaded`, `total_companies`

### Model Information

- **GET `/info`** - Get model details and feature list
  - Returns: `model_type`, `feature_count`, `features`, `model_suffix`

### Predictions

#### Single Prediction (Pre-computed Features)

- **POST `/predict`**
  - **Request Body**:
    ```json
    {
      "siren": "123456789"
    }
    ```
  - **Response**:
    ```json
    {
      "siren": "123456789",
      "score": 0.7523,
      "article_count": 12,
      "confidence": "high",
      "error": null
    }
    ```

#### Batch Prediction

- **POST `/predict/batch`**
  - **Request Body**:
    ```json
    {
      "sirens": ["123456789", "987654321"]
    }
    ```
  - **Response**:
    ```json
    {
      "predictions": [
        {
          "siren": "123456789",
          "score": 0.7523,
          "article_count": 12,
          "confidence": "high",
          "error": null
        }
      ],
      "total": 2,
      "successful": 2,
      "failed": 0
    }
    ```
  - **Limits**: Maximum 100 SIRENs per batch request

#### Prediction from Source Format

- **POST `/predict/from-source`**
  - **Request Body**:
    ```json
    {
      "companies": [
        {
          "siren": "494887854",
          "socialName": "MARS WRIGLEY CONFECTIONERY FRANCE",
          "siret": "49488785400037",
          "caConsolide": 150000000,
          "effectif": 350,
          "kpis": [{"year": 2022, "ca_bilan": 140000000}]
        }
      ],
      "articles": [
        {
          "siren": "494887854",
          "title": "Company news article",
          "publishedAt": "2022-06-15T10:00:00Z",
          "allCompanies": [{"siren": "494887854", "name": "MARS WRIGLEY"}]
        }
      ],
      "projects": [],
      "target_siren": "494887854"
    }
    ```
  - **Response**:
    ```json
    {
      "siren": "494887854",
      "score": 0.9327,
      "article_count": 70,
      "confidence": "high",
      "error": null
    }
    ```
  - **Description**: Accepts data in source format (same as `agro_alim_companies.json`, `agro_alim_articles.json`, `agro_alim_projects.json`) and computes features on-the-fly
  - **Note**: Slower than `/predict` but works for new companies not in pre-computed features

## Example Usage

### Using cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"siren": "123456789"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"sirens": ["123456789", "987654321"]}'

# Health check
curl "http://localhost:8000/health"
```

### Using Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"siren": "123456789"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"sirens": ["123456789", "987654321"]}
)
print(response.json())
```

### Using JavaScript/TypeScript

```javascript
// Single prediction
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ siren: '123456789' })
});
const data = await response.json();
console.log(data);
```

## Response Format

### Prediction Response

- **siren** (string): Company SIREN identifier
- **score** (float, 0-1): Opportunity score (higher = better opportunity)
- **article_count** (int): Number of articles available for the company
- **confidence** (string): "high" if article_count >= 5, "low" otherwise
- **error** (string, optional): Error message if prediction failed

### Error Responses

- **400 Bad Request**: Invalid request (e.g., too many SIRENs in batch)
- **500 Internal Server Error**: Model or prediction error
- **503 Service Unavailable**: Features data not available

## Model Details

### Low-Evidence Shrinkage

Companies with fewer than 5 articles have their scores adjusted (shrunk) toward a base rate of 0.1 to account for limited evidence. This results in:
- Lower confidence scores for companies with sparse data
- More conservative predictions for companies with limited information

### Features

The model uses 84 features including:
- Financial metrics (growth scores, profit scores, capital ratios)
- Text embeddings (PCA components from FastText)
- Workforce data (effectif, effectifConsolide)
- Company structure (nbFilialesDirectes, nbMarques)
- Signal scores (Decidento scores, signal counts)
- Classification flags (startup, fintech, entreprise_familiale, etc.)

## Configuration

### Environment Variables

- **OSE_MODEL_SUFFIX**: Optional model suffix (e.g., "_v2") for loading different model versions
  ```bash
  export OSE_MODEL_SUFFIX="_v2"
  ```

### Required Files

The API expects the following files to exist:
- `models/final_calibrated{SUFFIX}.joblib` - Trained model
- `inference/preprocess{SUFFIX}.joblib` - Preprocessor
- `inference/feature_list{SUFFIX}.json` - Feature list
- `data/features/features.parquet` - Features DataFrame

## Production Deployment

### Using Gunicorn with Uvicorn Workers

```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment-Specific Configuration

For production, consider:
- Setting `allow_origins` in CORS middleware to specific domains
- Using environment variables for configuration
- Adding authentication/authorization middleware
- Setting up logging to a file or service
- Using a reverse proxy (nginx) for SSL termination

## Monitoring

The `/health` endpoint can be used for:
- Kubernetes liveness/readiness probes
- Load balancer health checks
- Monitoring system integration

## Troubleshooting

### Model Not Loading

- Check that model files exist in `models/` and `inference/` directories
- Verify `OSE_MODEL_SUFFIX` environment variable matches file suffixes
- Check file permissions

### Features Not Loading

- Ensure `data/features/features.parquet` exists
- Verify the parquet file is not corrupted
- Check file permissions

### Prediction Errors

- Verify SIREN format (should be 9 digits as string)
- Check that the SIREN exists in the features DataFrame
- Review logs for detailed error messages

