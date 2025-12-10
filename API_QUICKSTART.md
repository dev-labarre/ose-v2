# OSE v3 API - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
# Option 1: Using the run script
python run_api.py

# Option 2: Using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **Swagger UI (Interactive Docs)**: http://localhost:8000/docs
- **ReDoc (Alternative Docs)**: http://localhost:8000/redoc
- **API Base**: http://localhost:8000

## 📝 Quick Test

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction (replace with actual SIREN)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"siren": "123456789"}'
```

### Using Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"siren": "123456789"}
)
print(response.json())
```

### Using the Browser

1. Open http://localhost:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Enter your data
5. Click "Execute"

## 📚 Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/info` | Model information |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions (up to 100) |

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Use a different model version
export OSE_MODEL_SUFFIX="_v2"
```

### Required Files

The API needs these files to work:
- `models/final_calibrated{SUFFIX}.joblib`
- `inference/preprocess{SUFFIX}.joblib`
- `inference/feature_list{SUFFIX}.json`
- `data/features/features.parquet`

## 🐳 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
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

## 📖 Full Documentation

See `api/README.md` for complete API documentation.

## ❓ Troubleshooting

**API won't start?**
- Check that all required files exist
- Verify Python version (3.8+)
- Check port 8000 is not in use

**Predictions fail?**
- Verify `data/features/features.parquet` exists
- Check SIREN format (should be 9-digit string)
- Review logs for detailed errors

**Model not loading?**
- Check `OSE_MODEL_SUFFIX` matches file suffixes
- Verify model files exist in `models/` and `inference/`
- Check file permissions

