# Inference Package

## Usage

### Single Prediction

```python
from inference.predict import predict_single
import pandas as pd

df_features = pd.read_parquet('data/features/features.parquet')
result = predict_single('123456789', df_features)
print(result)
```

### Batch Prediction

```python
from inference.predict import predict_batch

siren_list = ['123456789', '987654321']
results = predict_batch(siren_list, df_features)
```

## Output Format

```json
{
  "siren": "123456789",
  "score": 0.7523,
  "article_count": 12,
  "confidence": "high"
}
```

**Note**: Probabilities are NOT rounded. Scores are returned as raw floats.

## Low-Evidence Shrinkage

If `article_count < 5`, probabilities are shrunk toward the base rate (0.1) to account for low evidence.
