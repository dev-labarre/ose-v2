# Core Concept: Handling Client JSON Data

## The Problem

Your API currently only works with pre-computed features. If a client sends new company data (new SIRENs), you need to compute features on-the-fly.

## The Solution: Feature Engineering Pipeline

### Current Flow (Pre-computed)
```
Client → API → Lookup in features.parquet → Predict
```

### New Flow (On-the-fly)
```
Client → API → Feature Engineering → Predict
```

## Core Concept Breakdown

### 1. Data Transformation Pipeline

Client sends raw JSON (same structure as your source files: company_basic, financial, articles, signals, etc.)

The API runs it through the same feature engineering steps used during training:

- **Merge datasets on SIREN** - Combine all data sources by company identifier
- **Apply temporal filtering** - Use data before t0 (2023-01-01) for features
- **Compute financial scores** - Calculate growth, profit, margins from raw numbers
- **Build signal features** - Count signals by type (B, W, E, F, etc.)
- **Process text** - FastText → PCA if articles are provided
- **Fill missing features** - Use defaults (0, empty lists, etc.)
- **Select 84 required features** - Extract only what the model needs

### 2. Temporal Windowing

- **Features**: Use data before t0 (2023-01-01)
- **Labels**: Use articles from [t0 → t1] for article_count
- The API filters dates accordingly

### 3. Feature Computation

- Reuse the same functions from your training pipeline
- For missing data: fill with defaults (0, empty lists, etc.)
- Ensure all 84 required features are present

### 4. Prediction

After features are computed, use the same model prediction logic and apply shrinkage if article_count < 5.

## Key Architectural Decisions

### Option A: Full Pipeline (More Accurate)

- Run the complete feature engineering pipeline
- Slower but matches training-time features exactly
- Handles complex transformations (text PCA, financial scores, etc.)

### Option B: Simplified Mapping (Faster)

- Accept only essential fields
- Map directly to the 84 features
- Faster but may miss derived features

### Option C: Hybrid Approach (Recommended)

- Try lookup first (fast path)
- Fall back to feature engineering if not found
- Cache computed features for reuse

## Challenges and Considerations

### 1. Performance

- Feature engineering is slower than lookup
- Text processing (FastText → PCA) can be expensive
- Consider caching computed features

### 2. Data Completeness

- Client may not send all 9 datasets
- Need defaults for missing data
- Some features may be less accurate with incomplete data

### 3. External Dependencies

- INPI ratios require external API calls
- May need async processing or background jobs
- Could skip if not critical

### 4. Temporal Consistency

- Client data must respect temporal windows
- Articles must have correct dates
- Financial data should be pre-t0

### 5. Model Compatibility

- Computed features must match training-time features
- Same transformations, same defaults
- Same feature order and names

## The Flow in Simple Terms

1. **Client sends raw company data** (JSON)
2. **API merges datasets** on SIREN
3. **API filters by dates** (pre-t0 for features)
4. **API computes derived features**:
   - Financial scores from raw numbers
   - Signal counts from signal lists
   - Text embeddings from article titles
5. **API fills missing features** with defaults
6. **API extracts the 84 required features**
7. **API runs the model prediction**
8. **API applies shrinkage** if needed
9. **API returns the score**

## Why This Works

- **Same pipeline as training**: Features match what the model expects
- **Flexible**: Accepts partial data and fills defaults
- **Consistent**: Same transformations and logic
- **Extensible**: Can add caching, async processing, etc.

## Trade-offs

| Aspect | Full Pipeline | Simplified Mapping |
|--------|--------------|-------------------|
| **Accuracy** | High (matches training) | Medium (may miss derived features) |
| **Speed** | Slower (complex computation) | Faster (direct mapping) |
| **Complexity** | High (many transformations) | Low (simple mapping) |
| **Data Requirements** | All 9 datasets preferred | Essential fields only |

## The Core Idea

**Reuse your training pipeline at runtime** to transform raw client data into the same 84 features the model expects, then predict.

This ensures:
- ✅ Features match training-time format
- ✅ Model receives compatible input
- ✅ Predictions are consistent with training data
- ✅ New companies can be evaluated without pre-computation

## Implementation Strategy

1. **Start with Option C (Hybrid)**
   - Fast path: lookup in pre-computed features
   - Slow path: compute features on-the-fly
   - Cache results for future requests

2. **Prioritize Essential Features**
   - Financial data → financial scores
   - Articles → text features (if available)
   - Signals → signal counts
   - Fill missing with defaults

3. **Optimize Performance**
   - Cache computed features
   - Use async processing for external APIs
   - Batch process multiple companies

4. **Handle Edge Cases**
   - Missing datasets → use defaults
   - Invalid dates → filter or reject
   - Missing SIREN → return error

## Next Steps

1. Design the API endpoint schema
2. Implement feature engineering service
3. Add caching layer
4. Test with sample client data
5. Monitor performance and optimize

