# OSE - Business Opportunity Classifier

Rebuild of OSE with leak-proof, time-consistent, text-balanced, calibrated, sector-aware pipeline.

## Key Features

- **Temporal Windowing**: Features strictly < t0 (2023-01-01), labels from [t0 → t1] (2023-01-01 to 2024-01-01)
- **FastText → PCA(20)**: Text embeddings reduced to 20 components
- **Text Cap**: Text features limited to ≤30% of selected features
- **Calibration**: Isotonic calibration (cv=3)
- **SHAP Explainability**: TreeExplainer on dense numeric arrays
- **External Ratios**: INPI ratios from data.economie.gouv.fr
- **No Random Split**: Temporal split by default, fallback to GroupShuffleSplit by SIREN

## Installation

```bash
make setup
make install
```

## Usage

### Full Pipeline

```bash
make all
```

### Individual Steps

```bash
make extract          # Extract data from source files
make fetch_ratios     # Fetch external INPI ratios
make features         # Engineer features (windowing, ratios, text PCA)
make train            # Train and calibrate model
make explain          # Generate SHAP explanations
make eval             # Compute metrics and slices
make ablate           # Run ablation studies
make report           # Generate PDF report
```

## Project Structure

```
_OUTPUT_PROJECT/
├── src/
│   ├── data/          # Data loading, windowing, quality
│   ├── features/      # Feature engineering (text, signals, financials, selection)
│   ├── models/        # Model evaluation, explainability
│   ├── pipeline/       # Pipeline construction
│   ├── external/       # External data fetching
│   ├── reporting/     # PDF report generation
│   └── tests/         # Leak tests
├── data/
│   ├── features/      # Feature datasets
│   ├── labels/        # Label datasets
│   ├── external/      # External data (INPI ratios)
│   └── raw_json/      # Raw extracted data
├── models/            # Trained models
├── reports/           # All reports (JSON, PNG, PDF)
└── inference/         # Inference package
```

## Deliverables

All deliverables are generated in the `reports/` directory:

- `window_validation.json` - Temporal windowing validation
- `missing_summary.json` - Missing data summary
- `kept_features.json` - Kept features report
- `ratio_summary.json` - INPI ratios summary
- `feature_policy.json` - Feature selection policy
- `feature_selection.json` - Selected features
- `metrics.json` - Model metrics
- `sector_metrics.json` - Sector-based metrics
- `text_ablation.json` - Text ablation study
- `leak_tests.json` - Leak test results
- `shap_summary.png` - SHAP global summary
- `shap_examples/` - Local SHAP explanations
- `full_report.pdf` - Comprehensive PDF report

## Inference

See `inference/README.md` for usage examples.

## Configuration

Temporal windows are hardcoded:
- `t0 = 2023-01-01` (feature cutoff)
- `t1 = 2024-01-01` (label end)

Random seed: `42` (all randomness seeded)
