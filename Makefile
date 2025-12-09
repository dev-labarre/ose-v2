.PHONY: setup install clean extract fetch_ratios features train explain eval classifications ablate report test all help

# Default Python version
PYTHON_VERSION := 3.10.6
VENV_NAME := ose-env

help:
	@echo "OSE - Business Opportunity Classifier"
	@echo "Available commands:"
	@echo "  make setup         - Create virtual environment"
	@echo "  make install       - Install dependencies"
	@echo "  make clean         - Remove generated files (models, reports, data artifacts)"
	@echo "  make extract       - Extract data from source files to data/raw_json/"
	@echo "  make fetch_ratios  - Cache external INPI ratios to data/external/inpi_ratios.parquet"
	@echo "  make features      - Build temporal windows (features < t0, labels [t0→t1])"
	@echo "  make train         - Fit and calibrate model"
	@echo "  make explain       - Generate SHAP artifacts"
	@echo "  make eval          - Compute metrics, slices, calibration, PR-AUC"
	@echo "  make classifications - Generate all company classifications JSON"
	@echo "  make ablate        - Run text-only, tabular-only, full ablations"
	@echo "  make report        - Generate PDF report"
	@echo "  make test          - Run pytest test suite"
	@echo "  make all           - Full pipeline (clean → extract → features → train → explain → eval → classifications → ablate → report)"

setup:
	@echo "🚀 Setting up Python environment..."
	@if command -v pyenv > /dev/null 2>&1; then \
		echo "📦 Using pyenv..."; \
		pyenv install -s $(PYTHON_VERSION) || true; \
		pyenv virtualenv $(PYTHON_VERSION) $(VENV_NAME) || true; \
		echo "✅ Virtual environment '$(VENV_NAME)' created"; \
		echo "📋 To activate: pyenv activate $(VENV_NAME)"; \
	else \
		echo "📦 Using venv..."; \
		python3 -m venv $(VENV_NAME); \
		echo "✅ Virtual environment '$(VENV_NAME)' created"; \
		echo "📋 To activate: source $(VENV_NAME)/bin/activate"; \
	fi

install:
	@echo "📚 Installing dependencies..."
	@if command -v pyenv > /dev/null 2>&1 && pyenv versions | grep -q $(VENV_NAME); then \
		eval "$$(pyenv init -)" && pyenv activate $(VENV_NAME) && python -m pip install --upgrade pip && pip install -r requirements.txt; \
	elif [ -d "$(VENV_NAME)" ]; then \
		$(VENV_NAME)/bin/pip install --upgrade pip && $(VENV_NAME)/bin/pip install -r requirements.txt; \
	else \
		python3 -m pip install --upgrade pip && pip install -r requirements.txt; \
	fi
	@echo "✅ Dependencies installed"

clean:
	@echo "🧹 Cleaning generated files..."
	@rm -rf models/*.joblib
	@rm -rf models/*.pkl
	@rm -rf reports/*.json
	@rm -rf reports/*.png
	@rm -rf reports/*.pdf
	@rm -rf data/features/*
	@rm -rf data/labels/*
	@rm -rf data/external/*.parquet
	@rm -rf inference/*.joblib
	@rm -rf inference/*.json
	@find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✅ Clean completed"

extract:
	@echo "📥 Extracting data from source files..."
	@if command -v pyenv > /dev/null 2>&1 && pyenv versions | grep -q $(VENV_NAME); then \
		eval "$$(pyenv init -)" && pyenv activate $(VENV_NAME) && python -m src.extract_all; \
	elif [ -d "$(VENV_NAME)" ]; then \
		$(VENV_NAME)/bin/python -m src.extract_all; \
	else \
		python3 -m src.extract_all; \
	fi
	@echo "✅ Extraction completed"

fetch_ratios:
	@echo "📊 Fetching external INPI ratios..."
	@if command -v pyenv > /dev/null 2>&1 && pyenv versions | grep -q $(VENV_NAME); then \
		eval "$$(pyenv init -)" && pyenv activate $(VENV_NAME) && python -m src.external.fetch_inpi_ratios; \
	elif [ -d "$(VENV_NAME)" ]; then \
		$(VENV_NAME)/bin/python -m src.external.fetch_inpi_ratios; \
	else \
		python3 -m src.external.fetch_inpi_ratios; \
	fi
	@echo "✅ Ratio fetching completed"

features:
	@echo "🔧 Engineering features..."
	@if command -v pyenv > /dev/null 2>&1 && pyenv versions | grep -q $(VENV_NAME); then \
		eval "$$(pyenv init -)" && pyenv activate $(VENV_NAME) && python -c "from src.data.windowing import build_temporal_windows; build_temporal_windows()"; \
	elif [ -d "$(VENV_NAME)" ]; then \
		$(VENV_NAME)/bin/python -c "from src.data.windowing import build_temporal_windows; build_temporal_windows()"; \
	else \
		python3 -c "from src.data.windowing import build_temporal_windows; build_temporal_windows()"; \
	fi
	@echo "✅ Feature engineering completed"

train:
	@echo "🚀 Training model..."
	@if command -v pyenv > /dev/null 2>&1 && pyenv versions | grep -q $(VENV_NAME); then \
		eval "$$(pyenv init -)" && pyenv activate $(VENV_NAME) && python -m src.train; \
	elif [ -d "$(VENV_NAME)" ]; then \
		$(VENV_NAME)/bin/python -m src.train; \
	else \
		python3 -m src.train; \
	fi
	@echo "✅ Training completed"

explain:
	@echo "📊 Generating SHAP explanations..."
	@if command -v pyenv > /dev/null 2>&1 && pyenv versions | grep -q $(VENV_NAME); then \
		eval "$$(pyenv init -)" && pyenv activate $(VENV_NAME) && python -c "from src.models.explainability import generate_shap_explanations; generate_shap_explanations()"; \
	elif [ -d "$(VENV_NAME)" ]; then \
		$(VENV_NAME)/bin/python -c "from src.models.explainability import generate_shap_explanations; generate_shap_explanations()"; \
	else \
		python3 -c "from src.models.explainability import generate_shap_explanations; generate_shap_explanations()"; \
	fi
	@echo "✅ SHAP explanations completed"

eval:
	@echo "📈 Evaluating model..."
	@if command -v pyenv > /dev/null 2>&1 && pyenv versions | grep -q $(VENV_NAME); then \
		eval "$$(pyenv init -)" && pyenv activate $(VENV_NAME) && python -c "from src.models.evaluator import run_evaluation; run_evaluation()"; \
	elif [ -d "$(VENV_NAME)" ]; then \
		$(VENV_NAME)/bin/python -c "from src.models.evaluator import run_evaluation; run_evaluation()"; \
	else \
		python3 -c "from src.models.evaluator import run_evaluation; run_evaluation()"; \
	fi
	@echo "✅ Evaluation completed"

classifications:
	@echo "📋 Generating all company classifications..."
	@if command -v pyenv > /dev/null 2>&1 && pyenv versions | grep -q $(VENV_NAME); then \
		eval "$$(pyenv init -)" && pyenv activate $(VENV_NAME) && python -c "from src.reporting.generate_classifications import extract_all_classifications; extract_all_classifications()"; \
	elif [ -d "$(VENV_NAME)" ]; then \
		$(VENV_NAME)/bin/python -c "from src.reporting.generate_classifications import extract_all_classifications; extract_all_classifications()"; \
	else \
		python3 -c "from src.reporting.generate_classifications import extract_all_classifications; extract_all_classifications()"; \
	fi
	@echo "✅ Classifications generation completed"

ablate:
	@echo "🔬 Running ablation studies..."
	@if command -v pyenv > /dev/null 2>&1 && pyenv versions | grep -q $(VENV_NAME); then \
		eval "$$(pyenv init -)" && pyenv activate $(VENV_NAME) && python -c "from src.models.evaluator import run_ablation_study; run_ablation_study()"; \
	elif [ -d "$(VENV_NAME)" ]; then \
		$(VENV_NAME)/bin/python -c "from src.models.evaluator import run_ablation_study; run_ablation_study()"; \
	else \
		python3 -c "from src.models.evaluator import run_ablation_study; run_ablation_study()"; \
	fi
	@echo "✅ Ablation studies completed"

report:
	@echo "📄 Generating PDF report..."
	@if command -v pyenv > /dev/null 2>&1 && pyenv versions | grep -q $(VENV_NAME); then \
		eval "$$(pyenv init -)" && pyenv activate $(VENV_NAME) && python -c "from src.reporting.generate_report import generate_full_report; generate_full_report()"; \
	elif [ -d "$(VENV_NAME)" ]; then \
		$(VENV_NAME)/bin/python -c "from src.reporting.generate_report import generate_full_report; generate_full_report()"; \
	else \
		python3 -c "from src.reporting.generate_report import generate_full_report; generate_full_report()"; \
	fi
	@echo "✅ Report generation completed"

test:
	@echo "🧪 Running test suite..."
	@if command -v pyenv > /dev/null 2>&1 && pyenv versions | grep -q $(VENV_NAME); then \
		eval "$$(pyenv init -)" && pyenv activate $(VENV_NAME) && python -m pytest src/tests -v; \
	elif [ -d "$(VENV_NAME)" ]; then \
		$(VENV_NAME)/bin/python -m pytest src/tests -v; \
	else \
		python3 -m pytest src/tests -v; \
	fi
	@echo "✅ Test suite completed"

all: clean extract features train explain eval classifications ablate report
	@echo "✅ Full pipeline completed!"
