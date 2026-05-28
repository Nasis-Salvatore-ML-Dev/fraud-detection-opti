.PHONY: install lint format type-check test test-integration \
        train tune baseline export-onnx model-card bias-test \
        docker-build docker-run all

install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

type-check:
	mypy src/ --ignore-missing-imports

test:
	pytest tests/unit/ --cov=src --cov-report=term-missing --cov-fail-under=0 -v

test-integration:
	pytest tests/integration/ -v --timeout=30

train:
	python scripts/train.py

tune:
	PYTHONPATH=. python scripts/tune.py

baseline:
	python scripts/compute_baseline.py

export-onnx:
	python scripts/export_onnx.py

model-card:
	python scripts/generate_model_card.py

bias-test:
	PYTHONPATH=. python scripts/run_bias_test.py

docker-build:
	docker build -t fraud-detection-api:local .

docker-run:
	docker run --rm -p 9000:8080 \
		-e MODEL_PATH=models/xgboost_fraud_v1.pkl \
		-e AWS_DEFAULT_REGION=eu-central-1 \
		fraud-detection-api:local

all: format lint type-check test
