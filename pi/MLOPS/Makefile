# =============================================================================
# Makefile — 6G Smart City IDS MLOps Pipeline
# Ateliers 3 · 4 · 5 · 6 · 7
# =============================================================================
# Usage:
#   make install          install Python dependencies
#   make prepare          load & inspect datasets
#   make train            train RandomForest (default)
#   make train MODEL=XGBoost
#   make train-all        train all 6 models
#   make evaluate         evaluate saved model
#   make api              start FastAPI server
#   make mlflow           start MLflow tracking server
#   make lint             run flake8 + black check
#   make format           auto-format code with black
#   make security         run bandit security scan
#   make test             run unit tests
#   make docker-build     build Docker image
#   make docker-run       run Docker container
#   make docker-push      push image to Docker Hub
#   make monitoring-up    start Elasticsearch + Kibana stack
#   make monitoring-down  stop the monitoring stack
#   make all              full CI/CD pipeline
# =============================================================================

PYTHON      := python3
PIP         := pip3
VENV        := venv
MODEL       ?= RandomForest
IMAGE_NAME  ?= 6g_ids_mlops
DOCKER_USER ?= your_dockerhub_username
TAG         ?= latest
DATA_DIR    ?= .
MLFLOW_PORT := 5000
API_PORT    := 8000
ES_PORT     := 9200
KIBANA_PORT := 5601

.PHONY: all install venv prepare train train-all evaluate retrain api \
        mlflow lint format security test \
        docker-build docker-run docker-push docker-stop \
        monitoring-up monitoring-down clean help

# ── Default target ─────────────────────────────────────────────────────────
all: install lint test prepare train-all api
	@echo "✅  Full pipeline complete."

# ── Environment ───────────────────────────────────────────────────────────────
venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Activate with: source $(VENV)/bin/activate"

install:
	$(PIP) install -r requirements.txt
	@echo "✅  Dependencies installed."

# ── ML Pipeline (Ateliers 2 & 3) ─────────────────────────────────────────────
prepare:
	$(PYTHON) main.py --prepare

train:
	$(PYTHON) main.py --train --model $(MODEL)

train-all:
	@for m in RandomForest XGBoost LogisticRegression ExtraTrees MLP LightGBM; do \
		echo ">>> Training $$m ..."; \
		$(PYTHON) main.py --train --model $$m; \
	done

evaluate:
	$(PYTHON) main.py --evaluate --model-path model.joblib

retrain:
	$(PYTHON) main.py --retrain

# ── FastAPI service (Atelier 4) ───────────────────────────────────────────────
api:
	uvicorn app:app --reload --host 0.0.0.0 --port $(API_PORT)

api-bg:
	uvicorn app:app --host 0.0.0.0 --port $(API_PORT) &

# ── MLflow server (Atelier 5) ─────────────────────────────────────────────────
mlflow:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 0.0.0.0 \
		--port $(MLFLOW_PORT)

mlflow-bg:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 0.0.0.0 \
		--port $(MLFLOW_PORT) &
	@echo "MLflow UI → http://localhost:$(MLFLOW_PORT)"

# ── Code quality CI (Atelier 3) ───────────────────────────────────────────────
lint:
	flake8 . --max-line-length=100 --exclude=$(VENV),__pycache__,.git

format:
	black . --line-length 100

format-check:
	black . --line-length 100 --check

security:
	bandit -r . --exclude $(VENV) -ll

# ── Tests (Atelier 3) ─────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

# ── Docker (Atelier 6) ────────────────────────────────────────────────────────
docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) .
	@echo "✅  Image built: $(IMAGE_NAME):$(TAG)"

docker-run:
	docker run -d \
		--name 6g_ids_container \
		-p $(API_PORT):$(API_PORT) \
		-p $(MLFLOW_PORT):$(MLFLOW_PORT) \
		-e MLFLOW_TRACKING_URI=http://localhost:$(MLFLOW_PORT) \
		$(IMAGE_NAME):$(TAG)
	@echo "Container running → http://localhost:$(API_PORT)/docs"

docker-stop:
	docker stop 6g_ids_container || true
	docker rm   6g_ids_container || true

docker-push:
	docker tag  $(IMAGE_NAME):$(TAG) $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)
	docker push $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)
	@echo "✅  Image pushed → $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)"

docker-images:
	docker images | grep $(IMAGE_NAME)

# ── Monitoring stack (Atelier 7) ──────────────────────────────────────────────
monitoring-up:
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "Elasticsearch → http://localhost:$(ES_PORT)"
	@echo "Kibana        → http://localhost:$(KIBANA_PORT)"

monitoring-down:
	docker-compose -f docker-compose.monitoring.yml down

monitoring-status:
	@curl -s http://localhost:$(ES_PORT)/_cluster/health | python3 -m json.tool || \
		echo "Elasticsearch not reachable"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f mlflow.db
	rm -rf mlruns .pytest_cache

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make train MODEL=XGBoost"
	@echo "  make docker-build IMAGE_NAME=myname_myclass_mlops"
