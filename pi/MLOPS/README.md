# 6G Smart City IDS — MLOps Pipeline

Full MLOps implementation of the 6G Intrusion Detection System, applying all 7 ateliers from the course.

---

## Project Structure

```
6g_ids_mlops/
├── model_pipeline.py          # Atelier 2 — Modular ML functions
├── main.py                    # Atelier 3 — CLI entry point
├── app.py                     # Atelier 4 — FastAPI REST service
├── Makefile                   # Atelier 3/4/5/6/7 — Automation
├── Dockerfile                 # Atelier 6 — Containerisation
├── docker-compose.monitoring.yml  # Atelier 7 — ES + Kibana stack
├── requirements.txt
├── tests/
│   ├── test_pipeline.py       # Unit tests (Atelier 3)
│   └── test_api.py            # API tests (Atelier 4)
└── streamlit_app/
    └── models/                # Exported ensemble artefacts
```

---

## Ateliers Applied

### Atelier 2 — Modularisation du Code (`model_pipeline.py`)

The original Jupyter notebook was restructured into reusable functions:

| Function | Responsibility |
|---|---|
| `prepare_data()` | Load all 4 CSV datasets with auto-detected separator |
| `make_xy()` | Split DataFrame into feature matrix X and target y |
| `build_preprocessor()` | ColumnTransformer: median impute → log1p → RobustScaler |
| `train_model()` | Train any of 6 classifiers, log to MLflow |
| `evaluate_model()` | Metrics + confusion matrix + ROC curve |
| `save_model()` | Persist model to disk via joblib |
| `load_model()` | Reload model from disk |
| `classify_attack_type()` | Rule-based attack subtype (Stage 3 of decision pipeline) |

---

### Atelier 3 — Makefile + main.py

```bash
make install          # pip install -r requirements.txt
make prepare          # load & inspect datasets
make train            # train RandomForest (default)
make train MODEL=XGBoost
make train-all        # train all 6 models
make lint             # flake8 code quality check
make format           # black auto-formatter
make security         # bandit security scan
make test             # pytest unit tests
```

CLI usage:
```bash
python main.py --prepare
python main.py --train --model RandomForest
python main.py --train --model XGBoost
python main.py --all
```

---

### Atelier 4 — FastAPI REST Service (`app.py`)

Start:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# or:
make api
```

Interactive docs: http://localhost:8000/docs

**POST /predict**
```json
{
  "dataset": "eMBB",
  "features": {
    "Dur": 0.2, "TotPkts": 15, "TotBytes": 4800,
    "Rate": 75.0, "Load": 1200.0, "Loss": 0.0,
    "pLoss": 0.01, "TcpRtt": 0.001
  }
}
```

**POST /retrain** (Excellence — exposes retrain as REST endpoint)
```json
{
  "model_name": "RandomForest",
  "n_estimators": 300,
  "max_depth": 20,
  "data_dir": "."
}
```

---

### Atelier 5 — MLflow Tracking

Start MLflow server:
```bash
make mlflow
# → http://localhost:5000
```

Every `train_model()` call logs:
- **Parameters**: dataset, model, n_features, smote_applied, test_size
- **Metrics**: accuracy, f1_macro, roc_auc
- **Artefacts**: the trained model itself

---

### Atelier 6 — Docker

```bash
# Build (rename per instructions: firstname_lastname_class_mlops)
make docker-build IMAGE_NAME=firstname_lastname_class_mlops

# Run locally
make docker-run IMAGE_NAME=firstname_lastname_class_mlops

# Push to Docker Hub
make docker-push DOCKER_USER=yourusername IMAGE_NAME=firstname_lastname_class_mlops
```

The container runs both MLflow (port 5000) and FastAPI (port 8000).

---

### Atelier 7 — Monitoring with Elasticsearch & Kibana

```bash
# Start the monitoring stack
make monitoring-up

# Elasticsearch: http://localhost:9200
# Kibana:        http://localhost:5601
```

Every training run sends metrics to the `mlflow-metrics` Elasticsearch index. In Kibana:
1. Go to **Stack Management → Index Patterns**
2. Create pattern `mlflow-metrics`
3. Use **Discover** to explore logged metrics

The `send_to_elasticsearch()` function in `model_pipeline.py` handles the push automatically during each `train_model()` call.

Excellence: set `ES_HOST` env var to monitor a remote Elasticsearch cluster.

---

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Start MLflow + monitoring
make mlflow-bg
make monitoring-up

# 3. Place datasets in project root, then train
make train-all

# 4. Launch FastAPI
make api
# → http://localhost:8000/docs
```
