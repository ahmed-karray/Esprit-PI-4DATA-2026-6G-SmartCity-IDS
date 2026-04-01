# =============================================================================
# Dockerfile — 6G Smart City IDS MLOps
# Atelier 6: Conteneurisation avec Docker
# =============================================================================
# Build:
#   docker build -t firstname_lastname_class_mlops .
# Run:
#   docker run -p 8000:8000 -p 5000:5000 firstname_lastname_class_mlops
# =============================================================================

# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="6G IDS MLOps"
LABEL description="6G Smart City Intrusion Detection System — FastAPI + MLflow"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy project files ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining project files (datasets mounted at runtime via volume)
COPY . .

# ── Expose ports ──────────────────────────────────────────────────────────────
# 8000 — FastAPI / Uvicorn
# 5000 — MLflow tracking server
EXPOSE 8000
EXPOSE 5000

# ── Environment variables ─────────────────────────────────────────────────────
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV PYTHONUNBUFFERED=1

# ── Entrypoint script ─────────────────────────────────────────────────────────
# Starts MLflow in the background, then launches FastAPI
CMD mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlruns \
        --host 0.0.0.0 \
        --port 5000 & \
    uvicorn app:app \
        --host 0.0.0.0 \
        --port 8000
