from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any


BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "eMBB": BASE_DIR / "lightgbm_eMBB.joblib",
    "mMTC": BASE_DIR / "lightgbm_mMTC.joblib",
    "URLLC": BASE_DIR / "lightgbm_URLLC.joblib",
    "TON_IoT": BASE_DIR / "lightgbm_TON_IoT.joblib",
    "train_test_network": BASE_DIR / "lightgbm_TON_IoT.joblib",
}

app = FastAPI(title="6G IDS API", version="1.0")


class PredictRequest(BaseModel):
    dataset: str
    features: Dict[str, Any]


def load_bundle(dataset_name: str):
    if dataset_name not in MODEL_FILES:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_name}")

    model_path = MODEL_FILES[dataset_name]
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {model_path.name}. Train it first."
        )

    return joblib.load(model_path)


@app.get("/")
def root():
    return {"message": "6G IDS LightGBM API is running"}


@app.post("/predict")
def predict(req: PredictRequest):
    bundle = load_bundle(req.dataset)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    label_encoder = bundle["label_encoder"]
    features = bundle["features"]

    row = {feature: req.features.get(feature, None) for feature in features}
    X = pd.DataFrame([row])

    X_proc = preprocessor.transform(X)
    pred = model.predict(X_proc)[0]
    pred_label = label_encoder.inverse_transform([int(pred)])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_proc)[0]
        proba = {label_encoder.classes_[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "dataset": req.dataset,
        "prediction": pred_label,
        "probabilities": proba,
        "used_features": features
    }