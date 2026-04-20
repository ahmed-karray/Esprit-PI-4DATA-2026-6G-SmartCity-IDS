from pathlib import Path
import joblib
import pandas as pd
import warnings
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from attack_classifier import get_classifier
from shap_explainer import get_explainer
from database import (
    log_prediction,
    get_recent_predictions,
    get_attack_statistics,
    get_predictions_by_time,
    get_dataset_metrics,
)
from drift_monitor import get_drift_monitor

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE_DIR = Path(__file__).resolve().parent
classifier = get_classifier()
explainer = get_explainer()
drift_monitor = get_drift_monitor()

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
    explain: Optional[bool] = False  # Enable SHAP explanations
    generate_plots: Optional[bool] = False  # Generate visualization plots


def load_bundle(dataset_name: str):
    if dataset_name not in MODEL_FILES:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_name}")

    model_path = MODEL_FILES[dataset_name]
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {model_path.name}. Train it first.",
        )

    return joblib.load(model_path)


@app.get("/")
def root():
    return {"message": "6G IDS LightGBM API is running"}


@app.post("/predict")
def predict(req: PredictRequest):
    start_time = time.time()

    bundle = load_bundle(req.dataset)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    label_encoder = bundle["label_encoder"]
    features = bundle["features"]

    row = {feature: req.features.get(feature, None) for feature in features}
    X = pd.DataFrame([row])

    X_proc = preprocessor.transform(X)

    # Convert to numpy array to avoid feature name warnings
    if hasattr(X_proc, "values"):
        X_proc = X_proc.values

    pred = model.predict(X_proc)[0]
    pred_label = label_encoder.inverse_transform([int(pred)])[0]

    proba = None
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_proc)[0]
        proba = {label_encoder.classes_[i]: float(probs[i]) for i in range(len(probs))}
        confidence = float(max(probs))

    # Classify attack subtype if malicious
    attack_type = classifier.classify(req.dataset, req.features, pred_label)
    severity = classifier.get_attack_severity(attack_type)
    recommended_action = classifier.get_recommended_action(attack_type)

    # Confidence gate: flag low confidence predictions
    if confidence < 0.7 and pred_label == "Malicious":
        alert_status = "False Alarm (Low Confidence)"
    elif pred_label == "Malicious":
        alert_status = "Confirmed Attack"
    else:
        alert_status = "Benign Traffic"

    response = {
        "dataset": req.dataset,
        "prediction": pred_label,
        "attack_type": attack_type,
        "severity": severity,
        "confidence": confidence,
        "alert_status": alert_status,
        "recommended_action": recommended_action,
        "probabilities": proba,
        "used_features": features,
    }

    shap_explanation = None

    # Add SHAP explanations if requested
    if req.explain:
        try:
            shap_explanation = explainer.explain_prediction(
                model=model,
                preprocessor=preprocessor,
                features=req.features,
                feature_names=features,
                dataset_name=req.dataset,
            )
            response["shap_explanation"] = shap_explanation

            # Generate plots if requested
            if req.generate_plots:
                import numpy as np

                shap_values = np.array(shap_explanation["shap_values"])
                base_value = shap_explanation["base_value"]
                feature_names = [f"feature_{i}" for i in range(len(shap_values))]
                feature_values = X_proc[0]

                # Generate visualizations
                response["visualizations"] = {
                    "bar_plot": explainer.generate_bar_plot(
                        shap_explanation["feature_importance"]
                    ),
                    "waterfall_plot": explainer.generate_waterfall_plot(
                        shap_values, base_value, feature_names, feature_values
                    ),
                }

        except Exception as e:
            response["shap_explanation"] = {
                "error": f"Failed to generate SHAP explanation: {str(e)}"
            }

    # Calculate response time
    response_time_ms = (time.time() - start_time) * 1000

    # Log prediction to database
    try:
        prediction_id = log_prediction(
            dataset=req.dataset,
            prediction=pred_label,
            attack_type=attack_type,
            severity=severity,
            confidence=confidence,
            alert_status=alert_status,
            features=req.features,
            probabilities=proba,
            shap_explanation=shap_explanation,
            response_time_ms=response_time_ms,
        )
        response["prediction_id"] = prediction_id
    except Exception as e:
        # Don't fail the request if logging fails
        print(f"Warning: Failed to log prediction: {e}")

    response["response_time_ms"] = round(response_time_ms, 2)

    return response


@app.post("/explain")
def explain_prediction(req: PredictRequest):
    """
    Dedicated endpoint for SHAP explanations with visualizations
    """
    # Force explanation and plot generation
    req.explain = True
    req.generate_plots = True
    return predict(req)


@app.get("/stats/recent")
def get_recent_stats(limit: int = 100, dataset: Optional[str] = None):
    """Get recent predictions"""
    try:
        predictions = get_recent_predictions(limit=limit, dataset=dataset)
        return {"predictions": predictions, "count": len(predictions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/attacks")
def get_attack_stats(hours: int = 24, dataset: Optional[str] = None):
    """Get attack statistics for the last N hours"""
    try:
        stats = get_attack_statistics(hours=hours, dataset=dataset)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/timeline")
def get_timeline_stats(
    hours: int = 24, interval_minutes: int = 60, dataset: Optional[str] = None
):
    """Get predictions grouped by time intervals"""
    try:
        timeline = get_predictions_by_time(
            hours=hours, interval_minutes=interval_minutes, dataset=dataset
        )
        return {"timeline": timeline, "count": len(timeline)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/datasets")
def get_datasets_stats():
    """Get metrics per dataset"""
    try:
        metrics = get_dataset_metrics()
        return {"datasets": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/check")
def check_drift(dataset: Optional[str] = None):
    """Check for concept drift"""
    try:
        summary = drift_monitor.get_drift_summary(dataset)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/features/{dataset}")
def check_feature_drift(dataset: str, hours: int = 24, baseline_hours: int = 168):
    """Check for feature distribution drift"""
    try:
        drift = drift_monitor.detect_feature_drift(dataset, hours, baseline_hours)
        return drift
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/performance/{dataset}")
def check_performance_drift(dataset: str, hours: int = 24, baseline_hours: int = 168):
    """Check for model performance drift"""
    try:
        drift = drift_monitor.detect_performance_drift(dataset, hours, baseline_hours)
        return drift
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/retrain/{dataset}")
def should_retrain_model(dataset: str):
    """Check if model should be retrained"""
    try:
        should_retrain, reason = drift_monitor.should_retrain(dataset)
        return {
            "dataset": dataset,
            "should_retrain": should_retrain,
            "reason": reason,
            "timestamp": time.time(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
