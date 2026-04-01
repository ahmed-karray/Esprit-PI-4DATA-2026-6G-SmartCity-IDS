"""
tests/test_api.py — 6G Smart City IDS
FastAPI endpoint tests using TestClient
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_health():
    r = client.get("/")
    assert r.status_code == 200
    assert "status" in r.json()


def test_predict_missing_dataset():
    r = client.post("/predict", json={
        "dataset": "NONEXISTENT",
        "features": {"Dur": 1.0},
    })
    assert r.status_code == 404


def test_predict_embb_returns_structure():
    """Predict endpoint returns expected keys (models may not be loaded in CI)."""
    r = client.post("/predict", json={
        "dataset": "eMBB",
        "features": {
            "Dur": 0.2, "TotPkts": 15, "TotBytes": 4800,
            "Rate": 75.0, "Load": 1200.0, "Loss": 0.0,
            "pLoss": 0.01, "TcpRtt": 0.001,
        },
    })
    # 404 is acceptable when no pre-trained models are present in CI
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        body = r.json()
        assert "binary_prediction" in body
        assert "confidence" in body
        assert "attack_type" in body
