"""Unit tests for the current LightGBM pipeline helpers."""

import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path
from shap_explainer import SHAPExplainer, get_explainer

from model_pipeline import (
    FEATURE_MAP,
    build_preprocessor,
    list_dataset_files,
    make_xy,
    model_path_for,
    normalize_dataset_name,
)


@pytest.fixture
def sample_embb_df():
    """Minimal synthetic eMBB-style DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "Dur": rng.uniform(0, 5, n),
            "TotPkts": rng.integers(1, 500, n),
            "TotBytes": rng.integers(64, 100_000, n),
            "Rate": rng.uniform(0, 1000, n),
            "Load": rng.uniform(0, 50_000, n),
            "Loss": rng.uniform(0, 0.1, n),
            "pLoss": rng.uniform(0, 0.05, n),
            "TcpRtt": rng.uniform(0, 0.2, n),
            "Label": rng.choice(["Benign", "Malicious"], n),
        }
    )
    return df


@pytest.fixture
def sample_toniot_df():
    """Minimal synthetic TON_IoT-style DataFrame for testing."""
    rng = np.random.default_rng(7)
    n = 80
    df = pd.DataFrame(
        {
            "src_bytes": rng.integers(100, 200_000, n),
            "dst_bytes": rng.integers(100, 200_000, n),
            "src_pkts": rng.integers(1, 2000, n),
            "dst_pkts": rng.integers(1, 500, n),
            "duration": rng.uniform(0, 120, n),
            "proto": rng.choice(["tcp", "udp", "icmp"], n),
            "conn_state": rng.choice(["SF", "REJ", "RSTO", "S0"], n),
            "service": rng.choice(["http", "ssh", "-", "ftp"], n),
            "Label": rng.choice(["Benign", "Malicious"], n),
        }
    )
    return df


def test_make_xy_returns_correct_shapes(sample_embb_df):
    X, y = make_xy(sample_embb_df)
    assert len(X) == len(y) == len(sample_embb_df)
    assert "Label" not in X.columns


def test_make_xy_drops_metadata_columns():
    df = pd.DataFrame(
        {
            "Dur": [1.0],
            "TotPkts": [10],
            "Label": ["Benign"],
            "UniqueID": [99],
            "timestamp": ["2024-01-01"],
        }
    )
    X, _ = make_xy(df)
    assert "UniqueID" not in X.columns
    assert "timestamp" not in X.columns


def test_build_preprocessor_numeric(sample_embb_df):
    X, _ = make_xy(sample_embb_df)
    pre = build_preprocessor(X)
    X_proc = pre.fit_transform(X)
    assert X_proc.shape[0] == len(X)
    assert not np.any(np.isnan(X_proc))


def test_build_preprocessor_mixed(sample_toniot_df):
    X, _ = make_xy(sample_toniot_df)
    pre = build_preprocessor(X)
    X_proc = pre.fit_transform(X)
    assert X_proc.shape[0] == len(X)
    assert not np.any(np.isnan(X_proc))


def test_feature_map_all_datasets():
    for ds in ["mMTC", "URLLC", "eMBB", "TON_IoT"]:
        assert ds in FEATURE_MAP
        assert len(FEATURE_MAP[ds]) > 0


def test_normalize_dataset_name_alias():
    assert normalize_dataset_name("train_test_network") == "TON_IoT"
    assert normalize_dataset_name("eMBB") == "eMBB"


def test_list_dataset_files_contains_expected_keys():
    files = list_dataset_files()
    assert set(files) == {"mMTC", "URLLC", "eMBB", "TON_IoT"}


def test_model_path_for_uses_dataset_name():
    model_path = model_path_for("eMBB")
    assert model_path.name == "lightgbm_eMBB.joblib"


# SHAP Explainability Tests
BASE_DIR = Path(__file__).resolve().parent


@pytest.fixture
def explainer():
    return SHAPExplainer()


@pytest.fixture
def real_model_bundle():
    """Load a real trained model for testing"""
    model_path = BASE_DIR / "lightgbm_mMTC.joblib"

    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}. Run 'make train' first.")

    bundle = joblib.load(model_path)
    return bundle


@pytest.fixture
def model(real_model_bundle):
    """Extract model from bundle"""
    return real_model_bundle["model"]


@pytest.fixture
def preprocessor(real_model_bundle):
    """Extract preprocessor from bundle"""
    return real_model_bundle["preprocessor"]


@pytest.fixture
def feature_names(real_model_bundle):
    """Extract feature names from bundle"""
    return real_model_bundle["features"]


@pytest.fixture
def sample_features():
    """Sample features for mMTC dataset"""
    return {
        "TotPkts": 500,
        "Rate": 100,
        "SrcGap": 0.01,
        "DstGap": 0.02,
        "Dur": 10.5,
        "Load": 0.8,
        "Loss": 5,
        "TcpRtt": 0.05,
    }


def test_explainer_singleton():
    """Test that get_explainer returns singleton instance"""
    e1 = get_explainer()
    e2 = get_explainer()
    assert e1 is e2


def test_explain_prediction_structure(
    explainer, model, preprocessor, feature_names, sample_features
):
    """Test that explanation has correct structure"""
    explanation = explainer.explain_prediction(
        model=model,
        preprocessor=preprocessor,
        features=sample_features,
        feature_names=feature_names,
        dataset_name="mMTC",
    )

    # Check required keys
    assert "shap_values" in explanation
    assert "base_value" in explanation
    assert "feature_importance" in explanation
    assert "top_features" in explanation
    assert "explanation" in explanation
    assert "prediction_score" in explanation


def test_shap_values_type(
    explainer, model, preprocessor, feature_names, sample_features
):
    """Test that SHAP values are returned as list"""
    explanation = explainer.explain_prediction(
        model=model,
        preprocessor=preprocessor,
        features=sample_features,
        feature_names=feature_names,
        dataset_name="mMTC",
    )

    assert isinstance(explanation["shap_values"], list)
    assert len(explanation["shap_values"]) > 0


def test_feature_importance_ranking(
    explainer, model, preprocessor, feature_names, sample_features
):
    """Test that features are ranked by importance"""
    explanation = explainer.explain_prediction(
        model=model,
        preprocessor=preprocessor,
        features=sample_features,
        feature_names=feature_names,
        dataset_name="mMTC",
    )

    feature_importance = explanation["feature_importance"]

    # Check that features are sorted by importance (descending)
    for i in range(len(feature_importance) - 1):
        assert (
            feature_importance[i]["importance"]
            >= feature_importance[i + 1]["importance"]
        )


def test_top_features_limit(
    explainer, model, preprocessor, feature_names, sample_features
):
    """Test that top features are limited to 5"""
    explanation = explainer.explain_prediction(
        model=model,
        preprocessor=preprocessor,
        features=sample_features,
        feature_names=feature_names,
        dataset_name="mMTC",
    )

    assert len(explanation["top_features"]) <= 5


def test_explanation_text_generation(
    explainer, model, preprocessor, feature_names, sample_features
):
    """Test that explanation text is generated"""
    explanation = explainer.explain_prediction(
        model=model,
        preprocessor=preprocessor,
        features=sample_features,
        feature_names=feature_names,
        dataset_name="mMTC",
    )

    explanation_text = explanation["explanation"]

    # Check that explanation contains key information
    assert "Prediction:" in explanation_text
    assert "confidence:" in explanation_text
    assert "Base prediction score:" in explanation_text
    assert "Top contributing features:" in explanation_text


def test_prediction_score_calculation(
    explainer, model, preprocessor, feature_names, sample_features
):
    """Test that prediction score is calculated correctly"""
    explanation = explainer.explain_prediction(
        model=model,
        preprocessor=preprocessor,
        features=sample_features,
        feature_names=feature_names,
        dataset_name="mMTC",
    )

    # Prediction score should be base_value + sum(shap_values)
    expected_score = explanation["base_value"] + sum(explanation["shap_values"])
    assert abs(explanation["prediction_score"] - expected_score) < 0.001


def test_bar_plot_generation(explainer):
    """Test bar plot generation"""
    feature_importance = [
        {"feature": "f1", "shap_value": 0.5, "importance": 0.5},
        {"feature": "f2", "shap_value": -0.3, "importance": 0.3},
        {"feature": "f3", "shap_value": 0.2, "importance": 0.2},
    ]

    plot = explainer.generate_bar_plot(feature_importance, top_n=3)

    # Check that plot is generated (base64 string or None)
    assert plot is None or isinstance(plot, str)
    if plot:
        assert plot.startswith("data:image/png;base64,")


def test_explainer_caching(
    explainer, model, preprocessor, feature_names, sample_features
):
    """Test that explainers are cached per dataset"""
    # First call
    explainer.explain_prediction(
        model=model,
        preprocessor=preprocessor,
        features=sample_features,
        feature_names=feature_names,
        dataset_name="mMTC",
    )

    # Second call with same dataset
    explainer.explain_prediction(
        model=model,
        preprocessor=preprocessor,
        features=sample_features,
        feature_names=feature_names,
        dataset_name="mMTC",
    )

    # Check that explainer is cached
    assert "mMTC" in explainer.explainers


def test_multiple_datasets(explainer):
    """Test handling of multiple datasets"""
    # This test will load all available models
    datasets = ["mMTC", "URLLC", "eMBB", "TON_IoT"]

    for dataset in datasets:
        model_path = BASE_DIR / f"lightgbm_{dataset}.joblib"

        if not model_path.exists():
            continue  # Skip if model not trained

        bundle = joblib.load(model_path)
        model = bundle["model"]
        preprocessor = bundle["preprocessor"]
        feature_names = bundle["features"]

        # Create sample features (use first feature with dummy value)
        features = {feature_names[0]: 100}

        explanation = explainer.explain_prediction(
            model=model,
            preprocessor=preprocessor,
            features=features,
            feature_names=feature_names,
            dataset_name=dataset,
        )

        assert explanation is not None
        assert "shap_values" in explanation
