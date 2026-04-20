"""
SHAP Explainability Service
Provides interpretable explanations for model predictions using SHAP values
"""

import shap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for thread safety
import matplotlib.pyplot as plt  # noqa: E402
import io  # noqa: E402
import base64  # noqa: E402
from typing import Dict, Any, List  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")


class SHAPExplainer:
    """SHAP-based explainability service for LightGBM models"""

    def __init__(self):
        self.explainers = {}  # Cache explainers per dataset
        self.feature_names = {}  # Cache feature names per dataset

    def get_explainer(self, model, X_background, dataset_name: str):
        """
        Get or create SHAP TreeExplainer for a model

        Args:
            model: Trained LightGBM model
            X_background: Background data for SHAP (sample of training data)
            dataset_name: Dataset identifier

        Returns:
            SHAP TreeExplainer instance
        """
        if dataset_name not in self.explainers:
            # Use TreeExplainer for tree-based models (LightGBM)
            self.explainers[dataset_name] = shap.TreeExplainer(
                model, X_background, feature_perturbation="tree_path_dependent"
            )
        return self.explainers[dataset_name]

    def explain_prediction(
        self,
        model,
        preprocessor,
        features: Dict[str, Any],
        feature_names: List[str],
        dataset_name: str,
        X_background=None,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction

        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            features: Feature dictionary
            feature_names: List of feature names
            dataset_name: Dataset identifier
            X_background: Background data (optional, uses zeros if None)

        Returns:
            Dictionary with SHAP values and explanations
        """
        # Prepare input
        row = {feature: features.get(feature, None) for feature in feature_names}
        X = pd.DataFrame([row])
        X_proc = preprocessor.transform(X)

        # Convert to numpy if needed
        if hasattr(X_proc, "values"):
            X_proc = X_proc.values

        # Create background data if not provided
        if X_background is None:
            X_background = np.zeros((1, X_proc.shape[1]))

        # Get explainer
        explainer = self.get_explainer(model, X_background, dataset_name)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_proc)

        # Handle binary classification (LightGBM returns values for positive class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class (Malicious)

        # Get base value (expected value)
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]

        # Create feature importance ranking
        shap_abs = np.abs(shap_values[0])
        feature_importance = []

        for i, importance in enumerate(shap_abs):
            feature_importance.append(
                {
                    "feature": f"feature_{i}",
                    "shap_value": float(shap_values[0][i]),
                    "importance": float(importance),
                    "feature_value": float(X_proc[0][i]),
                }
            )

        # Sort by absolute importance
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        # Get top contributing features
        top_features = feature_importance[:5]

        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            top_features, base_value, shap_values[0].sum()
        )

        return {
            "shap_values": shap_values[0].tolist(),
            "base_value": float(base_value),
            "feature_importance": feature_importance,
            "top_features": top_features,
            "explanation": explanation_text,
            "prediction_score": float(base_value + shap_values[0].sum()),
        }

    def _generate_explanation_text(
        self, top_features: List[Dict], base_value: float, total_shap: float
    ) -> str:
        """
        Generate human-readable explanation text

        Args:
            top_features: Top contributing features
            base_value: SHAP base value
            total_shap: Sum of SHAP values

        Returns:
            Explanation text
        """
        prediction_score = base_value + total_shap

        if prediction_score > 0:
            decision = "MALICIOUS"
            confidence = "high" if abs(prediction_score) > 1.0 else "moderate"
        else:
            decision = "BENIGN"
            confidence = "high" if abs(prediction_score) > 1.0 else "moderate"

        explanation = f"Prediction: {decision} (confidence: {confidence})\n\n"
        explanation += f"Base prediction score: {base_value:.3f}\n"
        explanation += f"Final prediction score: {prediction_score:.3f}\n\n"
        explanation += "Top contributing features:\n"

        for i, feat in enumerate(top_features, 1):
            direction = "increases" if feat["shap_value"] > 0 else "decreases"
            explanation += (
                f"{i}. {feat['feature']} (value: {feat['feature_value']:.3f})\n"
            )
            explanation += (
                f"   → {direction} malicious score by {abs(feat['shap_value']):.3f}\n"
            )

        return explanation

    def generate_waterfall_plot(
        self,
        shap_values: np.ndarray,
        base_value: float,
        feature_names: List[str],
        feature_values: np.ndarray,
        max_display: int = 10,
    ) -> str:
        """
        Generate SHAP waterfall plot as base64 image

        Args:
            shap_values: SHAP values array
            base_value: Base value
            feature_names: Feature names
            feature_values: Feature values
            max_display: Maximum features to display

        Returns:
            Base64 encoded PNG image
        """
        try:
            plt.figure(figsize=(10, 6))

            # Create explanation object for waterfall plot
            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=feature_values,
                feature_names=feature_names,
            )

            shap.plots.waterfall(explanation, max_display=max_display, show=False)

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close()
            buf.seek(0)

            # Encode to base64
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            print(f"Error generating waterfall plot: {e}")
            return None

    def generate_force_plot(
        self,
        shap_values: np.ndarray,
        base_value: float,
        feature_names: List[str],
        feature_values: np.ndarray,
    ) -> str:
        """
        Generate SHAP force plot as base64 image

        Args:
            shap_values: SHAP values array
            base_value: Base value
            feature_names: Feature names
            feature_values: Feature values

        Returns:
            Base64 encoded PNG image
        """
        try:
            plt.figure(figsize=(12, 3))

            # Create force plot
            shap.force_plot(
                base_value,
                shap_values,
                feature_values,
                feature_names=feature_names,
                matplotlib=True,
                show=False,
            )

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close()
            buf.seek(0)

            # Encode to base64
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            print(f"Error generating force plot: {e}")
            return None

    def generate_bar_plot(self, feature_importance: List[Dict], top_n: int = 10) -> str:
        """
        Generate feature importance bar plot

        Args:
            feature_importance: List of feature importance dicts
            top_n: Number of top features to show

        Returns:
            Base64 encoded PNG image
        """
        try:
            # Get top N features
            top_features = feature_importance[:top_n]

            # Create bar plot
            plt.figure(figsize=(10, 6))
            features = [f["feature"] for f in top_features]
            importances = [f["importance"] for f in top_features]

            colors = ["red" if f["shap_value"] > 0 else "blue" for f in top_features]

            plt.barh(features, importances, color=colors)
            plt.xlabel("SHAP Value (Impact on Prediction)")
            plt.title("Top Feature Contributions")
            plt.gca().invert_yaxis()

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="red", label="Increases Malicious Score"),
                Patch(facecolor="blue", label="Decreases Malicious Score"),
            ]
            plt.legend(handles=legend_elements, loc="lower right")

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close()
            buf.seek(0)

            # Encode to base64
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            print(f"Error generating bar plot: {e}")
            return None


# Singleton instance
_explainer = None


def get_explainer() -> SHAPExplainer:
    """Get or create singleton explainer instance"""
    global _explainer
    if _explainer is None:
        _explainer = SHAPExplainer()
    return _explainer
