"""
Concept Drift Monitoring Service
Detects data drift and model performance degradation
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path
import joblib
import json
from database import get_db_connection

BASE_DIR = Path(__file__).resolve().parent


class DriftMonitor:
    """Monitor for concept drift and model performance degradation"""

    def __init__(self):
        self.drift_thresholds = {
            "ks_test": 0.05,  # p-value threshold for KS test
            "f1_drop": 0.10,  # 10% drop in F1 score triggers alert
            "confidence_drop": 0.15,  # 15% drop in avg confidence
        }
        self.baseline_stats = {}
        self.load_baseline_stats()

    def load_baseline_stats(self):
        """Load baseline statistics from trained models"""
        datasets = ["mMTC", "URLLC", "eMBB", "TON_IoT"]

        for dataset in datasets:
            model_path = BASE_DIR / f"lightgbm_{dataset}.joblib"
            if model_path.exists():
                try:
                    bundle = joblib.load(model_path)
                    # Store baseline from training
                    self.baseline_stats[dataset] = {
                        "features": bundle["features"],
                        "loaded_at": datetime.now().isoformat(),
                    }
                except Exception as e:
                    print(f"Warning: Could not load baseline for {dataset}: {e}")

    def kolmogorov_smirnov_test(
        self, baseline_data: np.ndarray, current_data: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test to detect distribution drift

        Returns:
            (statistic, p_value): KS statistic and p-value
        """
        if len(baseline_data) == 0 or len(current_data) == 0:
            return 0.0, 1.0

        statistic, p_value = stats.ks_2samp(baseline_data, current_data)
        return float(statistic), float(p_value)

    def detect_feature_drift(
        self, dataset: str, hours: int = 24, baseline_hours: int = 168
    ) -> Dict[str, any]:
        """
        Detect drift in feature distributions

        Args:
            dataset: Dataset name
            hours: Recent time window to check
            baseline_hours: Baseline time window (older data)

        Returns:
            Dictionary with drift detection results
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get recent predictions
            cursor.execute(
                """
                SELECT features FROM predictions
                WHERE dataset = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """,
                (dataset, hours),
            )
            recent_rows = cursor.fetchall()

            # Get baseline predictions (older data)
            cursor.execute(
                """
                SELECT features FROM predictions
                WHERE dataset = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                AND timestamp < datetime('now', '-' || ? || ' hours')
            """,
                (dataset, baseline_hours, hours),
            )
            baseline_rows = cursor.fetchall()

            if not recent_rows or not baseline_rows:
                return {
                    "drift_detected": False,
                    "reason": "Insufficient data for drift detection",
                    "recent_count": len(recent_rows),
                    "baseline_count": len(baseline_rows),
                }

            # Parse features
            recent_features = [json.loads(row["features"]) for row in recent_rows]
            baseline_features = [json.loads(row["features"]) for row in baseline_rows]

            # Convert to DataFrames
            recent_df = pd.DataFrame(recent_features)
            baseline_df = pd.DataFrame(baseline_features)

            # Perform KS test for each numeric feature
            drift_results = {}
            drifted_features = []

            for col in recent_df.columns:
                if pd.api.types.is_numeric_dtype(recent_df[col]):
                    recent_vals = recent_df[col].dropna().values
                    baseline_vals = baseline_df[col].dropna().values

                    if len(recent_vals) > 0 and len(baseline_vals) > 0:
                        ks_stat, p_value = self.kolmogorov_smirnov_test(
                            baseline_vals, recent_vals
                        )

                        drift_results[col] = {
                            "ks_statistic": ks_stat,
                            "p_value": p_value,
                            "drift_detected": p_value
                            < self.drift_thresholds["ks_test"],
                            "recent_mean": float(recent_vals.mean()),
                            "baseline_mean": float(baseline_vals.mean()),
                            "mean_change_pct": (
                                float(
                                    (recent_vals.mean() - baseline_vals.mean())
                                    / baseline_vals.mean()
                                    * 100
                                )
                                if baseline_vals.mean() != 0
                                else 0.0
                            ),
                        }

                        if p_value < self.drift_thresholds["ks_test"]:
                            drifted_features.append(col)

            return {
                "drift_detected": len(drifted_features) > 0,
                "drifted_features": drifted_features,
                "drift_count": len(drifted_features),
                "total_features": len(drift_results),
                "feature_drift_details": drift_results,
                "recent_count": len(recent_rows),
                "baseline_count": len(baseline_rows),
                "threshold": self.drift_thresholds["ks_test"],
            }

    def detect_performance_drift(
        self, dataset: str, hours: int = 24, baseline_hours: int = 168
    ) -> Dict[str, any]:
        """
        Detect drift in model performance metrics

        Args:
            dataset: Dataset name
            hours: Recent time window
            baseline_hours: Baseline time window

        Returns:
            Dictionary with performance drift results
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Recent performance
            cursor.execute(
                """
                SELECT
                    AVG(confidence) as avg_confidence,
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction = 'Malicious' THEN 1 ELSE 0 END) as malicious,
                    SUM(CASE WHEN confidence < 0.7 THEN 1 ELSE 0 END) as low_confidence
                FROM predictions
                WHERE dataset = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
            """,
                (dataset, hours),
            )
            recent = cursor.fetchone()

            # Baseline performance
            cursor.execute(
                """
                SELECT
                    AVG(confidence) as avg_confidence,
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction = 'Malicious' THEN 1 ELSE 0 END) as malicious,
                    SUM(CASE WHEN confidence < 0.7 THEN 1 ELSE 0 END) as low_confidence
                FROM predictions
                WHERE dataset = ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                AND timestamp < datetime('now', '-' || ? || ' hours')
            """,
                (dataset, baseline_hours, hours),
            )
            baseline = cursor.fetchone()

            if (
                not recent
                or not baseline
                or recent["total"] == 0
                or baseline["total"] == 0
            ):
                return {
                    "drift_detected": False,
                    "reason": "Insufficient data for performance drift detection",
                }

            # Calculate metrics
            recent_conf = recent["avg_confidence"] or 0.0
            baseline_conf = baseline["avg_confidence"] or 0.0
            conf_change = (
                (recent_conf - baseline_conf) / baseline_conf
                if baseline_conf > 0
                else 0
            )

            recent_malicious_rate = recent["malicious"] / recent["total"]
            baseline_malicious_rate = baseline["malicious"] / baseline["total"]
            malicious_rate_change = (
                (recent_malicious_rate - baseline_malicious_rate)
                / baseline_malicious_rate
                if baseline_malicious_rate > 0
                else 0
            )

            recent_low_conf_rate = recent["low_confidence"] / recent["total"]
            baseline_low_conf_rate = baseline["low_confidence"] / baseline["total"]

            # Detect drift
            confidence_drift = conf_change < -self.drift_thresholds["confidence_drop"]
            malicious_rate_drift = abs(malicious_rate_change) > 0.5  # 50% change

            return {
                "drift_detected": confidence_drift or malicious_rate_drift,
                "confidence_drift": confidence_drift,
                "malicious_rate_drift": malicious_rate_drift,
                "metrics": {
                    "recent_avg_confidence": recent_conf,
                    "baseline_avg_confidence": baseline_conf,
                    "confidence_change_pct": conf_change * 100,
                    "recent_malicious_rate": recent_malicious_rate,
                    "baseline_malicious_rate": baseline_malicious_rate,
                    "malicious_rate_change_pct": malicious_rate_change * 100,
                    "recent_low_confidence_rate": recent_low_conf_rate,
                    "baseline_low_confidence_rate": baseline_low_conf_rate,
                },
                "recent_count": recent["total"],
                "baseline_count": baseline["total"],
            }

    def get_drift_summary(self, dataset: Optional[str] = None) -> Dict[str, any]:
        """
        Get comprehensive drift summary

        Args:
            dataset: Specific dataset or None for all

        Returns:
            Drift summary with recommendations
        """
        datasets = [dataset] if dataset else ["mMTC", "URLLC", "eMBB", "TON_IoT"]
        results = {}

        for ds in datasets:
            feature_drift = self.detect_feature_drift(ds, hours=24, baseline_hours=168)
            performance_drift = self.detect_performance_drift(
                ds, hours=24, baseline_hours=168
            )

            drift_detected = feature_drift.get(
                "drift_detected", False
            ) or performance_drift.get("drift_detected", False)

            # Generate recommendations
            recommendations = []
            if feature_drift.get("drift_detected"):
                recommendations.append(
                    f"Feature drift detected in {feature_drift['drift_count']} features"
                )
                recommendations.append("Consider retraining the model with recent data")

            if performance_drift.get("confidence_drift"):
                recommendations.append("Model confidence has dropped significantly")
                recommendations.append("Investigate model performance degradation")

            if performance_drift.get("malicious_rate_drift"):
                recommendations.append("Significant change in malicious traffic rate")
                recommendations.append(
                    "Review attack patterns and update detection rules"
                )

            if not drift_detected:
                recommendations.append("No significant drift detected")
                recommendations.append("Model is performing within expected parameters")

            results[ds] = {
                "drift_detected": drift_detected,
                "feature_drift": feature_drift,
                "performance_drift": performance_drift,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat(),
            }

        return results

    def should_retrain(self, dataset: str) -> Tuple[bool, str]:
        """
        Determine if model should be retrained

        Returns:
            (should_retrain, reason)
        """
        summary = self.get_drift_summary(dataset)
        drift_info = summary.get(dataset, {})

        if not drift_info:
            return False, "No drift information available"

        feature_drift = drift_info.get("feature_drift", {})
        performance_drift = drift_info.get("performance_drift", {})

        # Retrain if significant feature drift
        if feature_drift.get("drift_count", 0) >= 3:
            return (
                True,
                f"Significant feature drift detected in {feature_drift['drift_count']} features",
            )

        # Retrain if confidence dropped significantly
        if performance_drift.get("confidence_drift"):
            conf_change = performance_drift.get("metrics", {}).get(
                "confidence_change_pct", 0
            )
            return True, f"Model confidence dropped by {abs(conf_change):.1f}%"

        # Retrain if malicious rate changed dramatically
        if performance_drift.get("malicious_rate_drift"):
            rate_change = performance_drift.get("metrics", {}).get(
                "malicious_rate_change_pct", 0
            )
            return True, f"Malicious traffic rate changed by {abs(rate_change):.1f}%"

        return False, "Model performance is stable"

    def log_drift_check(self, dataset: str, drift_summary: Dict):
        """Log drift check results to database"""
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Create drift_logs table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    dataset TEXT NOT NULL,
                    drift_detected BOOLEAN,
                    feature_drift_count INTEGER,
                    confidence_drift BOOLEAN,
                    malicious_rate_drift BOOLEAN,
                    recommendations TEXT,
                    should_retrain BOOLEAN,
                    retrain_reason TEXT
                )
            """)

            should_retrain_flag, retrain_reason = self.should_retrain(dataset)

            cursor.execute(
                """
                INSERT INTO drift_logs (
                    dataset, drift_detected, feature_drift_count,
                    confidence_drift, malicious_rate_drift,
                    recommendations, should_retrain, retrain_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    dataset,
                    drift_summary.get("drift_detected", False),
                    drift_summary.get("feature_drift", {}).get("drift_count", 0),
                    drift_summary.get("performance_drift", {}).get(
                        "confidence_drift", False
                    ),
                    drift_summary.get("performance_drift", {}).get(
                        "malicious_rate_drift", False
                    ),
                    json.dumps(drift_summary.get("recommendations", [])),
                    should_retrain_flag,
                    retrain_reason,
                ),
            )


# Singleton instance
_monitor = None


def get_drift_monitor() -> DriftMonitor:
    """Get or create singleton drift monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = DriftMonitor()
    return _monitor
