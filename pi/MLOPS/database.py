"""
Database module for prediction logging and storage
Uses SQLite for simplicity, can be upgraded to PostgreSQL for production
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "predictions.db"


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """Initialize database schema"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                dataset TEXT NOT NULL,
                prediction TEXT NOT NULL,
                attack_type TEXT,
                severity TEXT,
                confidence REAL,
                alert_status TEXT,
                features TEXT NOT NULL,
                probabilities TEXT,
                shap_values TEXT,
                shap_base_value REAL,
                shap_prediction_score REAL,
                response_time_ms REAL
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON predictions(timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dataset
            ON predictions(dataset)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction
            ON predictions(prediction)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_attack_type
            ON predictions(attack_type)
        """)

        # Statistics table for aggregated metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                dataset TEXT NOT NULL,
                total_predictions INTEGER,
                malicious_count INTEGER,
                benign_count INTEGER,
                avg_confidence REAL,
                attack_type_distribution TEXT
            )
        """)

        print(f"✅ Database initialized: {DB_PATH}")


def log_prediction(
    dataset: str,
    prediction: str,
    attack_type: str,
    severity: str,
    confidence: float,
    alert_status: str,
    features: Dict[str, Any],
    probabilities: Dict[str, float],
    shap_explanation: Optional[Dict[str, Any]] = None,
    response_time_ms: Optional[float] = None,
) -> int:
    """
    Log a prediction to the database

    Returns:
        prediction_id: ID of the inserted record
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Extract SHAP values if present
        shap_values = None
        shap_base_value = None
        shap_prediction_score = None

        if shap_explanation:
            shap_values = json.dumps(shap_explanation.get("shap_values"))
            shap_base_value = shap_explanation.get("base_value")
            shap_prediction_score = shap_explanation.get("prediction_score")

        cursor.execute(
            """
            INSERT INTO predictions (
                dataset, prediction, attack_type, severity, confidence,
                alert_status, features, probabilities, shap_values,
                shap_base_value, shap_prediction_score, response_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                dataset,
                prediction,
                attack_type,
                severity,
                confidence,
                alert_status,
                json.dumps(features),
                json.dumps(probabilities),
                shap_values,
                shap_base_value,
                shap_prediction_score,
                response_time_ms,
            ),
        )

        return cursor.lastrowid


def get_recent_predictions(
    limit: int = 100, dataset: Optional[str] = None
) -> List[Dict]:
    """Get recent predictions"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if dataset:
            cursor.execute(
                """
                SELECT * FROM predictions
                WHERE dataset = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (dataset, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_attack_statistics(
    hours: int = 24, dataset: Optional[str] = None
) -> Dict[str, Any]:
    """Get attack statistics for the last N hours"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Base query
        where_clause = "WHERE timestamp >= datetime('now', '-' || ? || ' hours')"
        params = [hours]

        if dataset:
            where_clause += " AND dataset = ?"
            params.append(dataset)

        # Total predictions
        cursor.execute(
            f"""
            SELECT COUNT(*) as total FROM predictions {where_clause}
        """,
            params,
        )
        total = cursor.fetchone()["total"]

        # Malicious vs Benign
        cursor.execute(
            f"""
            SELECT prediction, COUNT(*) as count
            FROM predictions {where_clause}
            GROUP BY prediction
        """,
            params,
        )
        prediction_counts = {
            row["prediction"]: row["count"] for row in cursor.fetchall()
        }

        # Attack types
        cursor.execute(
            f"""
            SELECT attack_type, COUNT(*) as count
            FROM predictions
            {where_clause} AND prediction = 'Malicious'
            GROUP BY attack_type
            ORDER BY count DESC
        """,
            params,
        )
        attack_types = {row["attack_type"]: row["count"] for row in cursor.fetchall()}

        # Severity distribution
        cursor.execute(
            f"""
            SELECT severity, COUNT(*) as count
            FROM predictions
            {where_clause} AND prediction = 'Malicious'
            GROUP BY severity
            ORDER BY count DESC
        """,
            params,
        )
        severity_dist = {row["severity"]: row["count"] for row in cursor.fetchall()}

        # Average confidence
        cursor.execute(
            f"""
            SELECT AVG(confidence) as avg_confidence
            FROM predictions {where_clause}
        """,
            params,
        )
        avg_confidence = cursor.fetchone()["avg_confidence"] or 0.0

        return {
            "total_predictions": total,
            "malicious_count": prediction_counts.get("Malicious", 0),
            "benign_count": prediction_counts.get("Benign", 0),
            "attack_types": attack_types,
            "severity_distribution": severity_dist,
            "avg_confidence": avg_confidence,
        }


def get_predictions_by_time(
    hours: int = 24, interval_minutes: int = 60, dataset: Optional[str] = None
) -> List[Dict]:
    """Get predictions grouped by time intervals"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        where_clause = "WHERE timestamp >= datetime('now', '-' || ? || ' hours')"
        params = [hours]

        if dataset:
            where_clause += " AND dataset = ?"
            params.append(dataset)

        cursor.execute(
            f"""
            SELECT
                strftime('%Y-%m-%d %H:%M', timestamp,
                    '-' || (strftime('%M', timestamp) % ?) || ' minutes') as time_bucket,
                COUNT(*) as total,
                SUM(CASE WHEN prediction = 'Malicious' THEN 1 ELSE 0 END) as malicious,
                SUM(CASE WHEN prediction = 'Benign' THEN 1 ELSE 0 END) as benign,
                AVG(confidence) as avg_confidence
            FROM predictions
            {where_clause}
            GROUP BY time_bucket
            ORDER BY time_bucket
        """,
            [interval_minutes] + params,
        )

        return [dict(row) for row in cursor.fetchall()]


def get_dataset_metrics() -> List[Dict]:
    """Get metrics per dataset"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                dataset,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN prediction = 'Malicious' THEN 1 ELSE 0 END) as malicious_count,
                SUM(CASE WHEN prediction = 'Benign' THEN 1 ELSE 0 END) as benign_count,
                AVG(confidence) as avg_confidence,
                AVG(response_time_ms) as avg_response_time
            FROM predictions
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY dataset
        """)

        return [dict(row) for row in cursor.fetchall()]


def clear_old_predictions(days: int = 30):
    """Delete predictions older than N days"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM predictions
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """,
            (days,),
        )

        deleted = cursor.rowcount
        print(f"🗑️  Deleted {deleted} predictions older than {days} days")
        return deleted


# Initialize database on module import
init_database()
