"""
Streamlit Dashboard for 6G Smart City IDS
Real-time monitoring, SHAP visualizations, and attack statistics
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import base64
from io import BytesIO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="6G Smart City IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-high {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-medium {
        background-color: #fff9c4;
        border-left: 4px solid #ffc107;
    }
    .alert-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
</style>
""",
    unsafe_allow_html=True,
)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def make_prediction(dataset, features, explain=False, generate_plots=False):
    """Make a prediction via API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "dataset": dataset,
                "features": features,
                "explain": explain,
                "generate_plots": generate_plots,
            },
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def get_attack_statistics(hours=24, dataset=None):
    """Get attack statistics"""
    try:
        params = {"hours": hours}
        if dataset:
            params["dataset"] = dataset
        response = requests.get(f"{API_URL}/stats/attacks", params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_timeline_data(hours=24, interval_minutes=60, dataset=None):
    """Get timeline data"""
    try:
        params = {"hours": hours, "interval_minutes": interval_minutes}
        if dataset:
            params["dataset"] = dataset
        response = requests.get(f"{API_URL}/stats/timeline", params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_dataset_metrics():
    """Get per-dataset metrics"""
    try:
        response = requests.get(f"{API_URL}/stats/datasets", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_recent_predictions(limit=100, dataset=None):
    """Get recent predictions"""
    try:
        params = {"limit": limit}
        if dataset:
            params["dataset"] = dataset
        response = requests.get(f"{API_URL}/stats/recent", params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def check_drift(dataset=None):
    """Check for concept drift"""
    try:
        params = {}
        if dataset:
            params["dataset"] = dataset
        response = requests.get(f"{API_URL}/drift/check", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def check_feature_drift(dataset, hours=24, baseline_hours=168):
    """Check for feature drift"""
    try:
        params = {"hours": hours, "baseline_hours": baseline_hours}
        response = requests.get(
            f"{API_URL}/drift/features/{dataset}", params=params, timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def check_performance_drift(dataset, hours=24, baseline_hours=168):
    """Check for performance drift"""
    try:
        params = {"hours": hours, "baseline_hours": baseline_hours}
        response = requests.get(
            f"{API_URL}/drift/performance/{dataset}", params=params, timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def should_retrain(dataset):
    """Check if model should be retrained"""
    try:
        response = requests.get(f"{API_URL}/drift/retrain/{dataset}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def display_base64_image(base64_str):
    """Display base64 encoded image"""
    if base64_str and base64_str.startswith("data:image/png;base64,"):
        img_data = base64_str.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        st.image(img, use_container_width=True)


# Sidebar
st.sidebar.title("🛡️ 6G IDS Dashboard")

# Check API health
api_status = check_api_health()
if api_status:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Offline")
    st.error("⚠️ Cannot connect to API. Please start the API server with: `make api`")
    st.stop()

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔍 Live Prediction",
        "📊 Statistics",
        "🎯 SHAP Analysis",
        "📈 Timeline",
        "⚠️ Drift Monitor",
    ],
)

# Dataset filter
dataset_filter = st.sidebar.selectbox(
    "Filter by Dataset",
    ["All", "mMTC", "URLLC", "eMBB", "TON_IoT"],
)
dataset_param = None if dataset_filter == "All" else dataset_filter

# Time range filter
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
)
hours_map = {
    "Last Hour": 1,
    "Last 6 Hours": 6,
    "Last 24 Hours": 24,
    "Last 7 Days": 168,
}
hours = hours_map[time_range]

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Main content
st.markdown(
    '<div class="main-header">🛡️ 6G Smart City IDS Dashboard</div>',
    unsafe_allow_html=True,
)

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.header("System Overview")

    # Get statistics
    stats = get_attack_statistics(hours=hours, dataset=dataset_param)
    dataset_metrics = get_dataset_metrics()

    if stats:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Predictions",
                f"{stats['total_predictions']:,}",
                help="Total predictions in selected time range",
            )

        with col2:
            malicious_pct = (
                (stats["malicious_count"] / stats["total_predictions"] * 100)
                if stats["total_predictions"] > 0
                else 0
            )
            st.metric(
                "Malicious Traffic",
                f"{stats['malicious_count']:,}",
                f"{malicious_pct:.1f}%",
                delta_color="inverse",
            )

        with col3:
            st.metric(
                "Benign Traffic",
                f"{stats['benign_count']:,}",
                help="Benign traffic detected",
            )

        with col4:
            st.metric(
                "Avg Confidence",
                f"{stats['avg_confidence']:.1%}",
                help="Average prediction confidence",
            )

        st.divider()

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Attack Type Distribution")
            if stats["attack_types"]:
                attack_df = pd.DataFrame(
                    list(stats["attack_types"].items()),
                    columns=["Attack Type", "Count"],
                )
                fig = px.pie(
                    attack_df,
                    values="Count",
                    names="Attack Type",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No malicious traffic detected in this time range")

        with col2:
            st.subheader("Severity Distribution")
            if stats["severity_distribution"]:
                severity_df = pd.DataFrame(
                    list(stats["severity_distribution"].items()),
                    columns=["Severity", "Count"],
                )
                severity_colors = {
                    "Critical": "#f44336",
                    "High": "#ff9800",
                    "Medium": "#ffc107",
                    "Low": "#4caf50",
                }
                fig = px.bar(
                    severity_df,
                    x="Severity",
                    y="Count",
                    color="Severity",
                    color_discrete_map=severity_colors,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No severity data available")

        st.divider()

        # Dataset metrics
        if dataset_metrics and dataset_metrics.get("datasets"):
            st.subheader("Dataset Performance")
            metrics_df = pd.DataFrame(dataset_metrics["datasets"])
            if not metrics_df.empty:
                st.dataframe(
                    metrics_df.style.format(
                        {
                            "avg_confidence": "{:.2%}",
                            "avg_response_time": "{:.2f} ms",
                        }
                    ),
                    use_container_width=True,
                )

# ============================================================================
# PAGE: LIVE PREDICTION
# ============================================================================
elif page == "🔍 Live Prediction":
    st.header("Live Prediction Interface")

    # Dataset selection
    dataset = st.selectbox("Select Dataset", ["mMTC", "URLLC", "eMBB", "TON_IoT"])

    # Feature definitions
    feature_configs = {
        "mMTC": [
            "TotPkts",
            "Rate",
            "SrcGap",
            "DstGap",
            "Dur",
            "Load",
            "Loss",
            "TcpRtt",
        ],
        "URLLC": [
            "TcpRtt",
            "SynAck",
            "AckDat",
            "Loss",
            "Dur",
            "Rate",
            "TotPkts",
            "TotBytes",
        ],
        "eMBB": [
            "Dur",
            "TotPkts",
            "TotBytes",
            "Rate",
            "Load",
            "Loss",
            "pLoss",
            "TcpRtt",
        ],
        "TON_IoT": [
            "src_bytes",
            "dst_bytes",
            "src_pkts",
            "dst_pkts",
            "duration",
            "proto",
            "conn_state",
            "service",
        ],
    }

    features = {}
    st.subheader("Enter Feature Values")

    cols = st.columns(3)
    for idx, feature in enumerate(feature_configs[dataset]):
        with cols[idx % 3]:
            if feature in ["proto", "conn_state", "service"]:
                features[feature] = st.text_input(feature, value="tcp")
            else:
                features[feature] = st.number_input(feature, value=100.0, format="%.4f")

    # Options
    col1, col2 = st.columns(2)
    with col1:
        explain = st.checkbox("Generate SHAP Explanation", value=True)
    with col2:
        generate_plots = st.checkbox("Generate Visualizations", value=True)

    # Predict button
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            result = make_prediction(dataset, features, explain, generate_plots)

        if result:
            st.success("✅ Prediction Complete!")

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                pred_class = (
                    "alert-critical"
                    if result["prediction"] == "Malicious"
                    else "alert-low"
                )
                st.markdown(
                    f'<div class="metric-card {pred_class}"><h3>Prediction</h3>'
                    f'<h2>{result["prediction"]}</h2></div>',
                    unsafe_allow_html=True,
                )

            with col2:
                severity_class = f"alert-{result['severity'].lower()}"
                st.markdown(
                    f'<div class="metric-card {severity_class}"><h3>Attack Type</h3>'
                    f'<h2>{result["attack_type"]}</h2></div>',
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f'<div class="metric-card"><h3>Confidence</h3>'
                    f'<h2>{result["confidence"]:.1%}</h2></div>',
                    unsafe_allow_html=True,
                )

            st.divider()

            # Additional info
            st.subheader("Details")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Alert Status:** {result['alert_status']}")
                st.write(f"**Severity:** {result['severity']}")
                st.write(
                    f"**Response Time:** {result.get('response_time_ms', 0):.2f} ms"
                )

            with col2:
                st.write("**Recommended Action:**")
                st.info(result["recommended_action"])

            # SHAP Explanation
            if explain and "shap_explanation" in result:
                st.divider()
                st.subheader("🎯 SHAP Explanation")

                shap_exp = result["shap_explanation"]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Base Value", f"{shap_exp['base_value']:.4f}")
                with col2:
                    st.metric("Prediction Score", f"{shap_exp['prediction_score']:.4f}")

                st.write("**Explanation:**")
                st.text(shap_exp["explanation"])

                # Visualizations
                if generate_plots and "visualizations" in result:
                    st.subheader("📊 SHAP Visualizations")

                    viz = result["visualizations"]

                    col1, col2 = st.columns(2)
                    with col1:
                        if viz.get("bar_plot"):
                            st.write("**Feature Importance**")
                            display_base64_image(viz["bar_plot"])

                    with col2:
                        if viz.get("waterfall_plot"):
                            st.write("**Waterfall Plot**")
                            display_base64_image(viz["waterfall_plot"])

# ============================================================================
# PAGE: STATISTICS
# ============================================================================
elif page == "📊 Statistics":
    st.header("Attack Statistics")

    stats = get_attack_statistics(hours=hours, dataset=dataset_param)

    if stats:
        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Predictions", f"{stats['total_predictions']:,}")

        with col2:
            attack_rate = (
                (stats["malicious_count"] / stats["total_predictions"] * 100)
                if stats["total_predictions"] > 0
                else 0
            )
            st.metric("Attack Rate", f"{attack_rate:.1f}%")

        with col3:
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")

        st.divider()

        # Attack types table
        if stats["attack_types"]:
            st.subheader("Attack Types Breakdown")
            attack_df = pd.DataFrame(
                list(stats["attack_types"].items()),
                columns=["Attack Type", "Count"],
            )
            attack_df["Percentage"] = (
                attack_df["Count"] / attack_df["Count"].sum() * 100
            )
            st.dataframe(
                attack_df.style.format({"Percentage": "{:.1f}%"}),
                use_container_width=True,
            )

        # Recent predictions
        st.divider()
        st.subheader("Recent Predictions")

        recent = get_recent_predictions(limit=50, dataset=dataset_param)
        if recent and recent.get("predictions"):
            df = pd.DataFrame(recent["predictions"])
            # Select relevant columns
            display_cols = [
                "timestamp",
                "dataset",
                "prediction",
                "attack_type",
                "severity",
                "confidence",
            ]
            if all(col in df.columns for col in display_cols):
                st.dataframe(
                    df[display_cols].head(20),
                    use_container_width=True,
                )

# ============================================================================
# PAGE: SHAP ANALYSIS
# ============================================================================
elif page == "🎯 SHAP Analysis":
    st.header("SHAP Explainability Analysis")

    st.info(
        "💡 Use the Live Prediction page to generate SHAP explanations for individual predictions"
    )

    st.write("""
    ### What is SHAP?

    SHAP SHapley Additive exPlanations provides interpretable explanations for model predictions by:

    - **Feature Attribution**: Shows how much each feature contributed to the prediction
    - **Transparency**: Makes black-box models interpretable
    - **Trust**: Builds confidence in automated security decisions

    ### How to Use:

    1. Go to the **Live Prediction** page
    2. Enter feature values for your traffic sample
    3. Enable "Generate SHAP Explanation"
    4. Enable "Generate Visualizations"
    5. Click "Predict" to see detailed explanations

    ### Interpretation:

    - **Positive SHAP values** (red): Increase malicious score
    - **Negative SHAP values** (blue): Decrease malicious score
    - **Magnitude**: Larger absolute values = stronger influence
    """)

# ============================================================================
# PAGE: TIMELINE
# ============================================================================
elif page == "📈 Timeline":
    st.header("Prediction Timeline")

    timeline_data = get_timeline_data(
        hours=hours, interval_minutes=60, dataset=dataset_param
    )

    if timeline_data and timeline_data.get("timeline"):
        df = pd.DataFrame(timeline_data["timeline"])

        if not df.empty:
            # Line chart
            st.subheader("Predictions Over Time")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["time_bucket"],
                    y=df["malicious"],
                    name="Malicious",
                    line=dict(color="#f44336", width=2),
                    fill="tozeroy",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df["time_bucket"],
                    y=df["benign"],
                    name="Benign",
                    line=dict(color="#4caf50", width=2),
                    fill="tozeroy",
                )
            )
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Count",
                hovermode="x unified",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Confidence over time
            st.subheader("Average Confidence Over Time")
            fig2 = px.line(
                df,
                x="time_bucket",
                y="avg_confidence",
                markers=True,
                line_shape="spline",
            )
            fig2.update_layout(
                xaxis_title="Time",
                yaxis_title="Avg Confidence",
                yaxis_tickformat=".0%",
                height=300,
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Data table
            st.subheader("Timeline Data")
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No timeline data available for the selected time range")

# ============================================================================
# PAGE: DRIFT MONITOR
# ============================================================================
elif page == "⚠️ Drift Monitor":
    st.header("Concept Drift Monitoring")

    st.info("""
    **Concept Drift** occurs when the statistical properties of data change over time,
    which can degrade model performance. This page monitors:
    - **Feature Drift**: Changes in input feature distributions
    - **Performance Drift**: Changes in model confidence and prediction patterns
    """)

    # Dataset selection
    monitor_dataset = st.selectbox(
        "Select Dataset to Monitor",
        ["mMTC", "URLLC", "eMBB", "TON_IoT"],
        key="drift_dataset",
    )

    # Time range settings
    col1, col2 = st.columns(2)
    with col1:
        recent_hours = st.slider("Recent Window (hours)", 1, 72, 24)
    with col2:
        baseline_hours = st.slider("Baseline Window (hours)", 24, 336, 168)

    if st.button("🔍 Check for Drift", type="primary", use_container_width=True):
        with st.spinner("Analyzing drift..."):
            # Get drift summary
            drift_summary = check_drift(monitor_dataset)

            if drift_summary and monitor_dataset in drift_summary:
                summary = drift_summary[monitor_dataset]

                # Overall status
                if summary.get("drift_detected"):
                    st.error("⚠️ **DRIFT DETECTED** - Model may need retraining")
                else:
                    st.success("✅ **NO DRIFT DETECTED** - Model is stable")

                st.divider()

                # Feature Drift
                st.subheader("📊 Feature Drift Analysis")

                feature_drift = summary.get("feature_drift", {})

                if feature_drift.get("drift_detected"):
                    st.warning(
                        f"⚠️ Drift detected in "
                        f"{feature_drift.get('drift_count', 0)} "
                        f"out of {feature_drift.get('total_features', 0)} features"
                    )

                    # Show drifted features
                    if feature_drift.get("drifted_features"):
                        st.write("**Drifted Features:**")
                        for feat in feature_drift["drifted_features"]:
                            st.write(f"- {feat}")

                    # Feature drift details
                    if feature_drift.get("feature_drift_details"):
                        st.write("**Detailed Analysis:**")
                        drift_df = pd.DataFrame(
                            feature_drift["feature_drift_details"]
                        ).T
                        drift_df = drift_df.sort_values("ks_statistic", ascending=False)

                        # Format and display
                        st.dataframe(
                            drift_df.style.format(
                                {
                                    "ks_statistic": "{:.4f}",
                                    "p_value": "{:.4f}",
                                    "recent_mean": "{:.2f}",
                                    "baseline_mean": "{:.2f}",
                                    "mean_change_pct": "{:.1f}%",
                                }
                            ).background_gradient(subset=["ks_statistic"], cmap="Reds"),
                            use_container_width=True,
                        )

                        # Visualization
                        st.write("**KS Statistic by Feature:**")
                        fig = px.bar(
                            drift_df.reset_index(),
                            x="index",
                            y="ks_statistic",
                            color="drift_detected",
                            color_discrete_map={True: "#f44336", False: "#4caf50"},
                            labels={"index": "Feature", "ks_statistic": "KS Statistic"},
                        )
                        fig.add_hline(
                            y=0.05,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Drift Threshold",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No feature drift detected")

                st.divider()

                # Performance Drift
                st.subheader("📈 Performance Drift Analysis")

                perf_drift = summary.get("performance_drift", {})

                if perf_drift.get("drift_detected"):
                    st.warning("⚠️ Performance drift detected")

                    metrics = perf_drift.get("metrics", {})

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        conf_change = metrics.get("confidence_change_pct", 0)
                        st.metric(
                            "Confidence Change",
                            f"{conf_change:.1f}%",
                            delta=f"{conf_change:.1f}%",
                            delta_color="normal" if conf_change >= 0 else "inverse",
                        )

                    with col2:
                        mal_rate_change = metrics.get("malicious_rate_change_pct", 0)
                        st.metric(
                            "Malicious Rate Change",
                            f"{mal_rate_change:.1f}%",
                            delta=f"{mal_rate_change:.1f}%",
                        )

                    with col3:
                        recent_conf = metrics.get("recent_avg_confidence", 0)
                        st.metric("Recent Avg Confidence", f"{recent_conf:.1%}")

                    # Detailed metrics
                    st.write("**Detailed Metrics:**")
                    metrics_df = pd.DataFrame(
                        {
                            "Metric": [
                                "Avg Confidence",
                                "Malicious Rate",
                                "Low Confidence Rate",
                            ],
                            "Recent": [
                                f"{metrics.get('recent_avg_confidence', 0):.1%}",
                                f"{metrics.get('recent_malicious_rate', 0):.1%}",
                                f"{metrics.get('recent_low_confidence_rate', 0):.1%}",
                            ],
                            "Baseline": [
                                f"{metrics.get('baseline_avg_confidence', 0):.1%}",
                                f"{metrics.get('baseline_malicious_rate', 0):.1%}",
                                f"{metrics.get('baseline_low_confidence_rate', 0):.1%}",
                            ],
                        }
                    )
                    st.dataframe(metrics_df, use_container_width=True)
                else:
                    st.success("✅ No performance drift detected")

                st.divider()

                # Recommendations
                st.subheader("💡 Recommendations")

                recommendations = summary.get("recommendations", [])
                for rec in recommendations:
                    st.write(f"- {rec}")

                # Retrain decision
                st.divider()
                st.subheader("🔄 Retraining Decision")

                retrain_info = should_retrain(monitor_dataset)
                if retrain_info:
                    if retrain_info.get("should_retrain"):
                        st.error(
                            f"⚠️ **RETRAINING RECOMMENDED**: {retrain_info.get('reason')}"
                        )
                        st.button("🔄 Trigger Retraining", type="primary")
                    else:
                        st.success(f"✅ {retrain_info.get('reason')}")

            else:
                st.warning("No drift data available. Make some predictions first.")

    # Historical drift trends (if we have data)
    st.divider()
    st.subheader("📊 Drift History")
    st.info("Drift history tracking will be available after multiple drift checks")

# Footer
st.sidebar.divider()
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.caption("6G Smart City IDS v2.2.0")
