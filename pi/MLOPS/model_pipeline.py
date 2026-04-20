from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    RobustScaler,
)

from imblearn.over_sampling import SMOTE

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "Data5G"


# -----------------------------
# Dataset aliases / mapping
# -----------------------------
DATASET_FILES = {
    "mMTC": "mMTC.csv",
    "URLLC": "URLLC.csv",
    "eMBB": "eMBB.csv",
    "TON_IoT": "train_test_network.csv",
    "train_test_network": "train_test_network.csv",
}


# -----------------------------
# Exact notebook feature sets
# -----------------------------
embb_features = [
    "Dur",
    "TotPkts",
    "TotBytes",
    "Rate",
    "Load",
    "Loss",
    "pLoss",
    "TcpRtt",
]
mmtc_features = ["TotPkts", "Rate", "SrcGap", "DstGap", "Dur", "Load", "Loss", "TcpRtt"]
urllc_features = [
    "TcpRtt",
    "SynAck",
    "AckDat",
    "Loss",
    "Dur",
    "Rate",
    "TotPkts",
    "TotBytes",
]
toniot_features = [
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "duration",
    "proto",
    "conn_state",
    "service",
]

FEATURE_MAP = {
    "mMTC": mmtc_features,
    "URLLC": urllc_features,
    "eMBB": embb_features,
    "TON_IoT": toniot_features,
}

SMOTE_THRESHOLD = 2.0
log1p_transformer = FunctionTransformer(np.log1p, validate=False)


# -----------------------------
# Helpers
# -----------------------------
def normalize_dataset_name(name: str) -> str:
    if name == "train_test_network":
        return "TON_IoT"
    return name


def list_dataset_files():
    out = {}
    for k in ["mMTC", "URLLC", "eMBB", "TON_IoT"]:
        out[k] = str(DATA_DIR / DATASET_FILES[k])
    return out


def model_path_for(dataset_name: str) -> Path:
    return BASE_DIR / f"lightgbm_{dataset_name}.joblib"


def load_dataset(dataset_name: str):
    dataset_name = normalize_dataset_name(dataset_name)
    filename = DATA_DIR / DATASET_FILES[dataset_name]

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()

    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    print(f"{dataset_name}: detected separator={repr(sep)}")

    df = pd.read_csv(filename, sep=sep, low_memory=False)

    if dataset_name == "TON_IoT":
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df.columns = df.columns.str.strip()

        if "Label" in df.columns:
            df = df.rename(columns={"Label": "label_raw"})

        if "label_raw" not in df.columns:
            raise ValueError("TON_IoT dataset must contain 'Label' or 'label_raw'")

        df = df[pd.to_numeric(df["label_raw"], errors="coerce").notna()]
        df["label_raw"] = df["label_raw"].astype(float)
        df["Label"] = df["label_raw"].astype(int).map({0: "Benign", 1: "Malicious"})
    else:
        df.columns = df.columns.str.strip()

    print(f"Loaded {dataset_name}: {df.shape}")
    return dataset_name, df


def make_xy(df, label_col="Label"):
    drop_cols = [
        label_col,
        "label_raw",
        "predicted",
        "UniqueID",
        "X",
        "anomaly_type",
        "type",
        "SrcAddr",
        "DstAddr",
        "Sport",
        "Dport",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "source_ip",
        "destination_ip",
        "timestamp",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[label_col]
    return X, y


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", log1p_transformer),
            ("scaler", RobustScaler()),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )


def prepare_data(dataset_name=None):
    if not dataset_name:
        raise ValueError("dataset_name is required")

    dataset_name, df = load_dataset(dataset_name)

    X, y = make_xy(df)
    keep = [c for c in FEATURE_MAP[dataset_name] if c in X.columns]
    X = X[keep]

    print(f"{dataset_name} -> using {len(keep)} features, {len(X)} rows")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    pre = build_preprocessor(X_train)
    X_train_proc = pre.fit_transform(X_train)
    X_test_proc = pre.transform(X_test)

    # Convert to numpy arrays to avoid feature name warnings
    if hasattr(X_train_proc, "values"):
        X_train_proc = X_train_proc.values
    if hasattr(X_test_proc, "values"):
        X_test_proc = X_test_proc.values

    counts = pd.Series(y_train).value_counts()
    ratio = counts.max() / counts.min()

    return {
        "dataset_name": dataset_name,
        "df": df,
        "X": X,
        "y": y,
        "label_encoder": le,
        "features": keep,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": pre,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "imbalance_ratio": ratio,
    }


def train_model(dataset_name=None):
    info = prepare_data(dataset_name)

    dataset_name = info["dataset_name"]
    le = info["label_encoder"]
    X_train_proc = info["X_train_proc"]
    X_test_proc = info["X_test_proc"]
    y_train = info["y_train"]
    y_test = info["y_test"]
    ratio = info["imbalance_ratio"]

    print("\n==============================")
    print(f"Dataset: {dataset_name}")
    print("Model  : LightGBM")
    print("==============================")

    if dataset_name != "TON_IoT" and ratio > SMOTE_THRESHOLD:
        print(f"Applying SMOTE (ratio={ratio:.2f}:1)...")
        sm = SMOTE(sampling_strategy=0.5, random_state=42)
        X_train_proc, y_train = sm.fit_resample(X_train_proc, y_train)
        scale_pos = 1.0
        used_smote = True
    else:
        print(f"No SMOTE applied (ratio={ratio:.2f}:1)")
        scale_pos = ratio
        used_smote = False

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True,  # Suppress warnings
    )

    # Try MLflow tracking, but continue if it fails (e.g., permission issues)
    mlflow_enabled = True
    mlflow_run = None

    try:
        # Use SQLite database backend (recommended over file store)
        import os

        db_path = os.path.abspath("mlflow.db")
        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("6G_IDS_LightGBM")
        mlflow_run = mlflow.start_run(run_name=f"LightGBM_{dataset_name}")
    except Exception as e:
        print(f"Warning: MLflow tracking disabled due to error: {str(e)[:100]}")
        mlflow_enabled = False
        mlflow_run = None

    try:
        if mlflow_enabled and mlflow_run:
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("model", "LightGBM")
            mlflow.log_param("n_estimators", 300)
            mlflow.log_param("learning_rate", 0.05)
            mlflow.log_param("num_leaves", 63)
            mlflow.log_param("max_depth", -1)
            mlflow.log_param("min_child_samples", 20)
            mlflow.log_param("scale_pos_weight", scale_pos)
            mlflow.log_param("feature_count", len(info["features"]))
            mlflow.log_param("smote_threshold", SMOTE_THRESHOLD)
            mlflow.log_param("used_smote", used_smote)

        model.fit(
            X_train_proc,
            y_train,
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
            eval_set=[(X_test_proc, y_test)],
        )

        train_pred = model.predict(X_train_proc)
        test_pred = model.predict(X_test_proc)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        print("Train Accuracy:", train_acc)
        print("Test  Accuracy:", test_acc)

        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        auc = roc_auc_score(y_test, y_proba)

        print(f"Accuracy : {acc:.4f}")
        print(f"F1 macro : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        if mlflow_enabled and mlflow_run:
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("roc_auc", auc)
            # Use 'name' parameter instead of deprecated 'artifact_path'
            try:
                mlflow.sklearn.log_model(model, name="lightgbm_model")
            except TypeError:
                # Fallback for older MLflow versions
                mlflow.sklearn.log_model(model, artifact_path="lightgbm_model")
    finally:
        if mlflow_enabled and mlflow_run:
            mlflow.end_run()

        bundle = {
            "dataset_name": dataset_name,
            "model": model,
            "preprocessor": info["preprocessor"],
            "label_encoder": le,
            "features": info["features"],
        }
        joblib.dump(bundle, model_path_for(dataset_name))
        print(f"[OK] Model saved: {model_path_for(dataset_name)}")

    return {
        "dataset_name": dataset_name,
        "accuracy": acc,
        "f1_macro": f1,
        "roc_auc": auc,
    }


def evaluate_model(dataset_name=None):
    if not dataset_name:
        raise ValueError("dataset_name is required")

    dataset_name = normalize_dataset_name(dataset_name)
    model_path = model_path_for(dataset_name)

    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    pre = bundle["preprocessor"]
    le = bundle["label_encoder"]
    features = bundle["features"]

    dataset_name, df = load_dataset(dataset_name)
    X, y = make_xy(df)
    X = X[[c for c in features if c in X.columns]]

    y_enc = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    X_test_proc = pre.transform(X_test)

    # Convert to numpy arrays to avoid feature name warnings
    if hasattr(X_test_proc, "values"):
        X_test_proc = X_test_proc.values

    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_proba)

    print("[OK] Evaluation done.")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 macro : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return {
        "dataset_name": dataset_name,
        "accuracy": acc,
        "f1_macro": f1,
        "roc_auc": auc,
    }
