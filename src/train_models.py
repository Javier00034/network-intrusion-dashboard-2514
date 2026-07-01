import time
from dataclasses import dataclass, field

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# -----------------------------
# Configuration constants
# -----------------------------
TEST_SIZE = 0.3
RANDOM_STATE = 42
RF_N_ESTIMATORS = 100
ISO_CONTAMINATION = 0.1
KMEANS_N_CLUSTERS = 2
KMEANS_N_INIT = 10


@dataclass
class TrainingResult:
    """Container for the outputs of ``train_and_compare_models``."""
    results_df: pd.DataFrame
    rf_model: RandomForestClassifier
    scaler: StandardScaler
    X_test: pd.DataFrame
    y_test: pd.Series
    predictions: dict
    feature_names: list = field(default_factory=list)


def evaluate_model(name, y_true, y_pred, processing_time):
    """
    Calculates evaluation metrics for each model.
    """

    # labels=[0, 1] forces a full 2x2 matrix so unpacking never fails,
    # even if a model predicts only a single class.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Processing Time (s)": round(processing_time, 4),
        "True Negative": int(tn),
        "False Positive": int(fp),
        "False Negative": int(fn),
        "True Positive": int(tp)
    }


def _build_cluster_mapping(mapping_df):
    """
    Maps K-Means clusters to binary labels (0 = normal, 1 = attack).

    The cluster with the higher attack ratio becomes "attack" (1) and the
    other becomes "normal" (0). This guarantees both clusters are never
    assigned the same label, which would make the predictions meaningless.
    """
    cluster_ratios = {
        cluster: mapping_df.loc[mapping_df["Cluster"] == cluster, "Label"].mean()
        for cluster in mapping_df["Cluster"].unique()
    }

    if len(cluster_ratios) < 2:
        # Degenerate case: only one cluster present in the training predictions.
        only_cluster = next(iter(cluster_ratios))
        return {only_cluster: 1 if cluster_ratios[only_cluster] >= 0.5 else 0}

    # Rank clusters by attack ratio; lowest = normal, highest = attack.
    ranked = sorted(cluster_ratios, key=cluster_ratios.get)
    return {ranked[0]: 0, ranked[1]: 1}


def train_and_compare_models(X, y):
    """
    Trains and compares 3 ML models:
    1. Random Forest
    2. Isolation Forest
    3. K-Means Clustering

    Returns
    -------
    TrainingResult
        A dataclass bundling the results dataframe, the trained Random Forest
        model, the fitted scaler, the held-out test set, per-model predictions,
        and the feature names used during training.
    """

    results = []
    model_predictions = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # Model 1: Random Forest
    # -----------------------------
    start_time = time.time()

    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_time = time.time() - start_time

    results.append(evaluate_model("Random Forest", y_test, rf_pred, rf_time))
    model_predictions["Random Forest"] = rf_pred

    # -----------------------------
    # Model 2: Isolation Forest
    # -----------------------------
    start_time = time.time()

    iso_model = IsolationForest(
        contamination=ISO_CONTAMINATION,
        random_state=RANDOM_STATE
    )

    iso_model.fit(X_train_scaled)
    iso_pred_raw = iso_model.predict(X_test_scaled)

    # Isolation Forest:
    # 1 = normal, -1 = anomaly
    iso_pred = [0 if pred == 1 else 1 for pred in iso_pred_raw]

    iso_time = time.time() - start_time

    results.append(evaluate_model("Isolation Forest", y_test, iso_pred, iso_time))
    model_predictions["Isolation Forest"] = iso_pred

    # -----------------------------
    # Model 3: K-Means Clustering
    # -----------------------------
    start_time = time.time()

    kmeans_model = KMeans(
        n_clusters=KMEANS_N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=KMEANS_N_INIT
    )

    kmeans_model.fit(X_train_scaled)
    kmeans_pred_raw = kmeans_model.predict(X_test_scaled)

    train_clusters = kmeans_model.predict(X_train_scaled)

    mapping_df = pd.DataFrame({
        "Cluster": train_clusters,
        "Label": y_train.values
    })

    cluster_mapping = _build_cluster_mapping(mapping_df)

    # Default unseen clusters to 0 (normal) as a safety fallback.
    kmeans_pred = [cluster_mapping.get(cluster, 0) for cluster in kmeans_pred_raw]

    kmeans_time = time.time() - start_time

    results.append(evaluate_model("K-Means Clustering", y_test, kmeans_pred, kmeans_time))
    model_predictions["K-Means Clustering"] = kmeans_pred

    results_df = pd.DataFrame(results)

    return TrainingResult(
        results_df=results_df,
        rf_model=rf_model,
        scaler=scaler,
        X_test=X_test,
        y_test=y_test,
        predictions=model_predictions,
        feature_names=list(X_train.columns)
    )
