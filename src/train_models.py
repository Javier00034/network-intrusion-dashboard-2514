import time
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


def evaluate_model(name, y_true, y_pred, processing_time):
    """
    Calculates evaluation metrics for each model.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

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


def train_and_compare_models(X, y):
    """
    Trains and compares 3 ML models:
    1. Random Forest
    2. Isolation Forest
    3. K-Means Clustering
    """

    results = []
    model_predictions = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
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
        n_estimators=100,
        random_state=42,
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
        contamination=0.1,
        random_state=42
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
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    kmeans_model.fit(X_train_scaled)
    kmeans_pred_raw = kmeans_model.predict(X_test_scaled)

    train_clusters = kmeans_model.predict(X_train_scaled)

    mapping_df = pd.DataFrame({
        "Cluster": train_clusters,
        "Label": y_train.values
    })

    cluster_mapping = {}

    for cluster in mapping_df["Cluster"].unique():
        attack_ratio = mapping_df[mapping_df["Cluster"] == cluster]["Label"].mean()
        cluster_mapping[cluster] = 1 if attack_ratio >= 0.5 else 0

    kmeans_pred = [cluster_mapping[cluster] for cluster in kmeans_pred_raw]

    kmeans_time = time.time() - start_time

    results.append(evaluate_model("K-Means Clustering", y_test, kmeans_pred, kmeans_time))
    model_predictions["K-Means Clustering"] = kmeans_pred

    results_df = pd.DataFrame(results)

    return results_df, rf_model, scaler, X_test, y_test, model_predictions