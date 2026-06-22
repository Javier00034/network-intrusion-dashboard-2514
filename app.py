import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_cleaning import load_and_clean_data
from src.train_models import train_and_compare_models


st.set_page_config(
    page_title="Network Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("Network Intrusion Detection Dashboard")
st.write(
    "This dashboard uses machine learning to analyse network traffic "
    "and detect possible intrusion or anomaly records."
)

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dataset Overview",
        "Model Explanation",
        "ML Analysis",
        "Suspicious Records",
        "Feature Importance"
    ]
)

st.sidebar.divider()

st.sidebar.header("Dataset Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload network traffic CSV dataset",
    type=["csv"]
)

use_default = st.sidebar.checkbox("Use local dataset from data/dataset.csv")

sample_enabled = st.sidebar.checkbox("Use dataset sampling", value=True)

sample_size = st.sidebar.number_input(
    "Sample size",
    min_value=1000,
    max_value=200000,
    value=20000,
    step=1000
)

if uploaded_file is not None:
    file_path = uploaded_file
elif use_default:
    file_path = "data/dataset.csv"
else:
    st.info("Please upload a CSV file or select the local dataset option.")
    st.stop()

try:
    df, X, y = load_and_clean_data(file_path)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# -----------------------------
# Dataset sampling
# -----------------------------
if sample_enabled and len(df) > sample_size:
    sampled_df = df.sample(n=sample_size, random_state=42)

    X = X.loc[sampled_df.index]
    y = y.loc[sampled_df.index]
    df = sampled_df

# -----------------------------
# Store ML results in session state
# -----------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None

if "rf_model" not in st.session_state:
    st.session_state.rf_model = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "model_predictions" not in st.session_state:
    st.session_state.model_predictions = None


# -----------------------------
# Page 1: Dataset Overview
# -----------------------------
if page == "Dataset Overview":
    st.header("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    total_records = len(df)
    normal_records = (df["Label_Binary"] == 0).sum()
    attack_records = (df["Label_Binary"] == 1).sum()

    col1.metric("Total Records Used", total_records)
    col2.metric("Normal Records", normal_records)
    col3.metric("Attack Records", attack_records)

    if sample_enabled:
        st.info(f"Dataset sampling is enabled. Current records used: {total_records}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

    st.subheader("Normal vs Attack Traffic")

    traffic_counts = df["Label_Binary"].map({
        0: "Normal",
        1: "Attack"
    }).value_counts().reset_index()

    traffic_counts.columns = ["Traffic Type", "Count"]

    fig = px.pie(
        traffic_counts,
        names="Traffic Type",
        values="Count",
        title="Normal vs Attack Traffic"
    )

    st.plotly_chart(fig, use_container_width=True)

    if "Label" in df.columns:
        st.subheader("Attack Type Breakdown")

        label_counts = df["Label"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]

        fig2 = px.bar(
            label_counts,
            x="Label",
            y="Count",
            title="Traffic Label Breakdown"
        )

        st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# Page 2: Model Explanation
# -----------------------------
elif page == "Model Explanation":
    st.header("2. Model Explanation")

    st.subheader("Random Forest")
    st.write(
        "Random Forest is a supervised classification model. "
        "It learns from labelled traffic records and predicts whether new traffic is normal or attack traffic. "
        "It is useful for this project because the dataset contains labels, and it can also show feature importance."
    )

    st.subheader("Isolation Forest")
    st.write(
        "Isolation Forest is an anomaly detection model. "
        "It identifies records that behave differently from most of the dataset. "
        "This is useful for detecting unusual traffic patterns that may indicate suspicious activity."
    )

    st.subheader("K-Means Clustering")
    st.write(
        "K-Means is an unsupervised clustering model. "
        "It groups similar traffic records together. "
        "In this project, clusters are mapped to normal or attack traffic based on the majority label inside each cluster."
    )

    st.subheader("Evaluation Metrics")
    st.write(
        "The models are compared using accuracy, precision, recall, F1-score, and processing time. "
        "For intrusion detection, recall is especially important because it shows how many actual attacks were detected."
    )

    metrics_table = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "Processing Time",
            "False Positive",
            "False Negative"
        ],
        "Meaning": [
            "Overall percentage of correct predictions.",
            "Out of predicted attacks, how many were actually attacks.",
            "Out of actual attacks, how many were detected.",
            "Balance between precision and recall.",
            "How long the model takes to train and predict.",
            "Normal traffic wrongly flagged as attack.",
            "Attack traffic wrongly missed as normal."
        ]
    })

    st.dataframe(metrics_table, use_container_width=True)


# -----------------------------
# Page 3: ML Analysis
# -----------------------------
elif page == "ML Analysis":
    st.header("3. ML Model Analysis")

    st.write(
        "Click the button below to train and compare Random Forest, Isolation Forest, and K-Means Clustering."
    )

    if st.button("Run ML Analysis"):
        with st.spinner("Training and evaluating models. Please wait..."):
            results_df, rf_model, scaler, X_test, y_test, model_predictions = train_and_compare_models(X, y)

        st.session_state.results_df = results_df
        st.session_state.rf_model = rf_model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.model_predictions = model_predictions

        st.success("ML analysis completed.")

    if st.session_state.results_df is not None:
        results_df = st.session_state.results_df

        st.subheader("Model Performance Results")
        st.dataframe(results_df, use_container_width=True)

        csv = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Model Results as CSV",
            data=csv,
            file_name="model_results.csv",
            mime="text/csv"
        )

        st.subheader("F1-Score Comparison")

        fig3 = px.bar(
            results_df,
            x="Model",
            y="F1-Score",
            title="F1-Score Comparison by Model"
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Processing Time Comparison")

        fig4 = px.bar(
            results_df,
            x="Model",
            y="Processing Time (s)",
            title="Processing Time by Model"
        )

        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Confusion Matrix Summary")

        selected_model = st.selectbox(
            "Select model for confusion matrix",
            results_df["Model"].tolist()
        )

        selected_row = results_df[results_df["Model"] == selected_model].iloc[0]

        confusion_data = pd.DataFrame({
            "Predicted Normal": [
                selected_row["True Negative"],
                selected_row["False Negative"]
            ],
            "Predicted Attack": [
                selected_row["False Positive"],
                selected_row["True Positive"]
            ]
        }, index=["Actual Normal", "Actual Attack"])

        st.dataframe(confusion_data, use_container_width=True)

        fig_cm = px.imshow(
            confusion_data,
            text_auto=True,
            title=f"Confusion Matrix - {selected_model}"
        )

        st.plotly_chart(fig_cm, use_container_width=True)

    else:
        st.info("Run ML analysis first to view model results.")


# -----------------------------
# Page 4: Suspicious Records
# -----------------------------
elif page == "Suspicious Records":
    st.header("4. Suspicious Records")

    if st.session_state.model_predictions is None:
        st.warning("Please run ML analysis first from the ML Analysis page.")
        st.stop()

    model_choice = st.selectbox(
        "Select model prediction to view",
        list(st.session_state.model_predictions.keys())
    )

    prediction_filter = st.selectbox(
        "Filter records",
        [
            "Predicted Attack",
            "Predicted Normal",
            "False Positive",
            "False Negative",
            "True Positive",
            "True Negative"
        ]
    )

    X_test = st.session_state.X_test.copy()
    y_test = st.session_state.y_test
    predictions = st.session_state.model_predictions[model_choice]

    records_df = X_test.copy()
    records_df["Actual Label"] = y_test.values
    records_df["Predicted Label"] = predictions

    if prediction_filter == "Predicted Attack":
        filtered_df = records_df[records_df["Predicted Label"] == 1]
    elif prediction_filter == "Predicted Normal":
        filtered_df = records_df[records_df["Predicted Label"] == 0]
    elif prediction_filter == "False Positive":
        filtered_df = records_df[
            (records_df["Actual Label"] == 0) &
            (records_df["Predicted Label"] == 1)
        ]
    elif prediction_filter == "False Negative":
        filtered_df = records_df[
            (records_df["Actual Label"] == 1) &
            (records_df["Predicted Label"] == 0)
        ]
    elif prediction_filter == "True Positive":
        filtered_df = records_df[
            (records_df["Actual Label"] == 1) &
            (records_df["Predicted Label"] == 1)
        ]
    else:
        filtered_df = records_df[
            (records_df["Actual Label"] == 0) &
            (records_df["Predicted Label"] == 0)
        ]

    st.write(f"Showing: **{prediction_filter}** records for **{model_choice}**")
    st.metric("Number of Records", len(filtered_df))

    st.dataframe(filtered_df.head(100), use_container_width=True)

    suspicious_csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Records as CSV",
        data=suspicious_csv,
        file_name="filtered_records.csv",
        mime="text/csv"
    )


# -----------------------------
# Page 5: Feature Importance
# -----------------------------
elif page == "Feature Importance":
    st.header("5. Random Forest Feature Importance")

    if st.session_state.rf_model is None:
        st.warning("Please run ML analysis first from the ML Analysis page.")
        st.stop()

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": st.session_state.rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(15)

    st.write(
        "Feature importance shows which network traffic features had the strongest influence "
        "on the Random Forest model's prediction."
    )

    st.dataframe(feature_importance, use_container_width=True)

    fig5 = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Important Features"
    )

    st.plotly_chart(fig5, use_container_width=True)