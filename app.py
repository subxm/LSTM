import logging
import os
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET = Path("IMDB Dataset.csv")
MODEL = Path("sentiment_model.pkl")
VECTORIZER = Path("vectorizer.pkl")

MAX_FEATURES = 5000


def artifacts_ready() -> bool:
    return MODEL.exists() and VECTORIZER.exists()


@st.cache_data(show_spinner=False)
def load_data(dataset_mtime: float) -> pd.DataFrame:
    return pd.read_csv(DATASET)


def prepare_training_data(dataset_mtime: float):
    df = load_data(dataset_mtime).copy()
    df["sentiment"] = df["sentiment"].str.lower().map({"negative": 0, "positive": 1})
    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return X_train_vectorized, X_test_vectorized, y_train.to_numpy(), y_test.to_numpy(), vectorizer


def build_model():
    return LogisticRegression(max_iter=1000)


@st.cache_resource(show_spinner=False)
def load_saved_artifacts(
    model_mtime: float,
    vectorizer_mtime: float,
):
    with MODEL.open("rb") as file:
        model = pickle.load(file)

    with VECTORIZER.open("rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["negative", "positive"])
    return accuracy, report


st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")
st.title("IMDB Movie Review Sentiment Analysis")

if not DATASET.exists():
    st.error("Place 'IMDB Dataset.csv' in the same folder as app.py.")
    st.stop()

dataset_mtime = DATASET.stat().st_mtime
df = load_data(dataset_mtime)

col1, col2 = st.columns(2)
col1.metric("Reviews", f"{len(df):,}")
col2.metric("Classes", str(df["sentiment"].nunique()))

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

model_available = artifacts_ready()

if model_available:
    st.success("Saved model files found. You can evaluate the model or make predictions.")
else:
    st.info("No saved model found yet. Training starts only when you click the button.")

if st.button("Train Model", type="primary"):
    with st.spinner("Preparing data and training the model. This may take a few minutes the first time."):
        X_train, X_test, y_train, y_test, vectorizer = prepare_training_data(dataset_mtime)
        model = build_model()

        model.fit(X_train, y_train)

        with MODEL.open("wb") as file:
            pickle.dump(model, file)

        with VECTORIZER.open("wb") as file:
            pickle.dump(vectorizer, file)

        accuracy, report = evaluate_model(model, X_test, y_test)
        load_saved_artifacts.clear()

    st.session_state["metrics"] = {
        "accuracy": accuracy,
        "report": report,
    }
    model_available = True
    st.success("Training complete. Model files were saved successfully.")

st.subheader("Model Evaluation")

if "metrics" in st.session_state:
    st.write("Accuracy:", round(st.session_state["metrics"]["accuracy"], 4))
    st.text(st.session_state["metrics"]["report"])
elif model_available:
    if st.button("Run Evaluation"):
        with st.spinner("Loading the saved model and evaluating it..."):
            X_train, X_test, y_train, y_test, _ = prepare_training_data(dataset_mtime)
            model, _ = load_saved_artifacts(
                MODEL.stat().st_mtime,
                VECTORIZER.stat().st_mtime,
            )
            accuracy, report = evaluate_model(model, X_test, y_test)

        st.session_state["metrics"] = {
            "accuracy": accuracy,
            "report": report,
        }
        st.rerun()
else:
    st.caption("Train the model once to see evaluation results here.")

st.divider()
st.subheader("Predict Review Sentiment")

review = st.text_area("Enter movie review")

if st.button("Predict", disabled=not model_available):
    if not review.strip():
        st.warning("Enter a review before predicting.")
    else:
        with st.spinner("Loading the saved model and generating a prediction..."):
            model, vectorizer = load_saved_artifacts(
                MODEL.stat().st_mtime,
                VECTORIZER.stat().st_mtime,
            )
            review_vector = vectorizer.transform([review])
            score = float(model.predict_proba(review_vector)[0][1])
            label_index = int(score >= 0.5)
            label = "positive" if label_index == 1 else "negative"

        if label == "positive":
            st.success(f"Positive review ({score:.2%} confidence)")
        else:
            st.error(f"Negative review ({1 - score:.2%} confidence)")

if not model_available:
    st.caption("Prediction becomes available after the first training run.")