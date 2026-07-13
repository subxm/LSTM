import logging
import os
import pickle
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pandas as pd
import streamlit as st

DATASET = Path("IMDB Dataset.csv")
MODEL = Path("lstm_sentiment.keras")
TOKENIZER = Path("tokenizer.pkl")

MAX_WORDS = 10000
MAX_LENGTH = 200
EMBED_DIM = 128
LSTM_UNITS = 128
EPOCHS = 5
BATCH_SIZE = 64


def artifacts_ready() -> bool:
    return MODEL.exists() and TOKENIZER.exists()


@st.cache_data(show_spinner=False)
def load_data(dataset_mtime: float) -> pd.DataFrame:
    return pd.read_csv(DATASET)


def split_data(features, targets, test_size: float = 0.2, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(features))
    split_index = int(len(features) * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return (
        features[train_indices],
        features[test_indices],
        targets[train_indices],
        targets[test_indices],
    )


def build_classification_report(y_true, y_pred) -> str:
    lines = ["              precision    recall  f1-score   support", ""]

    total_correct = int((y_true == y_pred).sum())
    accuracy = total_correct / len(y_true) if len(y_true) else 0.0

    for label, name in ((0, "negative"), (1, "positive")):
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        false_positive = int(((y_true != label) & (y_pred == label)).sum())
        false_negative = int(((y_true == label) & (y_pred != label)).sum())
        support = int((y_true == label).sum())

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        lines.append(
            f"{name:>12} {precision:>10.2f} {recall:>9.2f} {f1_score:>9.2f} {support:>9}"
        )

    lines.extend(
        [
            "",
            f"{'accuracy':>12} {'':>10} {'':>9} {accuracy:>9.2f} {len(y_true):>9}",
        ]
    )
    return "\n".join(lines)


@st.cache_resource(show_spinner=False)
def prepare_training_data(dataset_mtime: float):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    df = load_data(dataset_mtime).copy()
    df["sentiment"] = df["sentiment"].str.lower().map({"negative": 0, "positive": 1})
    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["review"])

    features = pad_sequences(
        tokenizer.texts_to_sequences(df["review"]),
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )
    targets = df["sentiment"].values

    X_train, X_test, y_train, y_test = split_data(
        features,
        targets,
        test_size=0.2,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test, tokenizer


def build_model():
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Embedding(MAX_WORDS, EMBED_DIM),
            LSTM(LSTM_UNITS),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    return model, early_stopping


@st.cache_resource(show_spinner=False)
def load_saved_artifacts(
    model_mtime: float,
    tokenizer_mtime: float,
):
    from tensorflow.keras.models import load_model

    model = load_model(MODEL)

    with TOKENIZER.open("rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


def evaluate_model(model, X_test, y_test):
    predictions = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
    accuracy = float((predictions == y_test).mean())
    report = build_classification_report(y_test, predictions)
    return accuracy, report


st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")
st.title("IMDB Movie Review Sentiment Analysis with LSTM")

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
        X_train, X_test, y_train, y_test, tokenizer = prepare_training_data(dataset_mtime)
        model, early_stopping = build_model()

        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=1,
        )

        model.save(MODEL)

        with TOKENIZER.open("wb") as file:
            pickle.dump(tokenizer, file)

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
                TOKENIZER.stat().st_mtime,
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
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            model, tokenizer = load_saved_artifacts(
                MODEL.stat().st_mtime,
                TOKENIZER.stat().st_mtime,
            )
            sequence = tokenizer.texts_to_sequences([review])
            padded_review = pad_sequences(
                sequence,
                maxlen=MAX_LENGTH,
                padding="post",
                truncating="post",
            )
            score = float(model.predict(padded_review, verbose=0)[0][0])
            label_index = int(score >= 0.5)
            label = "positive" if label_index == 1 else "negative"

        if label == "positive":
            st.success(f"Positive review ({score:.2%} confidence)")
        else:
            st.error(f"Negative review ({1 - score:.2%} confidence)")

if not model_available:
    st.caption("Prediction becomes available after the first training run.")