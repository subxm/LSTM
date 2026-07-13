import logging
import os
import pickle
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET = Path("IMDB Dataset.csv")
MODEL = Path("lstm_sentiment.keras")
TOKENIZER = Path("tokenizer.pkl")
ENCODER = Path("label_encoder.pkl")

MAX_WORDS = 10000
MAX_LENGTH = 200
EMBED_DIM = 128
LSTM_UNITS = 128
EPOCHS = 5
BATCH_SIZE = 64


def artifacts_ready() -> bool:
    return MODEL.exists() and TOKENIZER.exists() and ENCODER.exists()


@st.cache_data(show_spinner=False)
def load_data(dataset_mtime: float) -> pd.DataFrame:
    return pd.read_csv(DATASET)


@st.cache_resource(show_spinner=False)
def prepare_training_data(dataset_mtime: float):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    df = load_data(dataset_mtime).copy()
    encoder = LabelEncoder()
    df["sentiment"] = encoder.fit_transform(df["sentiment"])

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["review"])

    features = pad_sequences(
        tokenizer.texts_to_sequences(df["review"]),
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )
    targets = df["sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test, tokenizer, encoder


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
    encoder_mtime: float,
):
    from tensorflow.keras.models import load_model

    model = load_model(MODEL)

    with TOKENIZER.open("rb") as file:
        tokenizer = pickle.load(file)

    with ENCODER.open("rb") as file:
        encoder = pickle.load(file)

    return model, tokenizer, encoder


def evaluate_model(model, X_test, y_test):
    predictions = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
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
        X_train, X_test, y_train, y_test, tokenizer, encoder = prepare_training_data(dataset_mtime)
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

        with ENCODER.open("wb") as file:
            pickle.dump(encoder, file)

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
            X_train, X_test, y_train, y_test, _, _ = prepare_training_data(dataset_mtime)
            model, _, _ = load_saved_artifacts(
                MODEL.stat().st_mtime,
                TOKENIZER.stat().st_mtime,
                ENCODER.stat().st_mtime,
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

            model, tokenizer, encoder = load_saved_artifacts(
                MODEL.stat().st_mtime,
                TOKENIZER.stat().st_mtime,
                ENCODER.stat().st_mtime,
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
            label = encoder.inverse_transform([label_index])[0]

        if label == "positive":
            st.success(f"Positive review ({score:.2%} confidence)")
        else:
            st.error(f"Negative review ({1 - score:.2%} confidence)")

if not model_available:
    st.caption("Prediction becomes available after the first training run.")