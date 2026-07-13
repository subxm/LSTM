import csv
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import streamlit as st

DATASET = Path("IMDB Dataset.csv")
MODEL = Path("sentiment_model.pkl")

TRAIN_RATIO = 0.8
ALPHA = 1.0
MAX_PREVIEW_ROWS = 5
TOKEN_PATTERN = re.compile(r"[a-z']+")


def artifacts_ready() -> bool:
    return MODEL.exists()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@st.cache_data(show_spinner=False)
def read_dataset_preview(dataset_mtime: float):
    preview = []
    total_rows = 0
    class_counts = Counter()

    with DATASET.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_rows += 1
            sentiment = (row.get("sentiment") or "").strip().lower()
            if sentiment:
                class_counts[sentiment] += 1
            if len(preview) < MAX_PREVIEW_ROWS:
                preview.append({"review": row.get("review", ""), "sentiment": row.get("sentiment", "")})

    return preview, total_rows, class_counts


def load_records(dataset_path: Path):
    records = []
    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            review = (row.get("review") or "").strip()
            sentiment = (row.get("sentiment") or "").strip().lower()
            if review and sentiment in {"positive", "negative"}:
                records.append((review, sentiment))
    return records


def split_data(records, train_ratio: float = TRAIN_RATIO, seed: int = 42):
    grouped = defaultdict(list)
    for review, sentiment in records:
        grouped[sentiment].append((review, sentiment))

    rng = random.Random(seed)
    train_records = []
    test_records = []

    for sentiment, items in grouped.items():
        rng.shuffle(items)
        if len(items) <= 1:
            split_index = len(items)
        else:
            split_index = max(1, int(len(items) * train_ratio))

        train_records.extend(items[:split_index])
        test_records.extend(items[split_index:])

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    return train_records, test_records


def build_model(train_records):
    class_doc_counts = Counter()
    token_counts = {"negative": Counter(), "positive": Counter()}
    class_token_totals = Counter()
    vocabulary = set()

    for review, sentiment in train_records:
        class_doc_counts[sentiment] += 1
        tokens = tokenize(review)
        token_counts[sentiment].update(tokens)
        class_token_totals[sentiment] += len(tokens)
        vocabulary.update(tokens)

    return {
        "alpha": ALPHA,
        "class_doc_counts": dict(class_doc_counts),
        "class_token_totals": dict(class_token_totals),
        "token_counts": {label: dict(counts) for label, counts in token_counts.items()},
        "vocabulary_size": len(vocabulary),
    }


def log_probabilities(model, review: str):
    tokens = tokenize(review)
    vocabulary_size = max(model["vocabulary_size"], 1)
    total_docs = sum(model["class_doc_counts"].values())
    scores = {}

    for label in ("negative", "positive"):
        class_docs = model["class_doc_counts"].get(label, 0)
        if class_docs == 0 or total_docs == 0:
            scores[label] = float("-inf")
            continue

        log_score = math.log(class_docs / total_docs)
        token_total = model["class_token_totals"].get(label, 0)
        counts = model["token_counts"].get(label, {})

        for token in tokens:
            token_count = counts.get(token, 0)
            likelihood = (token_count + model["alpha"]) / (token_total + model["alpha"] * vocabulary_size)
            log_score += math.log(likelihood)

        scores[label] = log_score

    return scores


def predict_label(model, review: str):
    scores = log_probabilities(model, review)
    positive_score = scores.get("positive", float("-inf"))
    negative_score = scores.get("negative", float("-inf"))

    if positive_score >= negative_score:
        label = "positive"
    else:
        label = "negative"

    max_score = max(scores.values())
    exp_positive = math.exp(positive_score - max_score) if positive_score > float("-inf") else 0.0
    exp_negative = math.exp(negative_score - max_score) if negative_score > float("-inf") else 0.0
    total = exp_positive + exp_negative
    positive_probability = exp_positive / total if total else 0.5

    return label, positive_probability


def evaluate_model(model, records):
    labels = []
    predictions = []

    for review, sentiment in records:
        predicted, _ = predict_label(model, review)
        labels.append(sentiment)
        predictions.append(predicted)

    total = len(labels)
    correct = sum(1 for actual, predicted in zip(labels, predictions) if actual == predicted)
    accuracy = correct / total if total else 0.0

    lines = ["              precision    recall  f1-score   support", ""]
    for label in ("negative", "positive"):
        tp = sum(1 for actual, predicted in zip(labels, predictions) if actual == label and predicted == label)
        fp = sum(1 for actual, predicted in zip(labels, predictions) if actual != label and predicted == label)
        fn = sum(1 for actual, predicted in zip(labels, predictions) if actual == label and predicted != label)
        support = sum(1 for actual in labels if actual == label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        lines.append(f"{label:>12} {precision:>10.2f} {recall:>9.2f} {f1_score:>9.2f} {support:>9}")

    lines.extend([
        "",
        f"{'accuracy':>12} {'':>10} {'':>9} {accuracy:>9.2f} {total:>9}",
    ])
    return accuracy, "\n".join(lines)


@st.cache_resource(show_spinner=False)
def load_saved_artifacts(model_mtime: float):
    with MODEL.open("rb") as file:
        return pickle.load(file)


st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")
st.title("IMDB Movie Review Sentiment Analysis")

if not DATASET.exists():
    st.error("Place 'IMDB Dataset.csv' in the same folder as app.py.")
    st.stop()

dataset_mtime = DATASET.stat().st_mtime
preview_rows, review_count, class_counts = read_dataset_preview(dataset_mtime)

col1, col2 = st.columns(2)
col1.metric("Reviews", f"{review_count:,}")
col2.metric("Classes", "2")

st.subheader("Dataset Preview")
st.table(preview_rows)

model_available = artifacts_ready()

if model_available:
    st.success("Saved model file found. You can evaluate the model or make predictions.")
else:
    st.info("No saved model found yet. Training starts only when you click the button.")

if st.button("Train Model", type="primary"):
    with st.spinner("Preparing data and training the model. This may take a few minutes the first time."):
        records = load_records(DATASET)
        train_records, test_records = split_data(records)
        model = build_model(train_records)

        with MODEL.open("wb") as file:
            pickle.dump(model, file)

        accuracy, report = evaluate_model(model, test_records)
        load_saved_artifacts.clear()

    st.session_state["metrics"] = {
        "accuracy": accuracy,
        "report": report,
    }
    model_available = True
    st.success("Training complete. Model file was saved successfully.")

st.subheader("Model Evaluation")

if "metrics" in st.session_state:
    st.write("Accuracy:", round(st.session_state["metrics"]["accuracy"], 4))
    st.text(st.session_state["metrics"]["report"])
elif model_available:
    if st.button("Run Evaluation"):
        with st.spinner("Loading the saved model and evaluating it..."):
            records = load_records(DATASET)
            train_records, test_records = split_data(records)
            model = load_saved_artifacts(MODEL.stat().st_mtime)
            accuracy, report = evaluate_model(model, test_records)

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
            model = load_saved_artifacts(MODEL.stat().st_mtime)
            label, score = predict_label(model, review)

        if label == "positive":
            st.success(f"Positive review ({score:.2%} confidence)")
        else:
            st.error(f"Negative review ({1 - score:.2%} confidence)")

if not model_available:
    st.caption("Prediction becomes available after the first training run.")