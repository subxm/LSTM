# IMDB Sentiment Analysis with LSTM

This project is a Streamlit app for training and using an LSTM-based sentiment classifier on the IMDB movie review dataset.

## What it does

- Loads and previews the IMDB review dataset
- Trains an LSTM model for binary sentiment classification
- Saves the trained model and preprocessing artifacts locally
- Evaluates the model on a held-out test split
- Predicts sentiment for custom movie reviews

## Requirements

- Python 3.11
- `IMDB Dataset.csv` in the project root

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Run the app

Start Streamlit from the project folder:

```bash
streamlit run app.py
```

If the browser does not open automatically, Streamlit will show a local URL in the terminal.

## How to use

1. Open the app in your browser.
2. Review the dataset preview and summary metrics.
3. Click `Train Model` to build and save the model artifacts.
4. Click `Run Evaluation` to view accuracy and the classification report.
5. Enter a review in the text box and click `Predict` to classify it as positive or negative.

## Saved artifacts

After training, the app writes these files to the project root:

- `lstm_sentiment.keras`
- `tokenizer.pkl`
- `label_encoder.pkl`

These files are reused for evaluation and prediction.

## Project structure

```text
app.py
IMDB Dataset.csv
README.md
```

## Notes

- The first training run can take a few minutes.
- If the saved model files are missing, the app will prompt you to train the model first.