# ─────────────────────────────────────────────────────
# src/predict.py
#
# Loads the trained model and exposes a prediction
# function that can be used by the FastAPI app
# and by tests.
# ─────────────────────────────────────────────────────

import joblib
import os
from src.preprocess import clean_text

# ── Constants ──────────────────────────────────────────
MODEL_PATH = 'models/spam_classifier.pkl'


# ── Load model ──────────────────────────────────────────
def load_model():
    """
    Loads the trained model from disk.
    Raises a clear error if model file doesn't exist.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Please run training first: python -m src.train"
        )
    return joblib.load(MODEL_PATH)


# ── Load model once at module level ────────────────────
# We load the model once when this file is imported
# not every time a prediction is made
# Loading a model takes time — we don't want that
# happening on every single API request
model = load_model()


# ── Prediction function ─────────────────────────────────
def predict(email_text: str) -> dict:
    """
    Takes raw email text and returns prediction.

    Steps:
    1. Clean the text using same steps as preprocessing
    2. Pass to model for prediction
    3. Get probability score
    4. Return human readable result

    Returns dict with:
    - prediction: 'spam' or 'ham'
    - confidence: probability score 0 to 1
    - is_spam: boolean True or False
    """

    # Step 1 — clean the text
    # Must use the exact same cleaning as training
    # If training saw stemmed lowercase text the model
    # expects stemmed lowercase text at prediction time too
    cleaned = clean_text(email_text)

    # Step 2 — get prediction
    # model.predict returns [0] or [1]
    # [0] means the first element of the array
    prediction = model.predict([cleaned])[0]

    # Step 3 — get probability score
    # predict_proba returns probabilities for each class
    # [[prob_ham, prob_spam]]
    # [0][1] means first row, second column = spam probability
    probabilities = model.predict_proba([cleaned])[0]
    confidence = round(float(probabilities[1]), 4)

    # Step 4 — return human readable result
    return {
        'prediction': 'spam' if prediction == 1 else 'ham',
        'confidence': confidence,
        'is_spam': bool(prediction == 1)
    }

# ─────────────────────────────────────────────────────
# main.py
#
# FastAPI application.
# Exposes the spam classifier as a REST API.
# Run with: uvicorn main:app --reload
# ─────────────────────────────────────────────────────

