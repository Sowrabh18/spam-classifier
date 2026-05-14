# ─────────────────────────────────────────────────────
# Deployed via CI/CD pipeline
# main.py
#
# FastAPI application.
# Exposes the spam classifier as a REST API.
# Run with: uvicorn main:app --reload

# ─────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict

# ── Create FastAPI app ──────────────────────────────────
# This is the main application object
# Every endpoint gets attached to this
app = FastAPI(
    title="Spam Classifier API",
    description="Classifies emails as spam or ham using Naive Bayes",
    version="1.0.0"
)


# ── Request model ───────────────────────────────────────
# Pydantic BaseModel defines the shape of incoming data
# FastAPI uses this to automatically validate requests
# If someone sends a request without 'email' field
# FastAPI returns a clear error automatically
class EmailRequest(BaseModel):
    email: str

    class Config:
        # This shows an example in the auto-generated docs
        json_schema_extra = {
            "example": {
                "email": "Congratulations! You have won a FREE prize. Click now!"
            }
        }


# ── Response model ──────────────────────────────────────
# Defines the shape of what we send back
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    is_spam: bool


# ── Home endpoint ───────────────────────────────────────
# GET / → just confirms the API is running
@app.get("/")
def home():
    """
    Health check endpoint.
    Returns a simple message confirming API is running.
    """
    return {
        "message": "Spam Classifier API is running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }


# ── Predict endpoint ────────────────────────────────────
# POST /predict → accepts email text, returns prediction
@app.post("/predict", response_model=PredictionResponse)
def predict_spam(request: EmailRequest):
    """
    Classifies an email as spam or ham.

    Send a POST request with JSON body:
    {"email": "your email text here"}

    Returns:
    - prediction: 'spam' or 'ham'
    - confidence: probability score 0 to 1
    - is_spam: boolean
    """

    # Validate that email is not empty
    if not request.email.strip():
        raise HTTPException(
            status_code=400,
            detail="Email text cannot be empty"
        )

    # Get prediction from our predict function
    result = predict(request.email)

    return result