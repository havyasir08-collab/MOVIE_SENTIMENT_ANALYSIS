
import tensorflow as tf
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Backend folder path
BASE_DIR = Path(__file__).resolve().parent

# Initialize FastAPI
app = FastAPI(
    title="IMDb Movie Sentiment Analysis API",
    description="Predict Positive or Negative movie reviews",
    version="1.0.0"
)

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load trained model
model = tf.keras.models.load_model(
    BASE_DIR / "models" / "sentiment_model.keras"
)

# Load tokenizer
with open(BASE_DIR / "models" / "tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

print("Sentiment model loaded!")
print("Tokenizer loaded!")

# Request format
class ReviewRequest(BaseModel):
    review: str

# Home route
@app.get("/")
def home():
    return {"message": "IMDb Sentiment Analysis API is running!"}

# Health check
@app.get("/health")
def health():
    return {"status": "healthy"}

# Prediction function
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=200
    )

    prediction = model.predict(padded_sequence, verbose=0)

    score = float(prediction[0][0])

    sentiment = "positive" if score > 0.5 else "negative"

    confidence = score if sentiment == "positive" else 1 - score

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }

# Prediction API
@app.post("/predict")
def predict(data: ReviewRequest):

    review = data.review.strip()

    if not review:
        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty"
        )

    result = predict_sentiment(review)

    return {
        "review": review,
        "sentiment": result["sentiment"],
        "confidence": round(result["confidence"], 4)
    }
