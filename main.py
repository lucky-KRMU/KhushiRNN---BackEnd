"""
KhushiRNN Sentiment Analysis API.

FastAPI application optimized for deployment on Render, container platforms,
and local development.
"""

from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import os
import sys
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure current directory is in sys.path so model can be imported regardless of execution working directory
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from model import SentimentRNN
except ImportError:
    from Backend.model import SentimentRNN

# Global model container
model_instance: Optional[SentimentRNN] = None
start_time: float = time.time()


def get_model() -> SentimentRNN:
    """
    Returns the loaded model instance, initializing it lazily if not already cached.
    """
    global model_instance
    if model_instance is None:
        model_instance = SentimentRNN()
    return model_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle event handler: loads model once at startup into memory
    to ensure ultra-fast request inference and avoid cold-load per request.
    """
    global start_time
    start_time = time.time()
    try:
        get_model()
        print("✅ KhushiRNN model weights and vocabulary loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise e
    yield
    print("Shutting down KhushiRNN API service.")


app = FastAPI(
    title="KhushiRNN Sentiment Analysis API",
    description="High-performance, lightweight RNN backend for classifying reviews as Positive or Negative.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration (Vite, React, Next.js, Vue, vanilla JS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for easy frontend connectivity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        description="The text review to analyze.",
        examples=["This movie was absolutely brilliant! Captivating from start to finish."],
    )
    threshold: Optional[float] = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Threshold score above which sentiment is classified as Positive.",
    )


class BatchReviewRequest(BaseModel):
    reviews: List[str] = Field(
        ...,
        min_length=1,
        description="List of reviews to analyze.",
    )
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    review: str
    sentiment: str
    score: float
    confidence: float
    token_count: int
    total_tokens: int
    truncated: bool


class BatchPredictionResponse(BaseModel):
    total_count: int
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    uptime_seconds: float


@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint providing service metadata.
    """
    return {
        "message": "Welcome to KhushiRNN Sentiment Analysis API",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /batch-predict",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """
    Health check endpoint for Render zero-downtime deployments and uptime monitors.
    """
    try:
        get_model()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not initialized: {e}",
        )
    return HealthResponse(
        status="healthy",
        model="KhushiRNN",
        version="1.0.0",
        uptime_seconds=round(time.time() - start_time, 2),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_sentiment(payload: ReviewRequest):
    """
    Classify a single review as Positive or Negative.
    """
    if not payload.review or not payload.review.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty or solely whitespace.",
        )

    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model failed to load: {e}",
        )

    prediction = model.predict(
        review=payload.review,
        threshold=payload.threshold if payload.threshold is not None else 0.5,
    )

    return PredictionResponse(
        review=payload.review,
        sentiment=prediction["sentiment"],
        score=prediction["score"],
        confidence=prediction["confidence"],
        token_count=prediction["token_count"],
        total_tokens=prediction["total_tokens"],
        truncated=prediction["truncated"],
    )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Inference"])
async def batch_predict_sentiment(payload: BatchReviewRequest):
    """
    Classify multiple reviews in a single request.
    """
    if not payload.reviews:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'reviews' list cannot be empty.",
        )

    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model failed to load: {e}",
        )

    threshold = payload.threshold if payload.threshold is not None else 0.5
    results: List[PredictionResponse] = []

    for item in payload.reviews:
        if not item or not item.strip():
            # If item is empty, mark as unknown/empty
            results.append(
                PredictionResponse(
                    review=item,
                    sentiment="Neutral",
                    score=0.5,
                    confidence=0.5,
                    token_count=0,
                    total_tokens=0,
                    truncated=False,
                )
            )
            continue

        pred = model.predict(review=item, threshold=threshold)
        results.append(
            PredictionResponse(
                review=item,
                sentiment=pred["sentiment"],
                score=pred["score"],
                confidence=pred["confidence"],
                token_count=pred["token_count"],
                total_tokens=pred["total_tokens"],
                truncated=pred["truncated"],
            )
        )

    return BatchPredictionResponse(total_count=len(results), results=results)


if __name__ == "__main__":
    import uvicorn

    # Render automatically supplies $PORT, defaulting to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
