import os
import logging
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


MODEL_PATH_DEFAULT = "../../../models/model.pkl"

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="SMS Spam Classifier API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, description="text")


class PredictOut(BaseModel):
    label: Literal["spam", "ham"]
    proba: float = Field(..., ge=0.0, le=1.0, description="Probability spam")


_model = None


@app.on_event("startup")
def _load_model_on_startup() -> None:
    global _model
    model_path = os.getenv("MODEL_PATH", MODEL_PATH_DEFAULT)
    if not os.path.exists(model_path):
        logger.error("Model file not found at %s", model_path)
        raise RuntimeError(f"Model file not found at {model_path}")

    try:
        _model = joblib.load(model_path)
        logger.info("Model loaded successfully from %s", model_path)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn) -> PredictOut:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        proba_spam: float = float(_model.predict_proba([payload.text])[0][1])
        label: Literal["spam", "ham"] = "spam" if proba_spam >= 0.5 else "ham"
        return PredictOut(label=label, proba=round(proba_spam, 4))
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        workers=1,
    )