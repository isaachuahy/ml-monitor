import os
import time
import uuid
import re
import logging
import threading
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from contextlib import asynccontextmanager
import psycopg2
from api.schemas import PredictionRequest, PredictionResponse
from api.prediction_store import PredictionConflict, PredictionStore, PredictionUnavailable

# Logging setup
# Configure logging to show timestamp, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api")

current_model = None
current_version ="v1.0.0"
model_lock = threading.Lock() # To prevent race conditions when loading/switching models

def load_active_model():
    """
    Load the active database model when missing or when its version changes.
    """
    global current_model, current_version
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute("SELECT version, filepath FROM model_versions WHERE is_active = TRUE ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()

        if row:
            new_version, filepath = row
            if current_model is None or new_version != current_version:
                logger.info(f"Loading new model version {new_version} from {filepath}")
                try:
                    with open(filepath, "rb") as f:
                        new_model = pickle.load(f)

                    # atomic swap
                    # we do this because pair of assignments should match
                    with model_lock:
                        current_model = new_model
                        current_version = new_version
                        logger.info(f"Switched to new model version {new_version}")
                except FileNotFoundError as e:
                    logger.warning(f"Model file not found: {filepath}. Model may not exist yet.")
    except Exception as e:
        logger.error(f"Failed to load active model: {e}", exc_info=True)
                
def background_model_reloader():
    """
    Runs in background thread to poll for updates.
    """
    while True:
        load_active_model()
        time.sleep(30) # Poll every 30 seconds

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for the FastAPI application.
    """

    logger.info("Starting background model reloader")
    # Note: /app/models is mounted as a volume from ./models, so no need to create it here
    
    # Start background poller
    t=threading.Thread(target=background_model_reloader, daemon=True)
    t.start()
    yield

# Context manager is lifespan which 
app = FastAPI(title="ML Monitoring Inference Service", lifespan=lifespan)

prediction_store = PredictionStore()


@app.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}


def infer_prediction(request: PredictionRequest):
    """Run the captured active model or the explicitly labelled mock fallback."""
    with model_lock:
        model = current_model
        model_version = current_version

    if model is not None:
        input_df = pd.DataFrame({
            "income": [request.income],
            "debt": [request.debt],
            "credit_score": [request.credit_score],
        })
        probability = float(model.predict_proba(input_df)[0][1])
        prediction_class = int(model.predict(input_df)[0])
    else:
        logger.warning("Model not loaded, using mock prediction logic")
        normalized_score = (request.credit_score - 300) / 550
        probability = 1.0 - (0.7 * normalized_score + 0.3 * min(request.income / 100000, 1))
        probability = max(0, min(1, probability))
        prediction_class = 1 if probability > 0.5 else 0
        model_version = "mock"
    return {"prediction_prob": probability, "prediction_class": prediction_class,
            "model_version": model_version}


@app.post("/predict", response_model=PredictionResponse)
async def predict(body: PredictionRequest, request: Request, response: Response):
    """Return a committed prediction, replaying matching idempotency keys."""
    keys = request.headers.getlist("idempotency-key")
    if len(keys) > 1 or (keys and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", keys[0])):
        raise HTTPException(status_code=422, detail="Invalid Idempotency-Key")
    key = keys[0] if keys else "server:" + str(uuid.uuid4())
    headers = {"Idempotency-Key": key}
    try:
        result = await prediction_store.get_or_create(key, body.model_dump(), lambda: infer_prediction(body))
    except PredictionConflict:
        raise HTTPException(status_code=409, detail="Idempotency key belongs to different inputs", headers=headers) from None
    except PredictionUnavailable:
        raise HTTPException(status_code=503, detail="Prediction unavailable; retry with the same idempotency key", headers=headers) from None
    response.headers["Idempotency-Key"] = key
    return result
