import os
import time
import uuid
import json
import logging
import threading
import pickle
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
import psycopg2
from api.schemas import PredictionRequest, PredictionResponse

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
    Check DB for active model and loads if different from current model. 
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
            # Only reload if version changed
            if new_version != current_version:
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

# DB Connection - In production, use a connection pool (like sqlalchemy)
# For this portfolio, direct psycopg2 is simpler and easier to explain for development
def save_prediction_to_db(payload: dict):
    try:
        request_id = payload['request_id']
        logger.info(f"Attempting to save prediction to database for request {request_id}")
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO predictions 
               (request_id, model_version, input_data, prediction_prob, prediction_class, latency_ms) 
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                request_id, 
                payload['model_version'], 
                json.dumps(payload['input_data']), 
                payload['prediction_prob'], 
                payload['prediction_class'], 
                payload['latency_ms']
            )
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"SUCCESS: Prediction saved to database for request {request_id}")
    except Exception as e:
        logger.error(f"FAILURE: Write request {request_id} failed: {e}", exc_info=True)


@app.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received prediction request: {request.model_dump()}")
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Use loaded model if available, otherwise fallback to mock logic
    with model_lock:
        model = current_model
        model_version = current_version
    
    if model is not None:
        # Use the loaded ML model for prediction
        # Prepare input as DataFrame matching training format
        input_df = pd.DataFrame({
            "income": [request.income],
            "debt": [request.debt],
            "credit_score": [request.credit_score]
        })
        
        # Get prediction probability and class
        prob = float(model.predict_proba(input_df)[0][1])  # Probability of class 1
        pred_class = int(model.predict(input_df)[0])
    else:
        # Fallback to mock logic if model not loaded yet
        logger.warning("Model not loaded, using mock prediction logic")
        normalized_score = (request.credit_score - 300) / 550
        prob = 1.0 - (0.7 * normalized_score + 0.3 * min(request.income / 100000, 1))
        prob = max(0, min(1, prob)) # Clip between 0 and 1
        pred_class = 1 if prob > 0.5 else 0
        model_version = "mock"

    # Calculate latency in milliseconds for readability and standardisation
    latency = (time.time() - start_time) * 1000

    # Log for monitoring the prediction to the database
    log_payload = {
        "request_id": request_id,
        "model_version": model_version,
        "input_data": request.model_dump(), # Convert PredictionRequest to dict for logging
        "prediction_prob": prob,
        "prediction_class": pred_class,
        "latency_ms": latency
    }

    # "Fire and forget" the DB write to avoid blocking the request
    background_tasks.add_task(save_prediction_to_db, log_payload)

    return PredictionResponse(
        request_id=request_id,
        prediction_prob=prob,
        prediction_class=pred_class,
        model_version=model_version
    )