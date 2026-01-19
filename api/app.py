import os
import time
import uuid
import json
from fastapi import FastAPI, BackgroundTasks, HTTPException
import psycopg2
from api.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="ML Monitoring Inference Service")

# DB Connection - In production, use a connection pool (like sqlalchemy)
# For this portfolio, direct psycopg2 is simpler and easier to explain for development
def save_prediction_to_db(payload: dict):
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO predictions 
               (request_id, model_version, input_data, prediction_prob, prediction_class, latency_ms) 
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                payload['request_id'], 
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
    except Exception as e:
        print(f"Database logging failed: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    
    # --- MOCK MODEL LOGIC ---
    # we'll replace this with a real DVC-tracked model.
    # Logic: Higher credit score + higher income = lower probability of default.
    normalized_score = (request.credit_score - 300) / 550
    prob = 1.0 - (0.7 * normalized_score + 0.3 * min(request.income / 100000, 1))
    prob = max(0, min(1, prob)) # Clip between 0 and 1
    pred_class = 1 if prob > 0.5 else 0
    # ------------------------

    # Calculate latency in milliseconds for readability and standardisation
    latency = (time.time() - start_time) * 1000
    request_id = str(uuid.uuid4())
    model_version = "v1.0.0" # TODO: replace with DVC-tracked model version

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