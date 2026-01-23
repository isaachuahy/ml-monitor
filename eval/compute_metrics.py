import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from psycopg2.extras import execute_values
import logging
from eval.db_utils import get_db_conn  # Importing our shared tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("metrics_compute")

# Define the function to compute and save the metrics
def compute_and_save_metrics():
    # Get a database connection using the utility function.
    conn = get_db_conn()
    
    # 1. Fetch data (Predictions + Ground Truth) from the last 7 days
    query = """
        SELECT 
            p.prediction_class,
            g.actual_class
        FROM predictions p
        JOIN ground_truth g ON p.request_id = g.request_id
        WHERE p.timestamp > NOW() - INTERVAL '7 days' 
    """
    
    logger.info("Fetching data for evaluation from the last 7 days...")
    df = pd.read_sql(query, conn)
    
    if df.empty:
        logger.warning("No matched data found! (Did you run simulate_ground_truth.py to label the predictions?)")
        return

    # 2. Compute Metrics
    acc = accuracy_score(df['actual_class'], df['prediction_class'])
    f1 = f1_score(df['actual_class'], df['prediction_class'])
    
    logger.info(f"Computed Metrics -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # 3. Save to DB
    insert_query = """
        INSERT INTO metrics (metric_name, metric_value, model_version, window_start, window_end)
        VALUES %s
    """
    
    window_end = pd.Timestamp.now()
    window_start = window_end - pd.Timedelta(days=7)
    model_version = "v1.0.0" 
    
    metrics_to_insert = [
        ('accuracy', float(acc), model_version, window_start, window_end),
        ('f1_score', float(f1), model_version, window_start, window_end)
    ]
    
    cur = conn.cursor()
    execute_values(cur, insert_query, metrics_to_insert)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Metrics saved successfully.")

if __name__ == "__main__":
    compute_and_save_metrics()