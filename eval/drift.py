# Since we don't have real training data yet, we simulate it by generating random data.
import pandas as pd
import numpy as np
import json
from scipy.stats import ks_2samp
from psycopg2.extras import execute_values
import logging
from eval.db_utils import get_db_conn, get_latest_version, retry_db_write
from eval.alerting import send_discord_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("drift_detection")

# --- CONFIGURATION ---
# We simulate the "Training Data" statistics for INCOME.
# Assumption: In training, average income was $55k.
threshold = 0.05
REFERENCE_INCOME = np.random.normal(55000, 15000, 1000)

def detect_drift():
    """
    Detects data drift in the (income) feature of the latest predictions.

    Returns:
        Tuple (is_drift, p_value)
        Returns (False, 0.0) if not enough data to run drift detection or if the JSON parsing fails
        is_drift: True if drift is detected, False otherwise
        p_value: The p-value of the KS-Test (higher is better, > 0.05 means no drift)
    """
    conn = get_db_conn()
    
    # 1. Fetch Recent Data (Inputs and Outputs)
    query = """
        SELECT input_data, prediction_prob 
        FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT 100
    """
    df = pd.read_sql(query, conn)
    
    if len(df) < 50:
        logger.info("Not enough data to run drift detection (<50 samples). Returning False and 0.0")
        return (False, 0.0)

    # 2. Extract 'income' from the JSON column 'input_data'
    try:
        # Create a new column 'income' by extracting it from the JSON
        df['income'] = df['input_data'].apply(lambda x: x['income'] if isinstance(x, dict) else json.loads(x)['income'])
    except Exception as e:
        logger.error(f"Failed to parse input_data JSON: {e}. Returning False and 0.0")
        return (False, 0.0)

    # 3. Run KS Test on INCOME (Input Drift)
    # "Is the Income distribution of the last 100 applicants different from training?"
    statistic, p_value = ks_2samp(REFERENCE_INCOME, df['income'])
    
    logger.info(f"Drift Check (Income) -> P-Value: {p_value:.5f}")

    # 4. Save Metric to DB (with retry)
    window_end = pd.Timestamp.now()
    window_start = window_end - pd.Timedelta(hours=1)
    current_version = get_latest_version()
    insert_query = """
        INSERT INTO metrics (metric_name, metric_value, model_version, window_start, window_end)
        VALUES %s
    """
    rows = [('drift_income_p_value', float(p_value), current_version, window_start, window_end)]
    
    # Close connection for DB reads to prepare for writing
    conn.close()

    def _write_drift_metric(conn, insert_query, rows):
        execute_values(conn.cursor(), insert_query, rows)
        conn.commit()

    retry_db_write(_write_drift_metric, insert_query, rows)

    # 5. Alerting
    if p_value < threshold:
        logger.warning("Significant data drift detected: p-value < 0.05")
        msg = (
            f"🚨 **Significant Data Drift Detected** 🚨\n"
            f"**Feature:** INCOME\n"
            f"**P-Value:** `{p_value:.5f}` (Threshold: {threshold})\n"
            f"**Status:** Applicants are significantly poorer/richer than training data.\n"
            f"**Action:** Check for model degradation."
        )
        send_discord_alert(msg)

    logger.info("Drift detection completed successfully.")

    return (p_value < threshold, p_value)

