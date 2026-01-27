# Since we don't have real training data yet, we simulate it by generating random data.
import pandas as pd
import numpy as np
import json
from scipy.stats import ks_2samp
from psycopg2.extras import execute_values
import logging
from eval.db_utils import get_db_conn
from eval.alerting import send_discord_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("drift_detection")

# --- CONFIGURATION ---
# We simulate the "Training Data" statistics for INCOME.
# Assumption: In training, average income was $55k.
threshold = 0.05
REFERENCE_INCOME = np.random.normal(55000, 15000, 1000)

def detect_drift():
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
        logger.info("Not enough data to run drift detection (<50 samples).")
        return

    # 2. Extract 'income' from the JSON column 'input_data'
    try:
        # Create a new column 'income' by extracting it from the JSON
        df['income'] = df['input_data'].apply(lambda x: x['income'] if isinstance(x, dict) else json.loads(x)['income'])
    except Exception as e:
        logger.error(f"Failed to parse input_data JSON: {e}")
        return

    # 3. Run KS Test on INCOME (Input Drift)
    # "Is the Income distribution of the last 100 applicants different from training?"
    statistic, p_value = ks_2samp(REFERENCE_INCOME, df['income'])
    
    logger.info(f"Drift Check (Income) -> P-Value: {p_value:.5f}")
    
    # 4. Save Metric to DB
    insert_query = """
        INSERT INTO metrics (metric_name, metric_value, model_version, window_start, window_end)
        VALUES %s
    """
    window_end = pd.Timestamp.now()
    window_start = window_end - pd.Timedelta(hours=1)
    
    execute_values(conn.cursor(), insert_query, [
        ('drift_income_p_value', float(p_value), 'v1.0.0', window_start, window_end)
    ])
    conn.commit()

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
    
    conn.close()

