# This script simulates the ground truth labels for the predictions table, 
# in production we use real ground truth labels from the dataset, and insert into the ground_truth table.
# Script is used to evaluate the performance of the model by randomly labelling the predictions.

# Imports
import os
import random
import logging
from psycopg2.extras import execute_values
from eval.db_utils import get_db_conn  # this was originally defined for each script, but importing makes it easier to adjust

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ground_truth_sim")

def simulate_ground_truth():
    conn = get_db_conn()
    cur = conn.cursor()

    # 1. Find predictions that don't have a label yet
    logger.info("Fetching unlabelled predictions...")
    cur.execute("""
        SELECT p.request_id, p.prediction_class, p.prediction_prob 
        FROM predictions p
        LEFT JOIN ground_truth g ON p.request_id = g.request_id
        WHERE g.request_id IS NULL
    """)
    rows = cur.fetchall()

    if not rows:
        logger.info("No new predictions to label.")
        return

    logger.info(f"Generating labels for {len(rows)} predictions...")
    
    new_labels = []
    for row in rows:
        req_id, pred_class, prob = row
        
        # --- SIMULATION LOGIC ---
        # We want the model to be roughly correct.
        # If the model was confident (prob > 0.8 or < 0.2), usually match it.
        # If the model was unsure (0.4 - 0.6), flip a coin.
        
        if prob > 0.8: 
            actual_class = 1 if random.random() < 0.9 else 0
        elif prob < 0.2:
            actual_class = 0 if random.random() < 0.9 else 1
        else:
            # Random chance for uncertain predictions
            actual_class = 1 if random.random() < 0.5 else 0
            
        new_labels.append((req_id, actual_class))
        # ------------------------

    # 2. Bulk Insert labels
    insert_query = "INSERT INTO ground_truth (request_id, actual_class) VALUES %s"
    execute_values(cur, insert_query, new_labels)
    
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Successfully added {len(new_labels)} ground truth labels.")

if __name__ == "__main__":
    simulate_ground_truth()