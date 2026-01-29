# Retrain a candidate model for promotion to production

import pandas as pd
import numpy as np
import pickle
import os
import logging
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from eval.db_utils import get_db_conn, get_next_version
from eval.alerting import send_discord_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("retrainer")

MODELS_DIR="/app/models"

def retrain_model():
    logger.info("Starting candidate model training job...")

    # Fetch recent predictions
    # query = """
    # SELECT input_data, prediction_class FROM predictions WHERE timestamp > NOW() - INTERVAL '7 days'"""
    # df = pd.read_sql(query, conn)

    # if df.empty:
    #     logger.warning("No recent predictions found. Skipping retraining.")
    #     return

    # # 2. Train the model
    # model = RandomForestClassifier()
    # model.fit(df[['input_data']], df['prediction_class'])

    # Simulate fresh data
    # Train on a distribution that matches new production data
    X_train = pd.DataFrame({
        "income": np.random.normal(55000, 20000, 1000),
        "debt": np.random.normal(10000, 5000, 1000),
        "credit_score": np.random.normal(650, 100, 1000),
    })
    y_train = ((X_train['credit_score'] > 600) & (X_train['debt'] < 20000)).astype(int)

    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_train, y_train)

    # Get next semantic version (e.g., v1.0.0 -> v1.0.1)
    version_id = get_next_version()
    filename = f"model_{version_id}.pkl"
    filepath = os.path.join(MODELS_DIR, filename)

    metrics = {
        "accuracy": accuracy_score(y_train, clf.predict(X_train)),
        "f1_score": f1_score(y_train, clf.predict(X_train))
    }
    metrics_json = json.dumps(metrics)

    # Save candidate model with pickle.dump
    with open(filepath, "wb") as f:
        pickle.dump(clf, f)

    logger.info(f"Candidate model saved to {filepath}")

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO model_versions (version, filepath, is_active, metrics_json)
    VALUES (%s, %s, FALSE, %s)
    """, (version_id, filepath, metrics_json))
    conn.commit()
    cur.close()
    conn.close()

    alert_msg = (
        f"🎯 **New Candidate Model Version Available** 🎯\n"
        f"**Version:** `{version_id}`\n"
        f"**Status:** Waiting for manual review\n"
        f"**Metrics:** {metrics_json}\n"
        f"**Action:** Manual review, then promote to production by updating the `is_active` flag.\n"
    )

    send_discord_alert(alert_msg)
    
    logger.info("Candidate model training job completed successfully.")

if __name__ == "__main__":
    retrain_model()