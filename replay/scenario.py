"""Train, serve, and verify the model inside an isolated replay project."""

import argparse
from contextlib import closing
import hashlib
import json
from math import isclose
from pathlib import Path
import pickle
import time
from uuid import UUID

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp, norm
from sklearn.ensemble import RandomForestClassifier
import uvicorn

from eval.db_utils import get_db_conn
from eval.drift import detect_drift, threshold


MODEL_VERSION = "demo-v1.0.0"
MODEL_PATH = Path("/app/models") / f"{MODEL_VERSION}.pkl"
REFERENCE_COUNT = 1000
BATCH_COUNT = 100
INCOME_MEAN = 55_000
INCOME_STDDEV = 15_000
INCOME_SHIFTS = {"baseline": 0, "income_shift": 35_000}


def synthetic_features(income, rng):
    count = len(income)
    return pd.DataFrame({
        "income": income.clip(min=1).round(2),
        "debt": rng.uniform(0, 30_000, count).round(2),
        "credit_score": rng.integers(300, 851, count),
    })


def training_features():
    rng = np.random.default_rng(2718)
    return synthetic_features(rng.normal(INCOME_MEAN, INCOME_STDDEV, REFERENCE_COUNT), rng)


def bootstrap():
    features = training_features()
    noise = np.random.default_rng(3141).normal(0, 0.4, len(features))
    risk = (features["debt"] / features["income"].clip(lower=10_000)
            + (650 - features["credit_score"]) / 250 + noise)
    model = RandomForestClassifier(n_estimators=32, max_depth=5, random_state=2025)
    model.fit(features, (risk > 0.45).astype(int))
    MODEL_PATH.write_bytes(pickle.dumps(model, protocol=4))

    with closing(get_db_conn()) as conn, conn, conn.cursor() as cur:
        cur.execute("DELETE FROM model_versions")
        cur.execute(
            "INSERT INTO model_versions (version, filepath, is_active) VALUES (%s, %s, TRUE)",
            (MODEL_VERSION, str(MODEL_PATH)),
        )


def serve():
    from api import app as inference

    inference.load_active_model()
    if inference.current_model is None or inference.current_version != MODEL_VERSION:
        raise RuntimeError(f"API did not load {MODEL_VERSION}")
    uvicorn.run(inference.app, host="0.0.0.0", port=8000)


def submit_batch(scenario):
    rng = np.random.default_rng(1618)
    quantiles = (np.arange(BATCH_COUNT) + rng.random(BATCH_COUNT)) / BATCH_COUNT
    rng.shuffle(quantiles)
    batch = synthetic_features(norm.ppf(quantiles, INCOME_MEAN, INCOME_STDDEV), rng)
    batch["income"] += INCOME_SHIFTS[scenario]
    expected = {}
    with requests.Session() as session:
        for sample in batch.to_dict("records"):
            response = session.post("http://api:8000/predict", json=sample, timeout=10)
            response.raise_for_status()
            result = response.json()
            request_id = str(UUID(result["request_id"]))
            if result["model_version"] != MODEL_VERSION or request_id in expected:
                raise RuntimeError(f"{scenario}: unexpected model version or duplicate request ID")
            expected[request_id] = (
                MODEL_VERSION, sample, result["prediction_prob"], result["prediction_class"],
            )
    return expected


def verify_predictions(expected):
    deadline = time.monotonic() + 20
    with closing(get_db_conn()) as conn, conn.cursor() as cur:
        while True:
            cur.execute("""
                SELECT request_id::text, model_version, input_data, prediction_prob, prediction_class
                FROM predictions
            """)
            saved = {request_id: tuple(values) for request_id, *values in cur.fetchall()}
            if saved.keys() - expected.keys():
                raise RuntimeError("Found predictions outside this scenario")
            if saved.keys() == expected.keys():
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Only {len(saved)} of {BATCH_COUNT} predictions were saved")
            time.sleep(0.25)
    if saved != expected:
        raise RuntimeError("Stored inputs, outputs, or model versions differ from the API")
    return [saved[request_id][1]["income"] for request_id in expected]


def evaluate(scenario):
    income = verify_predictions(submit_batch(scenario))
    reference = training_features()["income"].to_numpy()
    expected_ks = ks_2samp(reference, income)
    drift_detected, p_value = detect_drift(reference_income=reference)

    with closing(get_db_conn()) as conn, conn.cursor() as cur:
        cur.execute("SELECT metric_name, metric_value, model_version FROM metrics")
        saved = cur.fetchall()
    if saved != [("drift_income_p_value", p_value, MODEL_VERSION)]:
        raise RuntimeError(f"{scenario}: expected exactly one matching drift metric")
    if not isclose(p_value, expected_ks.pvalue, rel_tol=1e-12):
        raise RuntimeError(f"{scenario}: stored p-value differs from KS on the verified batch")
    if drift_detected != (p_value < threshold) or drift_detected != (scenario == "income_shift"):
        raise RuntimeError(f"{scenario}: unexpected drift decision at p={p_value:.6g}")

    return {
        "model": {
            "version": MODEL_VERSION,
            "sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        },
        "reference_income": reference.tolist(),
        "drift_threshold": threshold,
        "result": {
            "income": income,
            "sample_count": len(income),
            "ks_statistic": float(expected_ks.statistic),
            "ks_p_value": float(p_value),
            "drift_detected": bool(drift_detected),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bootstrap", "serve", *INCOME_SHIFTS))
    args = parser.parse_args()
    if args.action == "bootstrap":
        bootstrap()
    elif args.action == "serve":
        serve()
    else:
        print(json.dumps(evaluate(args.action), allow_nan=False))


if __name__ == "__main__":
    main()
