import time
import logging
import uuid
from apscheduler.schedulers.blocking import BlockingScheduler
from eval.simulate_ground_truth import simulate_ground_truth
from eval.compute_metrics import compute_and_save_metrics
from eval.drift import detect_drift
from eval.retrain import retrain_model

# We use APScheduler for observability and reliability
# Reminder: functions log what happened, scheduler logs what we're doing about it and how.

# Retrain cooldown: skip retrain if last retrain started within this many seconds
RETRAIN_COOLDOWN_SECONDS = 300  # 5 minutes
last_retrain_start_time = None  # seconds since epoch, or None

# Setup Logging to Console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("scheduler")

# Job to simulate ground truth every 30 seconds (for the demo)
def job_simulation():
    logger.info("Triggering Ground Truth Simulation Job...")
    t0 = time.time()
    try:
        simulate_ground_truth()
        logger.info(f"Ground Truth Simulation Job completed in {time.time() - t0:.2f}s")
    except Exception as e:
        logger.error(f"Simulation Job Failed: {e}", exc_info=True)
        logger.error(f"Ground Truth Simulation Job failed after {time.time() - t0:.2f}s")


# Job to compute and save metrics every 30 seconds
def job_metrics():
    logger.info("Triggering Metric Computation Job...")
    t0 = time.time()
    try:
        compute_and_save_metrics()
        logger.info(f"Metric Computation Job completed in {time.time() - t0:.2f}s")
    except Exception as e:
        logger.error(f"Metric Computation Failed: {e}", exc_info=True)
        logger.error(f"Metric Computation Job failed after {time.time() - t0:.2f}s")


def job_drift():
    global last_retrain_start_time
    logger.info("Triggering Drift Detection Job...")
    t0 = time.time()
    try:
        drift_detected, p_value = detect_drift()
        logger.info(f"Drift Detection Job completed in {time.time() - t0:.2f}s")
        if drift_detected:
            now = time.time()
            if last_retrain_start_time is not None and (now - last_retrain_start_time) < RETRAIN_COOLDOWN_SECONDS:
                elapsed = now - last_retrain_start_time
                logger.warning(
                    "Skipping retrain: cooldown active (last retrain started %.0fs ago, "
                    "min %ds required)",
                    elapsed,
                    RETRAIN_COOLDOWN_SECONDS,
                )
                return
            run_id = uuid.uuid4().hex[:8]
            last_retrain_start_time = now
            logger.info(
                "run_id=%s triggering retrain after drift (p_value=%.5f)",
                run_id,
                p_value,
            )
            t_retrain = time.time()
            try:
                retrain_model(run_id=run_id)
                logger.info(f"run_id={run_id} Retraining completed in {time.time() - t_retrain:.2f}s")
            except Exception as e:
                logger.error(f"run_id={run_id} Retraining Job Failed: {e}", exc_info=True)
                logger.error(f"run_id={run_id} Retraining failed after {time.time() - t_retrain:.2f}s")
    except Exception as e:
        logger.error(f"Drift Detection Failed: {e}", exc_info=True)
        logger.error(f"Drift Detection Job failed after {time.time() - t0:.2f}s")

if __name__ == "__main__":
    # Create the scheduler
    # Runs in main thread and blocks the main thread from exiting
    scheduler = BlockingScheduler()

    # --- SCHEDULE CONFIGURATION ---
    # In a real production environment, we'd run these daily (hours=24)
    # For the demos, we run them every 30 SECONDS 
    # so we can see the graphs move live.
    
    # 1. Generate fake labels every 30 seconds
    scheduler.add_job(job_simulation, 'interval', seconds=30)
    
    # 2. Re-calculate metrics every 30 seconds (TODO: stagger slightly if needed)
    scheduler.add_job(job_metrics, 'interval', seconds=30)
    
    # 3. Detect drift every 60 seconds (less frequent than metrics)
    scheduler.add_job(job_drift, 'interval', seconds=60)
    
    logger.info("Scheduler started! Jobs will run every 30 seconds.")
    
    try:
        # This keeps the process alive permanently
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass