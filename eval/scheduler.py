import time
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from eval.simulate_ground_truth import simulate_ground_truth
from eval.compute_metrics import compute_and_save_metrics
from eval.drift import detect_drift

# We use APScheduler for observability and reliability

# Setup Logging to Console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("scheduler")

# Job to simulate ground truth every 30 seconds (for the demo)
def job_simulation():
    logger.info("Triggering Ground Truth Simulation Job...")
    try:
        simulate_ground_truth()
    except Exception as e:
        logger.error(f"Simulation Job Failed: {e}", exc_info=True)


# Job to compute and save metrics every 30 seconds
def job_metrics():
    logger.info("Triggering Metric Computation Job...")
    try:
        compute_and_save_metrics()
    except Exception as e:
        logger.error(f"Metric Computation Failed: {e}", exc_info=True)

def job_drift():
    logger.info("Triggering Drift Detection Job...")
    try:
        detect_drift()
    except Exception as e:
        logger.error(f"Drift Detection Failed: {e}", exc_info=True)

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