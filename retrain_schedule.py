import shutil
import logging
import traceback
from datetime import datetime
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "retrain.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

MODEL_PATH = Path("models/lightgbm_stock_signal.pkl")


def run_retrain():
    log.info("═" * 50)
    log.info("Scheduled retrain started")

    try:
        # Back up current model before doing anything
        if MODEL_PATH.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            backup = MODEL_PATH.with_name(f"lightgbm_stock_signal_{ts}.pkl")
            shutil.copy2(MODEL_PATH, backup)
            log.info(f"Backed up current model → {backup}")

        # Run training
        from train_model import train
        train()

        log.info("Retrain completed successfully")

    except Exception:
        log.error("Retrain FAILED — previous model preserved")
        log.error(traceback.format_exc())

        # If training produced a broken model, restore the backup
        if MODEL_PATH.exists() and 'backup' in locals():
            shutil.copy2(backup, MODEL_PATH)
            log.warning(f"Restored backup model from {backup}")


if __name__ == "__main__":
    run_retrain()