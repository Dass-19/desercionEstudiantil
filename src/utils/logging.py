from datetime import datetime, timezone
from pathlib import Path
import csv


def logPrediction(
        path: str,
        prob_rf: float,
        prob_lr: float,
        prediction: int,
        threshold: float,
        model_version: str
        ):
    LOG_PATH = Path("LOG_PATH")
    is_new = not LOG_PATH.exists()

    with open(str(LOG_PATH), mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if is_new:
            writer.writerow([
                "timestamp",
                "probability_rf",
                "probability_lr",
                "prediction",
                "threshold",
                "model_version"
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            round(prob_rf, 4),
            round(prob_lr, 4),
            int(prediction),
            threshold,
            model_version
        ])
