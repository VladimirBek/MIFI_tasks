import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict

LOG_HEADERS = [
    "timestamp_utc",
    "user_id",
    "ticker",
    "amount",
    "best_model",
    "metric_primary",
    "rmse",
    "mape",
    "profit_estimate",
    "buy_dates",
    "sell_dates",
]

def append_log_row(log_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        if not file_exists:
            writer.writeheader()
        out = {k: row.get(k, "") for k in LOG_HEADERS}
        writer.writerow(out)

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
