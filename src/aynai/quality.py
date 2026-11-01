import math
from pathlib import Path

from aynai.storage import DEFAULT_DB_PATH, get_results


MINIMUM_HISTORY_SIZE = 5


def analyze_data_quality(
    url: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict:
    """Generate a data quality report for a monitored service."""

    records = get_results(url=url, db_path=db_path)

    total_records = len(records)
    successful_records = 0
    failed_records = 0
    valid_latency_records = 0
    invalid_latency_records = 0
    analyzed_records = 0
    labeled_records = 0
    for record in records:
        if record["is_available"]:
            successful_records += 1

            response_time = record["response_time_ms"]

            try:
                latency = float(response_time)
                is_valid_latency = (
                    math.isfinite(latency) and latency >= 0
                )
            except (TypeError, ValueError):
                is_valid_latency = False

            if is_valid_latency:
                valid_latency_records += 1
            else:
                invalid_latency_records += 1

        else:
            failed_records += 1

        if record["is_anomaly"] is not None:
            analyzed_records += 1

        if (
            record["latency_anomaly_label"] is not None
            or record["availability_issue_label"] is not None
        ):
            labeled_records += 1
   
    return {
        "url": url,
        "total_records": total_records,
        "successful_records": successful_records,
        "failed_records": failed_records,
        "valid_latency_records": valid_latency_records,
        "invalid_latency_records": invalid_latency_records,
        "analyzed_records": analyzed_records,
        "labeled_records": labeled_records,
        "history_sufficient": (
            valid_latency_records >= MINIMUM_HISTORY_SIZE
        ),
    }