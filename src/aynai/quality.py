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
    latency_labeled_records = 0
    availability_labeled_records = 0
    unlabeled_records = 0
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

        latency_label = record["latency_anomaly_label"]
        availability_label = record["availability_issue_label"]

        has_latency_label = latency_label in (0, 1)
        has_availability_label = availability_label in (0, 1)

        if has_latency_label:
            latency_labeled_records += 1

        if has_availability_label:
            availability_labeled_records += 1

        if has_latency_label or has_availability_label:
            labeled_records += 1
        else:
            unlabeled_records += 1
    latency_label_coverage = (
        latency_labeled_records / total_records * 100
        if total_records > 0
        else 0.0
    )

    availability_label_coverage = (
        availability_labeled_records / total_records * 100
        if total_records > 0
        else 0.0
    )

    overall_label_coverage = (
        labeled_records / total_records * 100
        if total_records > 0
        else 0.0
    )
    return {
        "url": url,
        "total_records": total_records,
        "successful_records": successful_records,
        "failed_records": failed_records,
        "valid_latency_records": valid_latency_records,
        "invalid_latency_records": invalid_latency_records,
        "analyzed_records": analyzed_records,
        "labeled_records": labeled_records,
        "latency_labeled_records": latency_labeled_records,
        "availability_labeled_records": availability_labeled_records,
        "unlabeled_records": unlabeled_records,
        "latency_label_coverage": latency_label_coverage,
        "availability_label_coverage": availability_label_coverage,
        "overall_label_coverage": overall_label_coverage,
        "history_sufficient": (
            valid_latency_records >= MINIMUM_HISTORY_SIZE
        ),
    }