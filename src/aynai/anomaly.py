import math
from pathlib import Path
from statistics import median

from aynai.storage import DEFAULT_DB_PATH, get_results


def detect_latency_anomaly(
    current_response_time_ms: float,
    historical_response_times: list[float],
    threshold: float = 3.5,
) -> dict:
    """Detect unusually high latency using the robust MAD method."""
    if not math.isfinite(current_response_time_ms):
        raise ValueError("Response time must be finite")

    if not all(math.isfinite(value) for value in historical_response_times):
        raise ValueError("Historical response times must be finite")

    if current_response_time_ms < 0:
        raise ValueError("Response time cannot be negative")

    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Threshold must be finite and greater than zero")

    if any(value < 0 for value in historical_response_times):
        raise ValueError("Historical response times cannot be negative")

    if len(historical_response_times) < 5:
        return {
            "is_anomaly": False,
            "baseline_ms": None,
            "mad_ms": None,
            "reason": "Insufficient historical data",
        }

    baseline = median(historical_response_times)
    deviations = [abs(value - baseline) for value in historical_response_times]
    mad = median(deviations)

    if mad == 0:
        is_anomaly = (
            current_response_time_ms > baseline
            and current_response_time_ms > 1.5 * baseline
        )
    else:
        robust_z_score = 0.6745 * (current_response_time_ms - baseline) / mad
        is_anomaly = robust_z_score > threshold

    return {
        "is_anomaly": is_anomaly,
        "baseline_ms": round(baseline, 2),
        "mad_ms": round(mad, 2),
        "reason": (
            "Latency above expected range"
            if is_anomaly
            else "Latency within expected range"
        ),
    }


def analyze_service_latency(
    url: str,
    current_response_time_ms: float,
    db_path: str | Path = DEFAULT_DB_PATH,
    history_limit: int = 100,
    exclude_record_id: int | None = None,
) -> dict:
    """Compare current latency with previous successful checks."""
    if history_limit < 5:
        raise ValueError("history_limit must be at least 5")

    results = get_results(url=url, db_path=db_path)

    if exclude_record_id is not None:
        historical_results = [
            result for result in results if result["id"] != exclude_record_id
        ]
    else:
        historical_results = results

    historical_times = []

    for result in historical_results:
        if not result["is_available"]:
            continue

        try:
            response_time = float(result["response_time_ms"])
        except (TypeError, ValueError):
            continue

        if not math.isfinite(response_time) or response_time < 0:
            continue

        historical_times.append(response_time)

        if len(historical_times) >= history_limit:
            break

    return detect_latency_anomaly(
        current_response_time_ms=current_response_time_ms,
        historical_response_times=historical_times,
    )
