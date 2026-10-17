from datetime import datetime, timedelta, timezone
from math import ceil, isfinite
from pathlib import Path

from aynai.storage import DEFAULT_DB_PATH, get_results


def summarize_results(
    url: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    since_hours: float | None = None,
) -> dict:
    """Calculate monitoring statistics from stored results."""
    if since_hours is not None:
        if not isfinite(since_hours) or since_hours <= 0:
            raise ValueError("since_hours must be finite and greater than zero")

        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

        results = [
            result
            for result in get_results(url=url, db_path=db_path)
            if datetime.fromisoformat(result["checked_at"]).astimezone(timezone.utc)
            >= cutoff
        ]
    else:
        results = get_results(url=url, db_path=db_path)

    total_checks = len(results)
    successful = [result for result in results if result["is_available"]]

    response_times = sorted(result["response_time_ms"] for result in successful)
    successful_count = len(successful)

    availability_rate = (successful_count / total_checks) * 100 if total_checks else 0.0

    if response_times:
        average_response_time = sum(response_times) / len(response_times)
        p95_index = ceil(0.95 * len(response_times)) - 1
        p95_response_time = response_times[p95_index]
    else:
        average_response_time = None
        p95_response_time = None

    return {
        "url": url,
        "total_checks": total_checks,
        "successful_checks": successful_count,
        "failed_checks": total_checks - successful_count,
        "availability_rate_percent": round(availability_rate, 2),
        "average_response_time_ms": (
            round(average_response_time, 2)
            if average_response_time is not None
            else None
        ),
        "p95_response_time_ms": p95_response_time,
    }
