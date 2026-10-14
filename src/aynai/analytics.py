
from pathlib import Path
from math import ceil

from aynai.storage import DEFAULT_DB_PATH, get_results


def summarize_results(
    url: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict:
    """Calculate monitoring statistics from stored results."""
    results = get_results(url=url, db_path=db_path)

    total_checks = len(results)
    successful = [
        result
        for result in results
        if result["is_available"]
    ]

    response_times = sorted(
        result["response_time_ms"]
        for result in successful
    )

    successful_count = len(successful)

    if total_checks == 0:
        availability_rate = 0.0
    else:
        availability_rate = (
            successful_count / total_checks
        ) * 100

    if response_times:
        average_response_time = (
            sum(response_times) / len(response_times)
        )
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
        "availability_rate_percent": round(
            availability_rate, 2
        ),
        "average_response_time_ms": (
            round(average_response_time, 2)
            if average_response_time is not None
            else None
        ),
        "p95_response_time_ms": p95_response_time,
    }
