import time
from pathlib import Path
from collections.abc import Callable

from aynai.monitor import monitor_services
from aynai.storage import DEFAULT_DB_PATH


def run_periodic_monitoring(
    urls: list[str],
    interval_seconds: float = 30.0,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    max_cycles: int | None = None,
    monitor_fn: Callable = monitor_services,
    sleep_fn: Callable = time.sleep,
) -> int:
    """Monitor services periodically until stopped or cycles finish."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")

    if max_cycles is not None and max_cycles < 1:
        raise ValueError("max_cycles must be at least 1")

    cycles = 0

    try:
        while max_cycles is None or cycles < max_cycles:
            print(f"\nMonitoring cycle {cycles + 1}")

            monitor_fn(urls, db_path=db_path)
            cycles += 1

            if max_cycles is not None and cycles >= max_cycles:
                break

            print(f"Next check in {interval_seconds} seconds...")
            sleep_fn(interval_seconds)

    except KeyboardInterrupt:
        print("\nAYNAI monitoring stopped by user.")

    return cycles