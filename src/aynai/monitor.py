
from pathlib import Path

from aynai.probes.http_probe import check_http
from aynai.storage import DEFAULT_DB_PATH, save_result


def monitor_services(
    urls: list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Check multiple HTTP services, collect results,
    and persist each result to SQLite.

    Monitoring continues if checking or saving one service fails.
    """
    results = []

    for url in urls:
        try:
            result = check_http(url)
        except Exception as error:
            import traceback

            print(f"\nUnexpected error while checking {url}:")
            traceback.print_exc()

            result = {
                "url": url,
                "is_available": False,
                "status_code": None,
                "response_time_ms": 0.0,
                "error": f"Unexpected error: {type(error).__name__}: {error}",
            }
        results.append(result)

        try:
            result["record_id"] = save_result(result, db_path)
        except Exception as error:
            print(
                f"Warning: could not save result for {url}: "
                f"{type(error).__name__}: {error}"
            )

    return results
