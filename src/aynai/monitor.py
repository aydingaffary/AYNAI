from pathlib import Path
from aynai.anomaly import analyze_service_latency
from aynai.probes.http_probe import check_http
from aynai.storage import DEFAULT_DB_PATH, save_result


def monitor_services(
    urls: list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Check multiple HTTP services, analyze latency, and persist results.

    Monitoring continues if checking, analyzing, or saving one service fails.
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

        anomaly = None

        if result["is_available"]:
            try:
                anomaly = analyze_service_latency(
                    url=url,
                    current_response_time_ms=result["response_time_ms"],
                    db_path=db_path,
                )
            except Exception as error:
                print(
                    f"Warning: could not analyze latency for {url}: "
                    f"{type(error).__name__}: {error}"
                )

        if anomaly is not None:
            result["anomaly"] = anomaly

        try:
            result["record_id"] = save_result(
                result,
                db_path,
                anomaly=anomaly,
            )
        except Exception as error:
            print(
                f"Warning: could not save result for {url}: "
                f"{type(error).__name__}: {error}"
            )

        results.append(result)

    return results
