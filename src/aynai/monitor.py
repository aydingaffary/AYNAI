
from time import perf_counter

from aynai.probes.http_probe import check_http


def monitor_services(urls: list[str]) -> list[dict]:
    """
    Check multiple HTTP services and collect their results.

    Unexpected errors from one service are recorded so that
    monitoring can continue for the remaining services.
    """
    results = []

    for url in urls:
        try:
            result = check_http(url)
        except Exception as error:
            result = {
                "url": url,
                "is_available": False,
                "status_code": None,
                "response_time_ms": 0.0,
                "error": (
                    f"Unexpected error: "
                    f"{type(error).__name__}: {error}"
                ),
            }

        results.append(result)

    return results
