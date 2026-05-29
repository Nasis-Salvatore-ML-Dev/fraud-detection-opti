"""Check p99 latency against a staging endpoint using concurrent requests.

Usage:
    python scripts/latency_check.py --endpoint URL [--requests N] [--p99-limit MS]

Exit codes:
    0 — p99 latency <= --p99-limit
    1 — p99 latency  > --p99-limit
"""

import argparse
import asyncio
import logging
import sys
import time

import httpx
import numpy as np

log = logging.getLogger(__name__)

_SAMPLE_PAYLOAD = {
    "time": 1000.0,
    "amount": 149.62,
    "v1": -1.3598, "v2": -0.0728, "v3": 2.5363, "v4": 1.3782,
    "v5": -0.3383, "v6": 0.4624, "v7": 0.2396, "v8": 0.0987,
    "v9": 0.3638, "v10": 0.0908, "v11": -0.5516, "v12": -0.6178,
    "v13": -0.9914, "v14": -0.3112, "v15": 1.4682, "v16": -0.4704,
    "v17": 0.2076, "v18": 0.0258, "v19": 0.4036, "v20": 0.2514,
    "v21": -0.0183, "v22": 0.2779, "v23": -0.1105, "v24": 0.0669,
    "v25": 0.1285, "v26": -0.1891, "v27": 0.1336, "v28": -0.0210,
}


def compute_p99(times_ms: list[float]) -> float:
    """Return the 99th-percentile of times_ms using linear interpolation."""
    return float(np.percentile(times_ms, 99))


async def _send_one(client: httpx.AsyncClient, endpoint: str) -> float:
    """POST /predict and return elapsed time in milliseconds."""
    t0 = time.perf_counter()
    await client.post(f"{endpoint}/predict", json=_SAMPLE_PAYLOAD)
    return (time.perf_counter() - t0) * 1000.0


async def _gather_times(endpoint: str, n_requests: int) -> list[float]:
    """Send n_requests concurrent requests; return list of elapsed times (ms)."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        results = await asyncio.gather(
            *[_send_one(client, endpoint) for _ in range(n_requests)]
        )
    return list(results)


async def run_latency_check(
    endpoint: str,
    n_requests: int,
    p99_limit_ms: float,
) -> int:
    """Run the latency check and return 0 (pass) or 1 (fail)."""
    times_ms = await _gather_times(endpoint, n_requests)

    p50 = float(np.percentile(times_ms, 50))
    p90 = float(np.percentile(times_ms, 90))
    p99 = compute_p99(times_ms)

    log.info(
        "Latency over %d requests — p50=%.1fms  p90=%.1fms  p99=%.1fms  (limit=%.1fms)",
        n_requests, p50, p90, p99, p99_limit_ms,
    )

    if p99 > p99_limit_ms:
        log.error("p99 %.1fms exceeds limit %.1fms — FAIL", p99, p99_limit_ms)
        return 1

    log.info("p99 %.1fms within limit %.1fms — PASS", p99, p99_limit_ms)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check p99 latency against a staging endpoint."
    )
    parser.add_argument("--endpoint", required=True, help="Base URL to test.")
    parser.add_argument(
        "--requests", type=int, default=20, help="Number of concurrent requests."
    )
    parser.add_argument(
        "--p99-limit", type=float, default=50.0, dest="p99_limit",
        help="Maximum acceptable p99 latency in milliseconds (default: 50).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    _args = parse_args()
    sys.exit(asyncio.run(run_latency_check(_args.endpoint, _args.requests, _args.p99_limit)))
