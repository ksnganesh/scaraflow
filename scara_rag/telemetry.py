from typing import Any


def build_metadata(
    *,
    model: str | None = None,
    latency_ms: int | None = None,
    policy_version: str = "v1",
) -> dict[str, Any]:
    return {
        "model": model,
        "latency_ms": latency_ms,
        "policy_version": policy_version,
    }