from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class LiveConfig:
    """
    Configuration for LiveRAG behavior.
    """
    timestamp_field: str = "ts"
    default_window: timedelta = timedelta(minutes=5)
    max_context_chars: int = 4000
