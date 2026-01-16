import json
import uuid
from typing import Callable, Dict, Any, Optional, List

from scara_live.indexer import LiveIndexer


class WebSocketLiveAdapter:
    """
    WebSocket → LiveIndexer adapter.

    Designed to be used inside:
    - FastAPI WebSocket routes
    - Starlette
    - custom WebSocket servers
    """

    def __init__(
        self,
        *,
        indexer: LiveIndexer,
        message_parser: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        batch_size: int = 50,
    ):
        self.indexer = indexer
        self.batch_size = batch_size
        self.buffer: List[Dict[str, Any]] = []
        self.message_parser = message_parser or self._default_parser

    # ----------------------------
    # Message parsing
    # ----------------------------

    def _default_parser(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected payload format:
        {
            "text": "...",
            "metadata": {...}   # optional
        }
        """
        return {
            "doc_id": payload.get("id") or str(uuid.uuid4()),
            "text": payload.get("text"),
            "metadata": payload.get("metadata", {}),
        }

    # ----------------------------
    # Public API
    # ----------------------------

    def handle_message(self, raw_message: str) -> None:
        """
        Accepts a raw WebSocket message (string).
        """
        try:
            payload = json.loads(raw_message)
        except Exception:
            return

        parsed = self.message_parser(payload)
        if not parsed or not parsed.get("text"):
            return

        self.buffer.append(parsed)

        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """
        Flush buffered messages into LiveIndexer.
        """
        if not self.buffer:
            return

        doc_ids = [item["doc_id"] for item in self.buffer]
        texts = [item["text"] for item in self.buffer]
        metadatas = [item.get("metadata", {}) for item in self.buffer]

        self.indexer.index_batch(
            doc_ids=doc_ids,
            texts=texts,
            metadatas=metadatas,
        )

        self.buffer.clear()

    def close(self) -> None:
        """
        Flush remaining messages on connection close.
        """
        self.flush()
