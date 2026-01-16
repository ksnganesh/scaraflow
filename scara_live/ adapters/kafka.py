import json
import signal
from typing import Callable, Dict, Any, Optional, List

from confluent_kafka import Consumer, KafkaError, Message

from scara_live.indexer import LiveIndexer


class KafkaLiveAdapter:
    """
    Kafka → LiveIndexer adapter.

    Responsibilities:
    - Consume Kafka messages
    - Extract text + metadata
    - Forward to LiveIndexer
    """

    def __init__(
        self,
        *,
        indexer: LiveIndexer,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        message_parser: Optional[Callable[[Message], Dict[str, Any]]] = None,
        poll_timeout: float = 1.0,
        batch_size: int = 100,
        auto_offset_reset: str = "earliest",
    ):
        self.indexer = indexer
        self.topic = topic
        self.batch_size = batch_size
        self.poll_timeout = poll_timeout
        self.running = False

        self.message_parser = message_parser or self._default_parser

        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "enable.auto.commit": False,
                "auto.offset.reset": auto_offset_reset,
            }
        )

    # ----------------------------
    # Message parsing
    # ----------------------------

    def _default_parser(self, msg: Message) -> Dict[str, Any]:
        """
        Default parser assumes JSON payload with `text` field.
        """
        try:
            payload = json.loads(msg.value().decode("utf-8"))
        except Exception:
            return {}

        return {
            "text": payload.get("text"),
            "metadata": payload,
            "doc_id": f"{msg.topic()}:{msg.partition()}:{msg.offset()}",
        }

    # ----------------------------
    # Lifecycle
    # ----------------------------

    def start(self) -> None:
        self.consumer.subscribe([self.topic])
        self.running = True

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        self._run_loop()

    def _shutdown(self, *_):
        self.running = False

    # ----------------------------
    # Main loop
    # ----------------------------

    def _run_loop(self) -> None:
        buffer: List[Dict[str, Any]] = []

        while self.running:
            msg = self.consumer.poll(self.poll_timeout)

            if msg is None:
                self._flush(buffer)
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    raise RuntimeError(msg.error())

            parsed = self.message_parser(msg)
            if not parsed or not parsed.get("text"):
                continue

            buffer.append(parsed)

            if len(buffer) >= self.batch_size:
                self._flush(buffer)

        # Final flush on shutdown
        self._flush(buffer)
        self.consumer.close()

    # ----------------------------
    # Flush to LiveIndexer
    # ----------------------------

    def _flush(self, buffer: List[Dict[str, Any]]) -> None:
        if not buffer:
            return

        doc_ids = [item["doc_id"] for item in buffer]
        texts = [item["text"] for item in buffer]
        metadatas = [item.get("metadata", {}) for item in buffer]

        self.indexer.index_batch(
            doc_ids=doc_ids,
            texts=texts,
            metadatas=metadatas,
        )

        self.consumer.commit(asynchronous=False)
        buffer.clear()
