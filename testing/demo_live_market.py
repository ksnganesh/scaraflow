import sys
import time
import json
import threading
import signal
from datetime import timedelta
from typing import Any, List
from unittest.mock import MagicMock, patch

from qdrant_client import QdrantClient
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_live.indexer import LiveIndexer
from scara_live.engine import LiveRAGEngine
from scara_live.adapters.kafka import KafkaLiveAdapter
from scara_core.protocols import Embedder, LLM

# --- Mocks and Helpers ---

class MockEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        # Return a dummy vector of dimension 384 (common for all-MiniLM-L6-v2)
        return [0.1] * 384

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

class MockLLM(LLM):
    def __call__(self, prompt: str) -> str:
        # Simple mock that checks if the context contains the relevant info
        if "RBI announces surprise rate decision" in prompt:
            return "Based on the latest news, RBI has announced a surprise rate decision."
        return "I don't know."

class MockMessage:
    def __init__(self, value_dict):
        self._value = json.dumps(value_dict).encode("utf-8")
        self._topic = "market_news"
        self._partition = 0
        self._offset = 0

    def value(self):
        return self._value

    def topic(self):
        return self._topic

    def partition(self):
        return self._partition

    def offset(self):
        self._offset += 1
        return self._offset - 1

    def error(self):
        return None

# --- Main Demo ---

def main():
    print("--- Starting ScaraLive Market Intelligence Demo ---")

    # 1. Setup Infrastructure
    print("[1] Initializing Vector Store (In-Memory)...")
    client = QdrantClient(":memory:")

    config = QdrantConfig(
        url=":memory:",
        collection="scara_vectors",
        vector_dim=384,
    )

    store = QdrantVectorStore(
        config=config,
        client=client,
    )

    embedder = MockEmbedder()

    # 2. Setup Indexer
    print("[2] Setting up LiveIndexer...")
    indexer = LiveIndexer(embedder=embedder, store=store)

    # 3. Setup Kafka Adapter with Mock Consumer
    print("[3] Setting up Kafka Adapter...")

    # Payload from user
    kafka_payload = {
        "text": "Breaking: RBI announces surprise rate decision",
        "source": "news",
        "symbol": "NIFTY"
    }

    # Mocking Consumer and signal
    with patch("scara_live.adapters.kafka.Consumer") as MockConsumerCls, \
         patch("signal.signal") as mock_signal:

        mock_consumer_instance = MagicMock()
        MockConsumerCls.return_value = mock_consumer_instance

        # Setup polling behavior: return message once, then return None (simulating no more messages)
        # We need to be careful because the adapter loop runs until self.running is False.
        # We can make poll return the message, then sleep/wait, or we can run adapter in a separate thread and stop it.

        mock_consumer_instance.poll.side_effect = [
            MockMessage(kafka_payload),
            None, None, None, None # Return None subsequently
        ]

        adapter = KafkaLiveAdapter(
            indexer=indexer,
            bootstrap_servers="mock:9092",
            topic="market_news",
            group_id="demo_group"
        )

        # Run adapter in a thread so we can stop it
        adapter_thread = threading.Thread(target=adapter.start)
        adapter_thread.start()

        print("    -> Adapter started, consuming messages...")
        time.sleep(2) # Give it time to process

        print("    -> Stopping adapter...")
        adapter._shutdown()
        adapter_thread.join()

    # 4. Setup RAG Engine
    print("[4] Setting up LiveRAG Engine...")
    llm = MockLLM()
    live_rag = LiveRAGEngine(
        embedder=embedder,
        store=store,
        llm=llm
    )

    # 5. Query
    print("[5] Executing Query...")
    question = "What just happened in the Indian markets?"
    print(f"    Question: {question}")

    response = live_rag.query(
        question,
        window=timedelta(minutes=15),
    )

    print("\n[Result]")
    print(f"Answer: {response.answer}")

    # Verification
    if "RBI" in response.answer:
        print("\nSUCCESS: Relevant news retrieved and answered.")
    else:
        print("\nFAILURE: Did not retrieve relevant news.")
        sys.exit(1)

if __name__ == "__main__":
    main()
