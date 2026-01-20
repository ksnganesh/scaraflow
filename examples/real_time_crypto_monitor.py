import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_live.indexer import LiveIndexer
from scara_live.engine import LiveRAGEngine
from scara_core.protocols import Embedder, LLM


# --- 1. Real Components ---

class LocalEmbedder(Embedder):
    """
    Real embedder using SentenceTransformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()


class NewsSummarizerLLM(LLM):
    """
    A simple deterministic LLM for demonstration purposes.
    It summarizes the retrieved context into a news bulletin format.
    """
    def __call__(self, prompt: str) -> str:
        # In a real app, this would call OpenAI/Anthropic.
        # Here we parse the prompt (which contains the context) to generate a response.

        # The default prompt format usually puts context blocks first.
        # We'll just look for lines starting with "Content:" or similar markers from the context.
        # For this demo, we assume the prompt contains the text of the relevant docs.

        lines = prompt.split('\n')
        relevant_info = []
        for line in lines:
            # Heuristic to find content from the prompt assembly
            if "ETH" in line or "BTC" in line or "Bitcoin" in line or "Ethereum" in line:
                 relevant_info.append(line.strip())

        if not relevant_info:
            return "No significant market events detected in the specified timeframe."

        summary = " ** MARKET ALERT ** \nBased on the latest data:\n"
        for info in relevant_info[:3]: # Top 3
            summary += f"- {info}\n"

        return summary


# --- 2. Simulation Logic ---

def run_simulation():
    print("\n=== Real-Time Crypto Market Monitor Simulation ===\n")

    # A. Setup Infrastructure
    # -----------------------
    client = QdrantClient(":memory:")
    config = QdrantConfig(
        url=":memory:",
        collection="crypto_live",
        vector_dim=384,
    )
    store = QdrantVectorStore(config=config, client=client)

    embedder = LocalEmbedder()
    llm = NewsSummarizerLLM()

    indexer = LiveIndexer(embedder=embedder, store=store)
    rag_engine = LiveRAGEngine(embedder=embedder, store=store, llm=llm)

    # B. Simulate Timeline
    # --------------------
    now = datetime.now(timezone.utc)

    events = [
        {
            "offset_mins": -60,
            "text": "Bitcoin is trading sideways at $60,000. Volume is low.",
            "source": "MarketWatch"
        },
        {
            "offset_mins": -45,
            "text": "Ethereum upgrade 'Pectra' scheduled for next month.",
            "source": "CryptoNews"
        },
        {
            "offset_mins": -5,
            "text": "BREAKING: Large sell-off detected in BTC. Price drops to $58,500.",
            "source": "WhaleAlert"
        },
        {
            "offset_mins": -2,
            "text": "Panic selling intensifies. BTC hits $57,000 on major exchanges.",
            "source": "Twitter/X"
        },
        {
            "offset_mins": -1,
            "text": "Analysts attribute crash to rumor of new regulatory crackdown.",
            "source": "Bloomberg"
        }
    ]

    print("--- Ingesting Stream ---")
    for event in events:
        ts = now + timedelta(minutes=event["offset_mins"])
        doc_id = str(uuid.uuid4())

        print(f"[{ts.strftime('%H:%M:%S')}] Ingesting: {event['text'][:50]}...")

        indexer.index(
            doc_id=doc_id,
            text=event["text"],
            metadata={"source": event["source"]},
            timestamp=ts
        )

    # C. Execute Queries
    # ------------------

    print("\n--- Analyst Queries ---")

    # Query 1: Broad context (last 2 hours)
    q1 = "What is the general market sentiment?"
    print(f"\nQ1: {q1} (Window: 2 hours)")
    resp1 = rag_engine.query(q1, window=timedelta(hours=2))
    print(f"A1: {resp1.answer}")

    # Query 2: Immediate "What just happened?" (last 10 minutes)
    q2 = "Why is Bitcoin crashing?"
    print(f"\nQ2: {q2} (Window: 10 minutes)")
    resp2 = rag_engine.query(q2, window=timedelta(minutes=10))
    print(f"A2: {resp2.answer}")

    # Query 3: Stale/Old news check (Window: 10 mins) for something that happened an hour ago
    q3 = "Any news on Ethereum upgrades?"
    print(f"\nQ3: {q3} (Window: 10 minutes)")
    resp3 = rag_engine.query(q3, window=timedelta(minutes=10))
    print(f"A3: {resp3.answer}")
    # Expectation: Should NOT find the ETH news from 45 mins ago because window is 10 mins.


if __name__ == "__main__":
    run_simulation()
