import os
import sys
import time
import uuid
import random
from datetime import datetime, timezone, timedelta
from typing import List, Generator

# Ensure scaraflow is in path if running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from scara_core.protocols import Embedder, LLM
from scara_core.types import Vector
from scara_index.qdrant_store import QdrantVectorStore, QdrantConfig
from scara_live.indexer import LiveIndexer
from scara_live.engine import LiveRAGEngine
from scara_live.config import LiveConfig

# --- Implementations ---

class HuggingFaceEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # This will download the model if not present
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> Vector:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[Vector]:
        return self.model.encode(texts).tolist()

class OpenAILLM(LLM):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI: {e}"

class DummyLLM(LLM):
    """Fallback if no API key is provided."""
    def __call__(self, prompt: str) -> str:
        return f"Simulated LLM response. Received prompt length: {len(prompt)}"

# --- Data Generator ---

def news_generator() -> Generator[str, None, None]:
    """Generates simulated financial news."""
    topics = ["Crypto", "Stock", "Forex", "Commodity"]
    actions = ["surges", "plummets", "remains stable", "shows volatility"]
    reasons = ["due to market sentiment", "following new regulations", "after earnings report", "amidst global uncertainty"]

    while True:
        topic = random.choice(topics)
        action = random.choice(actions)
        reason = random.choice(reasons)
        value = random.uniform(100, 50000)
        timestamp = datetime.now().strftime("%H:%M:%S")
        yield f"[{timestamp}] {topic} market {action} at ${value:.2f} {reason}."

# --- Main ---

def main():
    print("Initializing Live RAG Demo...")

    # 1. Setup Embedder
    # all-MiniLM-L6-v2 produces 384-dimensional vectors
    embedder = HuggingFaceEmbedder("all-MiniLM-L6-v2")

    # 2. Setup Vector Store (In-Memory Qdrant)
    print("Setting up Qdrant (in-memory)...")
    client = QdrantClient(":memory:")

    # Configure for 384 dimensions to match the model
    config = QdrantConfig(
        url=":memory:",
        collection="live_demo",
        vector_dim=384,
    )

    store = QdrantVectorStore(
        config=config,
        client=client
    )

    # 3. Setup LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print("Using OpenAI LLM.")
        llm = OpenAILLM(api_key)
    else:
        print("OPENAI_API_KEY not found. Using Dummy LLM for demonstration.")
        llm = DummyLLM()

    # 4. Setup Live Components
    # LiveConfig defaults: timestamp_field="ts", default_window=timedelta(minutes=5)
    indexer = LiveIndexer(embedder=embedder, store=store)
    rag_engine = LiveRAGEngine(
        embedder=embedder,
        store=store,
        llm=llm,
        config=LiveConfig(default_window=timedelta(seconds=60))
    )

    # 5. Simulation Loop
    print("\nStarting simulation loop. Press Ctrl+C to stop.\n")
    gen = news_generator()

    try:
        # We will index a few items, then query
        for i in range(1, 16): # Run for 15 iterations
            # Ingest Data
            news_item = next(gen)
            doc_id = str(uuid.uuid4())
            print(f"Indexing: {news_item}")
            indexer.index(doc_id=doc_id, text=news_item)

            # Query every 3 iterations
            if i % 3 == 0:
                print("\n--- Querying Live RAG ---")
                question = "What is the latest market trend?"

                # Retrieve from last 10 seconds to show "live" aspect
                # This ensures we get only the very latest indexed items
                response = rag_engine.query(
                    question,
                    window=timedelta(seconds=10),
                    top_k=3
                )

                print(f"Question: {question}")
                print(f"Answer: {response.answer}")
                print(f"Context used: {len(response.context)} documents")
                for doc in response.context:
                    content = doc.content
                    score = doc.score
                    print(f" - {content} (score: {score:.4f})")
                print("-------------------------\n")

            time.sleep(1) # Simulate real-time delay

    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == "__main__":
    main()
