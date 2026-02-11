# -----------------------------------------
# Disable HF tokenizer fork warning (MUST be first)
# -----------------------------------------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import io
try:
    import requests
except ImportError:
    requests = None

from datetime import timedelta
from typing import List
import random

from qdrant_client import QdrantClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_live.indexer import LiveIndexer
from scara_live.engine import LiveRAGEngine
from scara_core.protocols import Embedder, LLM

# -------------------------------------------------
# Embedders (Real + Mock)
# -------------------------------------------------

class RealEmbedder(Embedder):
    def __init__(self):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()


class MockEmbedder(Embedder):
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        # Return deterministic random vector based on text length to be somewhat consistent
        random.seed(len(text))
        return [random.random() for _ in range(self.dim)]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


# -------------------------------------------------
# LLMs (Real + Mock)
# -------------------------------------------------

class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        if OpenAI is None:
            raise ImportError("openai not installed")
        self.client = OpenAI()
        self.model = model

    def __call__(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()


class MockLLM(LLM):
    def __call__(self, prompt: str) -> str:
        return "This is a MOCK response from the LLM. The system is running in demo mode without external dependencies."


# -------------------------------------------------
# GDELT GKG INGESTION (STABLE)
# -------------------------------------------------
def ingest_gdelt_gkg(indexer: LiveIndexer, query_keywords: list[str]):
    print("\nIngesting GDELT GKG (Global Knowledge Graph)")

    if requests is None:
        print("⚠️  Requests module not found. Using Mock Data ingestion.")
        _ingest_mock_data(indexer)
        return

    # GDELT legacy endpoints (HTTP FIRST)
    master_urls = [
        "http://data.gdeltproject.org/gkg/masterfilelist.txt",   # preferred
        "https://data.gdeltproject.org/gkg/masterfilelist.txt", # fallback
    ]

    response = None
    for url in master_urls:
        try:
            response = requests.get(url, timeout=20, verify=False)
            if response.status_code == 200:
                break
        except requests.RequestException:
            continue

    if response is None or response.status_code != 200:
        print("⚠️  Could not reach GDELT master file list (network/TLS issue). using Mock Data.")
        _ingest_mock_data(indexer)
        return

    lines = response.text.strip().splitlines()

    gkg_files = [
        line.split(" ")[2]
        for line in lines
        if line.endswith(".gkg.csv")
    ]

    if not gkg_files:
        print("⚠️  No GKG files found")
        return

    latest_gkg_url = gkg_files[-1]
    print(f"Using GKG file: {latest_gkg_url}")

    # Download GKG CSV (again with TLS fallback)
    try:
        r = requests.get(latest_gkg_url, timeout=30, verify=False)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️  Failed to download GKG file: {e}")
        return

    reader = csv.reader(io.StringIO(r.text), delimiter="\t")

    matched = 0

    for row in reader:
        if len(row) < 9:
            continue

        title = row[4]
        themes = row[7].lower()

        if any(k.lower() in title.lower() or k.lower() in themes for k in query_keywords):
            indexer.index(
                text=title,
                metadata={
                    "source": "gdelt-gkg",
                    "themes": row[7],
                    "url": row[5],
                },
            )
            matched += 1

        if matched >= 15:
            break

    print(f"Indexed {matched} relevant GDELT events")


def _ingest_mock_data(indexer: LiveIndexer):
    """Fallback ingestion if network is unavailable."""
    mock_events = [
        "India's economy grows by 7.5% in Q3 due to strong manufacturing output.",
        "RBI keeps repo rate unchanged at 6.5% citing inflation concerns.",
        "Stock market hits all-time high as tech stocks rally.",
        "New trade policies to boost exports from India.",
        "Global market trends show positive momentum for emerging economies."
    ]
    
    for event in mock_events:
        indexer.index(
            text=event,
            metadata={
                "source": "mock-data",
                "themes": "economy;market",
            },
        )
    print(f"Indexed {len(mock_events)} mock events.")

# -------------------------------------------------
# Main Demo
# -------------------------------------------------

def main():
    print("\n--- LiveRAG REAL-WORLD DEMO (GDELT GKG) ---")

    # Determine availability
    has_openai = OpenAI is not None and "OPENAI_API_KEY" in os.environ
    has_sbert = SentenceTransformer is not None

    print(f"Environment: OpenAI={has_openai}, SentenceTransformers={has_sbert}, Requests={requests is not None}")

    # Vector store
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(
        QdrantConfig(collection="live_world", vector_dim=384),
        client=client,
    )

    if has_sbert:
        print("Using RealEmbedder (all-MiniLM-L6-v2)")
        embedder = RealEmbedder()
    else:
        print("Using MockEmbedder (Random Vectors)")
        embedder = MockEmbedder()

    indexer = LiveIndexer(embedder=embedder, store=store)

    # REAL WORLD INGESTION (STABLE)
    ingest_gdelt_gkg(
        indexer,
        query_keywords=["india", "economy", "rbi", "market", "stock"],
    )

    if has_openai:
        print("Using OpenAILLM")
        llm = OpenAILLM()
    else:
        print("Using MockLLM")
        llm = MockLLM()

    # LiveRAG
    live_rag = LiveRAGEngine(
        embedder=embedder,
        store=store,
        llm=llm,
    )

    question = "What breaking financial news just happened?"

    print("\nQuestion:")
    print(question)

    response = live_rag.query(
        question,
        window=timedelta(minutes=180),
    )

    print("\nAnswer:")
    print(response.answer)

    print("\nContext Used:")
    for block in response.context:
        print("-", block.content)

    print("\nSUCCESS: GDELT GKG → LiveRAG → grounded answer")


if __name__ == "__main__":
    main()
