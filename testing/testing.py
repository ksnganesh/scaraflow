# -----------------------------------------
# Disable HF tokenizer fork warning (MUST be first)
# -----------------------------------------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uuid
import csv
import io
import requests
from datetime import timedelta
from typing import List

from qdrant_client import QdrantClient
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_live.indexer import LiveIndexer
from scara_live.engine import LiveRAGEngine
from scara_core.protocols import Embedder, LLM


# -------------------------------------------------
# Real Embedder
# -------------------------------------------------

class RealEmbedder(Embedder):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()


# -------------------------------------------------
# Real LLM (OpenAI)
# -------------------------------------------------

class OpenAILLM(LLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def __call__(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()


# -------------------------------------------------
# GDELT GKG INGESTION (STABLE)
# -------------------------------------------------
def ingest_gdelt_gkg(indexer: LiveIndexer, query_keywords: list[str]):
    import csv, io, uuid, requests

    print("\nIngesting GDELT GKG (Global Knowledge Graph)")

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
        print("⚠️  Could not reach GDELT master file list (network/TLS issue)")
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
                doc_id=str(uuid.uuid4()),
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

# -------------------------------------------------
# Main Demo
# -------------------------------------------------

def main():
    print("\n--- LiveRAG REAL-WORLD DEMO (GDELT GKG) ---")

    # Vector store
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(
        QdrantConfig(collection="live_world", vector_dim=384),
        client=client,
    )

    embedder = RealEmbedder()
    indexer = LiveIndexer(embedder=embedder, store=store)

    # REAL WORLD INGESTION (STABLE)
    ingest_gdelt_gkg(
        indexer,
        query_keywords=["india", "economy", "rbi", "market", "stock"],
    )

    # LiveRAG
    live_rag = LiveRAGEngine(
        embedder=embedder,
        store=store,
        llm=OpenAILLM(),
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
