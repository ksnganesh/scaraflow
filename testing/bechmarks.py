import time
import uuid
import numpy as np
import warnings
from typing import List, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Scaraflow imports
from scara_index.qdrant_store import QdrantVectorStore
from scara_index.config import QdrantConfig
from scara_rag.engine import RAGEngine
from scara_rag.policies import RetrievalPolicy

# LangChain imports
from langchain_qdrant import QdrantVectorStore as Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, Document as LIDocument, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore as LIQdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
NUM_DOCS = 10000
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
NUM_QUERIES = 100

print(f"Generating {NUM_DOCS} synthetic documents...")

def generate_documents(n: int) -> List[str]:
    """Generates synthetic documents."""
    docs = []
    base_texts = [
        "Retrieval Augmented Generation is a technique.",
        "Vector databases like Qdrant are efficient.",
        "Scaraflow prioritizes deterministic retrieval.",
        "LangChain is a popular orchestration framework.",
        "LlamaIndex focuses on data ingestion and indexing.",
        "Python is great for AI and Machine Learning.",
        "Production systems need low latency.",
        "Embeddings represent semantic meaning.",
        "Search quality depends on the embedding model.",
        "Benchmarks help compare infrastructure."
    ]
    for i in range(n):
        text = f"{base_texts[i % len(base_texts)]} (ID: {i})"
        docs.append(text)
    return docs

texts = generate_documents(NUM_DOCS)

# Pre-compute embeddings to separate embedding time from indexing time for fair comparison.
print("Benchmarking Embedding Time (SentenceTransformer)...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

start_time = time.time()
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
embedding_time = time.time() - start_time
print(f"Embedding Time: {embedding_time:.4f}s ({len(texts)/embedding_time:.1f} docs/s)")

# Generate queries
queries = [f"Query about {texts[i % len(texts)].split()[0]}" for i in range(NUM_QUERIES)]
# For Scaraflow, RAGEngine uses embedder.embed(query). We'll measure the full path including query embedding.
# But for latency comparison, we should be consistent.
# Scaraflow RAGEngine will embed the query.
# LangChain wrapper usually embeds the query.
# LlamaIndex Retriever embeds the query.
# So we just pass strings.

# Store results
results = {}

def measure_latency(func, queries, name):
    latencies = []
    # Warmup
    for q in queries[:5]:
        func(q)
    
    for q in queries:
        start = time.perf_counter()
        func(q)
        latencies.append((time.perf_counter() - start) * 1000) # ms
    
    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    std_dev = np.std(latencies)
    print(f"[{name}] Avg: {avg_lat:.2f}ms, P95: {p95_lat:.2f}ms, Std: {std_dev:.2f}ms")
    return avg_lat, p95_lat, std_dev

# --- Scaraflow Benchmark ---
print("\n--- Scaraflow Benchmark ---")
# Use in-memory Qdrant via QdrantConfig. 
# Note: QdrantClient(url=":memory:") works.
# But QdrantVectorStore creates QdrantClient(url=config.url).
# So we set url=":memory:".
config_scara = QdrantConfig(
    url=":memory:", 
    collection="scara_bench", 
    vector_dim=VECTOR_DIM,
    recreate=True
)
client_scara = QdrantClient(":memory:")
store_scara = QdrantVectorStore(config_scara, client=client_scara)

start_idx = time.time()
store_scara.upsert(
    ids=list(range(len(texts))),
    vectors=embeddings.tolist(),
    metadata=[{"text": t} for t in texts]
)
scara_index_time = time.time() - start_idx
print(f"Indexing Time: {scara_index_time:.4f}s")

# Setup RAG Engine (Mock LLM)
embedder_wrapper = type("E", (), {"embed": lambda t: model.encode(t).tolist()})
rag_scara = RAGEngine(
    embedder=embedder_wrapper,
    store=store_scara,
    llm=lambda prompt: "Mock Answer",
)

def query_scara(q):
    rag_scara.query(q, policy=RetrievalPolicy(top_k=5))

scara_metrics = measure_latency(query_scara, queries, "Scaraflow")
results['Scaraflow'] = {'index_time': scara_index_time, 'metrics': scara_metrics}


# --- LangChain Benchmark ---
print("\n--- LangChain Benchmark ---")
client_lc = QdrantClient(":memory:")
client_lc.recreate_collection(
    collection_name="langchain_bench",
    vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
)

embeddings_lc = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

start_idx = time.time()
# LangChain doesn't expose a clean "add_embeddings" method in the Qdrant wrapper publicly
# without using internal methods or from_embeddings which failed.
# To be fair and separate embedding from indexing, we will use the underlying client directly
# to populate the collection, which is what `Qdrant` wrapper wraps anyway.
# This makes the "Indexing Time" a benchmark of the Qdrant Client + Network/Memory,
# which is fair as we want to measure framework overhead in *Retrieval*.
# If we used add_texts, we'd be double-counting embedding time.

from qdrant_client.models import PointStruct
points = [
    PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": texts[i]})
    for i in range(len(texts))
]
# Batch upsert
batch_size = 64
for i in range(0, len(points), batch_size):
    client_lc.upsert(
        collection_name="langchain_bench",
        points=points[i : i + batch_size],
    )

vectorstore_lc = Qdrant(
    client=client_lc,
    collection_name="langchain_bench",
    embedding=embeddings_lc,
)
lc_index_time = time.time() - start_idx
print(f"Indexing Time: {lc_index_time:.4f}s")

def query_lc(q):
    vectorstore_lc.similarity_search(q, k=5)

lc_metrics = measure_latency(query_lc, queries, "LangChain")
results['LangChain'] = {'index_time': lc_index_time, 'metrics': lc_metrics}


# --- LlamaIndex Benchmark ---
print("\n--- LlamaIndex Benchmark ---")
client_li = QdrantClient(":memory:")
vector_store_li = LIQdrantVectorStore(client=client_li, collection_name="llamaindex_bench")

nodes = []
for i, text in enumerate(texts):
    # LlamaIndex requires UUID strings if not integers, "0" fails validation in Qdrant Local
    node = TextNode(text=text, id_=str(uuid.uuid4()))
    node.embedding = embeddings[i].tolist()
    nodes.append(node)

storage_context = StorageContext.from_defaults(vector_store=vector_store_li)

# Setup Embed Model globally or passed to index to avoid OpenAI error
embed_model_li = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
Settings.embed_model = embed_model_li
Settings.llm = None # Disable LLM to avoid OpenAI error there too

start_idx = time.time()
# VectorStoreIndex will use the embeddings already in nodes
index_li = VectorStoreIndex(nodes, storage_context=storage_context)
li_index_time = time.time() - start_idx
print(f"Indexing Time: {li_index_time:.4f}s")

# Retriever
retriever_li = index_li.as_retriever(similarity_top_k=5)

def query_li(q):
    retriever_li.retrieve(q)

li_metrics = measure_latency(query_li, queries, "LlamaIndex")
results['LlamaIndex'] = {'index_time': li_index_time, 'metrics': li_metrics}


# --- Summary ---
print("\n" + "="*70)
print(f"{'Framework':<15} | {'Index (s)':<10} | {'Avg Lat (ms)':<12} | {'P95 (ms)':<10} | {'Std (ms)':<10}")
print("-" * 70)

for name in ['Scaraflow', 'LangChain', 'LlamaIndex']:
    data = results[name]
    metrics = data['metrics']
    print(f"{name:<15} | {data['index_time']:<10.4f} | {metrics[0]:<12.2f} | {metrics[1]:<10.2f} | {metrics[2]:<10.2f}")
print("="*70)
print(f"Embedding Time (common): {embedding_time:.4f}s")
