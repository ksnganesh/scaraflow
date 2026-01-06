# BENCHMARKS

This document describes how Scaraflow benchmarks are executed, interpreted, and reproduced.

## Goal

Scaraflow benchmarks aim to measure **retrieval infrastructure quality**, not prompt tricks or agent workflows.

Metrics focus on:
- Embedding throughput
- Indexing time
- Query latency (avg / p95)
- Latency variance
- Determinism

## What Is Benchmarked

- Vector indexing via Qdrant (Rust HNSW)
- Retrieval via `scara-rag`
- Deterministic query execution

## What Is NOT Benchmarked

- Agents
- Tool calling
- Prompt engineering
- Caching layers
- UI frameworks

## Environment

- CPU-only execution
- Python 3.9+
- Qdrant (Docker, local, or cloud)
- SentenceTransformers (`all-MiniLM-L6-v2`)

## Dataset

10,000 synthetic documents:

```
Document {i} about retrieval augmented generation and vector databases.
```

Chosen for:
- reproducibility
- no copyrighted content
- non-trivial semantic similarity

## Benchmark Phases

1. Embedding (batched)
2. Indexing
3. Query execution (100 identical queries)

Each phase is timed independently.

## Example Results

```
Embedding time: 3.45s
Index time:     2.11s
Avg latency:    16.9ms
P95 latency:    20.0ms
Std dev:        low
```

## Interpretation

- Embedding time depends on hardware
- Index time reflects vector DB performance
- Query latency reflects retrieval stability
- Low variance is a key success metric

## Reproducibility

To reproduce results:
- Use same embedding model
- Same batch size
- Same Qdrant version
- Disable GPU acceleration

Benchmarks are designed to be replayable, not marketing artifacts.
