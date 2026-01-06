# Benchmark Results

## Environment
- **Platform**: Python 3.12, Qdrant Client 1.16.2
- **Hardware**: CPU (In-Memory Qdrant)
- **Dataset**: 10,000 synthetic documents
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Frameworks**: Scaraflow 0.1.0, LangChain (latest), LlamaIndex (latest)

## Results

| Framework | Indexing Time (s) | Avg Latency (ms) | P95 Latency (ms) | Std Dev (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Scaraflow** | 10.05 | **78.67** | 142.76 | 50.58 |
| LangChain | 9.56 | 88.15 | 147.07 | 48.24 |
| LlamaIndex | **3.98** | 83.86 | **140.14** | 51.27 |

> **Note**: Indexing time includes vector upsert to Qdrant (in-memory). Embeddings were pre-computed to isolate infrastructure performance. LlamaIndex indexing is significantly faster, likely due to larger default batch sizes or optimized bulk insertion. Scaraflow provides the lowest average query latency.

## Analysis

1.  **Scaraflow** demonstrates the **lowest average retrieval latency** (78ms), validating its "retrieval-first" design.
2.  **LlamaIndex** is extremely efficient at **indexing**, processing 10k documents in <4 seconds.
3.  **LangChain** shows slightly higher overhead in retrieval compared to others.
4.  All frameworks show similar variance (Std Dev ~50ms), suggesting the underlying Qdrant search is the dominant factor, but framework overhead adds ~5-10ms per query.

## Reproduction

Run the benchmark script:

```bash
python benchmark_compare.py
```
