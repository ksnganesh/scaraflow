# ARCHITECTURE

This document describes the internal architecture of Scaraflow.

## High-Level Goal

Scaraflow is designed as **retrieval infrastructure**, not an orchestration framework.

It prioritizes:
- Determinism
- Predictability
- Explicit contracts
- Streaming readiness

## Module Overview

```
scaraflow/
├── scara_core
├── scara_index
├── scara_rag
├── scara_live (planned)
├── scara_graph (planned)
```

## scara_core

The foundation of the system.

Responsibilities:
- Define all public contracts
- Provide shared types and errors
- Enforce invariants

No module may violate `scara_core` contracts.

## scara_index

Vector store implementations.

Current backend:
- Qdrant (Rust HNSW)

Responsibilities:
- Vector persistence
- ANN search
- Payload filtering

No retrieval logic lives here.

## scara_rag

Deterministic RAG engine.

Responsibilities:
- Query embedding
- Retrieval execution
- Context assembly
- Prompt construction
- Answerability checks

No vector DB specifics live here.

## scara_live (Planned)

Streaming and temporal retrieval.

Planned features:
- Append-only ingestion
- Time-windowed search
- LiveRAG semantics

## scara_graph (Planned)

Graph-based retrieval.

Planned features:
- Entity-aware retrieval
- Relationship traversal
- Multi-hop grounding

## Design Constraints

- No hidden state
- No async magic
- No implicit retries
- No silent fallbacks

## Why This Architecture

Most frameworks mix:
- retrieval
- prompting
- orchestration

Scaraflow separates them cleanly, allowing:
- easier debugging
- stable performance
- production-grade behavior

## Summary

Scaraflow treats RAG as infrastructure.
Each module has one responsibility.
Contracts matter more than convenience.
