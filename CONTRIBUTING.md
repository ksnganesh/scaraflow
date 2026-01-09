# CONTRIBUTING

Thank you for your interest in contributing to Scaraflow.

Scaraflow prioritizes **correctness, clarity, and stability** over feature velocity.

## Philosophy

- Retrieval is infrastructure
- Contracts are sacred
- No hidden behavior
- Determinism over cleverness

If you disagree with these principles, this may not be the right project.

## Development Setup

```bash
git clone https://github.com/your-org/scaraflow
cd scaraflow
python -m venv .venv
source .venv/bin/activate
pip install scaraflow
```

## Code Standards

- Explicit is better than implicit
- No global state
- No hidden retries
- No background threads
- Prefer pure functions

## What Contributions Are Welcome

- Bug fixes
- Performance improvements
- Documentation
- Benchmarks
- New VectorStore backends (contract-compliant)

## What Will Likely Be Rejected

- Agent frameworks
- Prompt DSLs
- Magic abstractions
- Implicit state
- Breaking core contracts

## Testing

All changes should:
- Include minimal tests or benchmarks
- Preserve public contracts
- Not degrade determinism

## Versioning Rules

- PATCH: bug fixes
- MINOR: new features (backward compatible)
- MAJOR: contract changes (strongly discouraged)

## Communication

Open an issue before large changes.
Design discussions are preferred over pull requests for major ideas.
