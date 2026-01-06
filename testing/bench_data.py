def generate_docs(n: int = 10_000):
    return [
        f"Document {i} about retrieval augmented generation and vector databases."
        for i in range(n)
    ]
