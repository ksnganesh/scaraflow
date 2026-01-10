from .errors import ValidationError
from .types import Vector
import numpy as np

_NUMERIC_TYPES = (int, float)

def validate_vector(vec: Vector) -> None:
    if not vec:
        raise ValidationError("Vector is empty")

    if not hasattr(vec, "__len__"):
        raise ValidationError("Vector must be a sequence")

    if not all(isinstance(x, _NUMERIC_TYPES) for x in vec):
        raise ValidationError("Vector must contain only numbers")


def validate_batch(vectors: list[Vector]) -> None:
    if not vectors:
        raise ValidationError("Empty vector batch")

    try:
        # This one line checks:
        # 1. If all elements are numbers (via dtype)
        # 2. If all rows have the same length (via array creation)
        arr = np.array(vectors, dtype=float)
    except (ValueError, TypeError):
        raise ValidationError("Batch must contain only numbers and all vectors must have same dimension")

    if arr.ndim != 2:
        raise ValidationError("Invalid vector shape")
