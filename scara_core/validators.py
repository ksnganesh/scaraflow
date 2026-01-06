from .errors import ValidationError
from .types import Vector


def validate_vector(vec: Vector) -> None:
    if not vec:
        raise ValidationError("Vector is empty")

    if not all(isinstance(x, (int, float)) for x in vec):
        raise ValidationError("Vector must contain only numbers")


def validate_batch(vectors: list[Vector]) -> None:
    if not vectors:
        raise ValidationError("Empty vector batch")

    dim = len(vectors[0])
    for v in vectors:
        validate_vector(v)
        if len(v) != dim:
            raise ValidationError("All vectors must have same dimension")
