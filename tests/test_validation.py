import pytest
from scara_core.validators import validate_vector, validate_batch
from scara_core.errors import ValidationError

def test_validate_vector_valid():
    validate_vector([0.1, 0.2, 0.3])

def test_validate_vector_empty():
    with pytest.raises(ValidationError, match="Vector is empty"):
        validate_vector([])

def test_validate_vector_invalid_type():
    with pytest.raises(ValidationError, match="Vector must contain only numbers"):
        validate_vector([0.1, "string"])

def test_validate_batch_valid():
    validate_batch([[0.1], [0.2]])

def test_validate_batch_empty():
    with pytest.raises(ValidationError, match="Empty vector batch"):
        validate_batch([])

def test_validate_batch_mismatch():
    with pytest.raises(ValidationError, match="All vectors must have same dimension"):
        validate_batch([[0.1], [0.1, 0.2]])
