class ScaraError(Exception):
    """Base exception for all Scaraflow components."""


class ValidationError(ScaraError):
    pass


class ConfigurationError(ScaraError):
    pass


class RetrievalError(ScaraError):
    pass
