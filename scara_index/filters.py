from qdrant_client.models import FieldCondition, MatchValue, Filter


def eq(field: str, value) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key=field,
                match=MatchValue(value=value),
            )
        ]
    )
