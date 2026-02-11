from qdrant_client.models import FieldCondition, MatchValue, Filter, Range


def eq(field: str, value) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key=field,
                match=MatchValue(value=value),
            )
        ]
    )


def range_gte(field: str, gte: float) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key=field,
                range=Range(gte=gte),
            )
        ]
    )
