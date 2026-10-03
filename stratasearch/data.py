"""Neutral document schema and bounded public input parsing."""

import json
import math
import re
from pathlib import Path

from .types import Filter, QueryConfig

MAX_DOCUMENTS = 1000
MAX_JSON_BYTES = 5_000_000
FIELDS = {"title", "body", "tags", "kind", "topic", "collection", "year"}
ID = re.compile(r"^[a-z][a-z0-9-]{0,63}$")


def _text(value, name, maximum, empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise ValueError(f"{name} must be text with 1..{maximum} characters")
    return value.strip()


def parse_corpus(value):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_DOCUMENTS:
        raise ValueError(f"Corpus requires 1..{MAX_DOCUMENTS} documents")
    result = []
    ids = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != FIELDS | {"id"}:
            raise ValueError(
                "Document must contain exactly id, title, body, tags, kind, topic, collection, year"
            )
        doc_id = row["id"]
        if not isinstance(doc_id, str) or not ID.fullmatch(doc_id) or doc_id in ids:
            raise ValueError("Document ids must be unique lowercase identifiers")
        if type(row["year"]) is not int or not 2000 <= row["year"] <= 2100:
            raise ValueError("Document year must be an integer in 2000..2100")
        tags = row["tags"]
        if not isinstance(tags, list) or len(tags) > 20:
            raise ValueError("Tags must be a list of up to 20 strings")
        clean = {"id": doc_id, "year": row["year"], "tags": [_text(t, "tag", 40) for t in tags]}
        for key, maximum in [
            ("title", 160),
            ("body", 4000),
            ("kind", 40),
            ("topic", 40),
            ("collection", 40),
        ]:
            clean[key] = _text(row[key], key, maximum)
        ids.add(doc_id)
        result.append(clean)
    return result


def parse_filter(value):
    if not isinstance(value, dict) or set(value) != {"field", "op", "value"}:
        raise ValueError("Filter requires exactly field, op, value")
    name, op, item = value["field"], value["op"], value["value"]
    if (
        not isinstance(name, str)
        or not isinstance(op, str)
        or name not in FIELDS
        or op not in {"Eq", "NotEq", "Contains", "In", "NotIn", "Gte", "Lte"}
    ):
        raise ValueError("Unsupported filter field or operator")
    if name == "tags" and op not in {"Contains", "In", "NotIn"}:
        raise ValueError("Tags require Contains, In or NotIn")
    if name != "tags" and op == "Contains":
        raise ValueError("Contains is restricted to tags")
    if op in {"Gte", "Lte"} and name != "year":
        raise ValueError("Numeric comparisons are restricted to year")
    items = item if op in {"In", "NotIn"} else [item]
    if not isinstance(items, list) or not 1 <= len(items) <= 20:
        raise ValueError("In and NotIn require a nonempty bounded list")
    for part in items:
        if name == "year":
            if (
                type(part) not in {int, float}
                or not math.isfinite(part)
                or not 2000 <= part <= 2100
            ):
                raise ValueError("Year filter requires finite numbers in 2000..2100")
        else:
            _text(part, "filter value", 160)
    return Filter(name, op, item)


def parse_query(value):
    if not isinstance(value, dict) or set(value) - {
        "query_id",
        "description",
        "hard_criteria",
        "soft_criteria",
        "filters",
    }:
        raise ValueError("Unknown query keys or invalid query object")
    query_id = value.get("query_id")
    if not isinstance(query_id, str) or not ID.fullmatch(query_id):
        raise ValueError("query_id must be a lowercase identifier, not a path")
    description = _text(value.get("description"), "description", 2000)
    criteria = {}
    for key in ["hard_criteria", "soft_criteria"]:
        entries = value.get(key, [])
        if not isinstance(entries, list) or len(entries) > 20:
            raise ValueError(f"{key} requires a list of up to 20 criteria")
        criteria[key] = [_text(c, key, 240) for c in entries]
    filters = value.get("filters", [])
    if not isinstance(filters, list) or len(filters) > 20:
        raise ValueError("Filters require a list of up to 20 predicates")
    return QueryConfig(
        query_id,
        description,
        criteria["hard_criteria"],
        criteria["soft_criteria"],
        [parse_filter(f) for f in filters],
    )


def read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON input exceeds the 5 MB limit")

    def invalid_number(text):
        raise ValueError(f"JSON must not contain {text}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON object contains duplicate keys")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=invalid_number,
        object_pairs_hook=unique_keys,
    )


def load_corpus(path):
    return parse_corpus(read_json(path))
