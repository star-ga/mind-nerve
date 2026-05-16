"""Public dataclasses for the mind-nerve API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Route:
    """A single routing candidate returned by `route()`."""
    id: str
    name: str
    kind: str
    score: float
    source_repo: str
    url: str | None = None

    def as_dict(self) -> dict[str, Any]:
        d = {"id": self.id, "name": self.name, "kind": self.kind,
             "score": round(self.score, 6), "source_repo": self.source_repo}
        if self.url:
            d["url"] = self.url
        return d


@dataclass(frozen=True)
class RouteResult:
    """Result of one routing call."""
    query: str
    top_k: int
    routes: list[Route]
    encode_ms: float
    rank_ms: float
    catalog_size: int
    catalog_version: str
    model_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "routes": [r.as_dict() for r in self.routes],
            "encode_ms": round(self.encode_ms, 3),
            "rank_ms": round(self.rank_ms, 3),
            "catalog_size": self.catalog_size,
            "catalog_version": self.catalog_version,
            "model_version": self.model_version,
        }
