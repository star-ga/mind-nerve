"""mind-nerve — intent-classification preselector for agent runtimes.

Open the library, hide the cost. Given a user request and a catalog of
skills/tools/agents, return the top-K most relevant ones — so the
downstream LLM never sees the full library in its system prompt.

Public API:

    from mind_nerve import route, RouteResult, load_default_runtime
    result: RouteResult = route("git status", top_k=5)
    for r in result.routes:
        print(r.score, r.name, r.kind)
"""

from .inference import load_default_runtime, route
from .types import Route, RouteResult

__version__ = "0.1.0b2"
__all__ = ["Route", "RouteResult", "route", "load_default_runtime", "__version__"]
