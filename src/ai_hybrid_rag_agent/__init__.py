"""The installed package: the console entry point.

``main`` starts the chat API, which is the user-facing part of this project.
The indexing and retrieval stages have their own CLIs
(``src/indexing/run_index.py`` and ``src/retrieval/run_search.py``).
"""

from __future__ import annotations


def main() -> None:
    """Serve the chat API with uvicorn."""
    import uvicorn

    print("serving the chat API on http://127.0.0.1:8000")
    print("  POST /chat   - ask a question (SSE)")
    print("  GET  /health  - liveness check")
    print("  GET  /docs    - interactive API docs")
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000)
