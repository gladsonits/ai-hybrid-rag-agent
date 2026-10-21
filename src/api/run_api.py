"""Serve the chat API with uvicorn.

Usage (from the repo root):

    uv run python src/api/run_api.py
    uv run python src/api/run_api.py --port 8080 --reload
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the chat API.")
    parser.add_argument(
        "--host",
        default=os.getenv("API_HOST", "127.0.0.1"),
        help="address to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="port to bind (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="reload on source changes (development only)",
    )
    args = parser.parse_args()

    import uvicorn

    print(f"serving the chat API on http://{args.host}:{args.port}")
    print("  POST /chat   - ask a question (SSE)")
    print("  GET  /health  - liveness check")
    print("  GET  /docs    - interactive API docs")

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
