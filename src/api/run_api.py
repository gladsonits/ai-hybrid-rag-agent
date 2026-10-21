"""Serve the chat API with uvicorn.

Usage (from the repo root):

    uv run python src/api/run_api.py
    uv run python src/api/run_api.py --port 8080 --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import ApiSettings  # noqa: E402


def main() -> int:
    settings = ApiSettings.from_env()

    parser = argparse.ArgumentParser(description="Serve the chat API.")
    parser.add_argument(
        "--host",
        default=settings.host,
        help="address to bind (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="port to bind (default: %(default)s)",
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
