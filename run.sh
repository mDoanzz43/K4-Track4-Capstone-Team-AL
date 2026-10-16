#!/usr/bin/env bash
# Start the web app on http://localhost:8000
cd "$(dirname "$0")"
exec python -m uvicorn backend.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"
