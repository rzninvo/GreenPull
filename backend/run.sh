#!/bin/bash
set -e

echo "=== GreenPull Backend ==="

# Check Redis
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Starting Redis..."
    redis-server --daemonize yes
fi

# Start RQ worker in background
echo "Starting RQ worker..."
cd "$(dirname "$0")"
PYTHONPATH=. rq worker --url redis://localhost:6379/0 &

# Start FastAPI
echo "Starting FastAPI server on http://localhost:8000"
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
