#!/bin/bash
# ==============================================================================
# TinyTransformer LM: Unified Launch & Process Manager
# Starts the PyTorch FastAPI inference engine and Vite React web dashboard
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

echo "============================================================"
echo "⚡ TinyTransformer LM — Launching Full-Stack Services"
echo "============================================================"

# Function to clean up background processes on exit
cleanup() {
  echo ""
  echo "🛑 Stopping TinyTransformer services..."
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [ -n "$FRONTEND_PID" ]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 1. Check and clear any existing stale processes on ports 8008 and 5173
OLD_BACKEND=$(lsof -t -i:8008 2>/dev/null || true)
if [ -n "$OLD_BACKEND" ]; then
  echo "⚠️ Port 8008 is in use (PID: $OLD_BACKEND). Clearing stale process..."
  kill -9 $OLD_BACKEND 2>/dev/null || true
  sleep 1
fi

OLD_FRONTEND=$(lsof -t -i:5173 2>/dev/null || true)
if [ -n "$OLD_FRONTEND" ]; then
  echo "⚠️ Port 5173 is in use (PID: $OLD_FRONTEND). Clearing stale process..."
  kill -9 $OLD_FRONTEND 2>/dev/null || true
  sleep 1
fi

# 2. Launch FastAPI Inference Server
echo "🚀 [1/2] Starting Backend Inference API on http://127.0.0.1:8008..."
cd "$BACKEND_DIR"
"$BACKEND_DIR/.venv/bin/uvicorn" server:app --host 127.0.0.1 --port 8008 --log-level warning &
BACKEND_PID=$!

# Wait for backend to be healthy
echo "   Waiting for model weights to load on MPS/Metal..."
for i in {1..15}; do
  if curl -s http://127.0.0.1:8008/api/health >/dev/null 2>&1; then
    echo "   ✅ Backend online! PyTorch model loaded successfully."
    break
  fi
  sleep 0.5
done

# 3. Launch Vite Frontend
echo "🚀 [2/2] Starting Frontend UI on http://127.0.0.1:5173..."
cd "$FRONTEND_DIR"
npm run dev -- --host 127.0.0.1 --port 5173 --clearScreen false &
FRONTEND_PID=$!

# Wait briefly for Vite dev server
sleep 1.5

echo ""
echo "============================================================"
echo "✨ TinyTransformer LM is LIVE and Ready!"
echo "   - Web Dashboard: [http://127.0.0.1:5173](http://127.0.0.1:5173)"
echo "   - Inference API: [http://127.0.0.1:8008](http://127.0.0.1:8008)"
echo "   - Swagger Docs:  [http://127.0.0.1:8008/docs](http://127.0.0.1:8008/docs)"
echo "============================================================"
echo "💡 Press Ctrl+C at any time to gracefully terminate both servers."
echo ""

# Keep running until Ctrl+C
wait
