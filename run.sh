#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PID=""
FRONTEND_PID=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$BACKEND_PID" ]]; then kill "$BACKEND_PID" 2>/dev/null || true; fi
  if [[ -n "$FRONTEND_PID" ]]; then kill "$FRONTEND_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
cd "$PROJECT_DIR"
node -e 'const [a,b]=process.versions.node.split(".").map(Number); if(a<22 || (a===22 && b<12)) {console.error("Node 22.12+ is required. Use nvm use after installing Node 22."); process.exit(1)}'
for port in 8008 5173; do
  if lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use. Stop that service or choose another port." >&2
    exit 1
  fi
done
if [[ ! -x backend/.venv/bin/python ]]; then
  echo "Create backend/.venv and install backend/requirements.txt first. See README.md." >&2
  exit 1
fi
(cd backend && exec .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8008) &
BACKEND_PID=$!
ready=false
for i in {1..60}; do
  if curl -fsS http://127.0.0.1:8008/api/health 2>/dev/null | python3 -c 'import json,sys;sys.exit(not json.load(sys.stdin)["model_loaded"])' 2>/dev/null; then ready=true; break; fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then echo "Backend exited." >&2; exit 1; fi
  sleep .5
done
if [[ "$ready" != true ]]; then echo "Backend failed its health check." >&2; exit 1; fi
(cd frontend && exec npm run dev -- --host 127.0.0.1 --port 5173 --strictPort) &
FRONTEND_PID=$!
echo "TinyTransformer: http://127.0.0.1:5173 | API: http://127.0.0.1:8008/docs"
wait "$FRONTEND_PID"
