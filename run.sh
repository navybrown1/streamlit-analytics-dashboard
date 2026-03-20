#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=3001
HOST="127.0.0.1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./run.sh [--port 3001]"
      exit 1
      ;;
  esac
done

port_in_use() {
  lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

while port_in_use "$PORT"; do
  PORT=$((PORT + 1))
done

cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt >/dev/null

echo "Launching LexiLift Vocabulary App"
echo "URL: http://$HOST:$PORT"

exec streamlit run app.py --server.address "$HOST" --server.port "$PORT"
