#!/bin/bash
set -e

cd "$(dirname "$0")"
source venv/bin/activate

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $UVICORN_PID 2>/dev/null
    python -c "from pyngrok import ngrok; ngrok.kill()" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "🚀 Starting FastAPI server on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

# Give uvicorn a moment to start
sleep 2

echo "🔗 Starting ngrok tunnel to port 8000..."
python -c "
from pyngrok import ngrok
tunnel = ngrok.connect(8000)
print(f'✅ Ngrok tunnel active at: {tunnel.public_url}')
print(f'📋 API Root:   {tunnel.public_url}/')
print(f'📋 Predict:    {tunnel.public_url}/predict/')
"

echo ""
echo "=========================================="
echo "  Share the Public URL with your friends!"
echo "  Press Ctrl+C to stop everything."
echo "=========================================="

wait $UVICORN_PID
