#!/bin/bash
# =============================================================================
# Start Dashboard (foreground)
# =============================================================================
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "Starting Streamlit dashboard..."
echo "Access at: http://$(curl -s ifconfig.me):8501"
echo "Press Ctrl+C to stop"
echo ""

streamlit run src/dashboard/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
