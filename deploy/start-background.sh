#!/bin/bash
# =============================================================================
# Start Dashboard (background - survives SSH disconnect)
# =============================================================================
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Kill existing streamlit if running
pkill -f "streamlit run" 2>/dev/null || true

echo "Starting Streamlit dashboard in background..."
nohup streamlit run src/dashboard/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    > streamlit.log 2>&1 &

sleep 2

if pgrep -f "streamlit run" > /dev/null; then
    echo "Dashboard started successfully!"
    echo ""
    echo "Access at: http://$(curl -s ifconfig.me):8501"
    echo ""
    echo "Logs: tail -f streamlit.log"
    echo "Stop: pkill -f 'streamlit run'"
else
    echo "Failed to start dashboard. Check streamlit.log for errors."
    exit 1
fi
