#!/bin/bash
# =============================================================================
# NYC Bike Crash Dashboard - VM Setup Script
# =============================================================================
# Run this script on a fresh Ubuntu 22.04 VM (Scaleway, GCP, etc.)
# Usage: ./deploy/setup.sh
# =============================================================================

set -e  # Exit on error

echo "========================================"
echo "NYC Bike Crash Dashboard - Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root. Consider using a non-root user.${NC}"
fi

# =============================================================================
# 1. System Dependencies
# =============================================================================
echo -e "\n${GREEN}[1/5] Installing system dependencies...${NC}"
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git curl

# =============================================================================
# 2. Python Virtual Environment
# =============================================================================
echo -e "\n${GREEN}[2/5] Setting up Python virtual environment...${NC}"
cd "$(dirname "$0")/.."  # Go to project root

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created new virtual environment"
else
    echo "Virtual environment already exists"
fi

source .venv/bin/activate
pip install --upgrade pip

# =============================================================================
# 3. Python Dependencies
# =============================================================================
echo -e "\n${GREEN}[3/5] Installing Python dependencies...${NC}"
pip install -r requirements.txt

# =============================================================================
# 4. Data Pipeline
# =============================================================================
echo -e "\n${GREEN}[4/5] Running data pipeline (this takes ~30-60 minutes)...${NC}"
echo "Downloading ~20GB of data and processing..."

# Run full pipeline
make all

# =============================================================================
# 5. Create systemd service (optional, for auto-start)
# =============================================================================
echo -e "\n${GREEN}[5/5] Setup complete!${NC}"

echo ""
echo "========================================"
echo "Dashboard is ready to run!"
echo "========================================"
echo ""
echo "To start the dashboard manually:"
echo "  cd $(pwd)"
echo "  source .venv/bin/activate"
echo "  streamlit run src/dashboard/app.py --server.port=8501 --server.address=0.0.0.0"
echo ""
echo "Then access at: http://<YOUR-VM-IP>:8501"
echo ""
echo "To run in background (survives SSH disconnect):"
echo "  nohup streamlit run src/dashboard/app.py --server.port=8501 --server.address=0.0.0.0 > streamlit.log 2>&1 &"
echo ""
