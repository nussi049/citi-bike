# Deployment Guide

Quick deployment of the NYC Bike Crash Dashboard on a VM (Scaleway, GCP, etc.).

## Prerequisites

- Ubuntu 22.04 VM with at least:
  - 2 vCPU
  - 4GB RAM (2GB might work but tight)
  - 50GB disk space
- Port 8501 open in firewall/security group
- SSH access to VM

## Quick Start (Scaleway)

### 1. Create VM

1. Go to [Scaleway Console](https://console.scaleway.com)
2. Create Instance:
   - **Type:** DEV1-M (3 vCPU, 4GB RAM) - ~€0.02/h
   - **Image:** Ubuntu 22.04
   - **Region:** Paris (PAR1) or Amsterdam (AMS1)
   - **SSH Key:** Add your public key

3. Open Port 8501:
   - Go to Security Groups
   - Add inbound rule: TCP 8501 from 0.0.0.0/0

### 2. Deploy

```bash
# SSH to your VM
ssh root@<VM-IP>

# Clone repository
git clone https://github.com/YOUR_USER/city-bike.git
cd city-bike

# Make scripts executable
chmod +x deploy/*.sh

# Run setup (takes 30-60 min for data download)
./deploy/setup.sh

# Start dashboard in background
./deploy/start-background.sh
```

### 3. Access

Open in browser: `http://<VM-IP>:8501`

## Password Protection

To add password protection:

```bash
# Set environment variable before starting
export DASHBOARD_PASSWORD="YourSecretPassword"
./deploy/start-background.sh
```

Or create `.streamlit/secrets.toml`:
```toml
password = "YourSecretPassword"
```

Then add this to the top of `src/dashboard/app.py`:
```python
import sys
sys.path.insert(0, 'deploy')
from auth import check_password
if not check_password():
    st.stop()
```

## Commands

| Command | Description |
|---------|-------------|
| `./deploy/setup.sh` | Full setup (install + data pipeline) |
| `./deploy/start.sh` | Start dashboard (foreground) |
| `./deploy/start-background.sh` | Start dashboard (background) |
| `pkill -f 'streamlit run'` | Stop dashboard |
| `tail -f streamlit.log` | View logs |

## Troubleshooting

### Dashboard not accessible
- Check firewall: Port 8501 must be open
- Check if running: `pgrep -f streamlit`
- Check logs: `tail -f streamlit.log`

### Out of memory during `make all`
- The modeling step needs ~4GB RAM
- Use a larger VM or add swap:
  ```bash
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```

### Data download fails
- Check internet connection
- Some APIs may have rate limits - wait and retry
- Individual steps: `make trips`, `make crashes`, `make weather`

## Cost Estimate

| Provider | Instance | Cost |
|----------|----------|------|
| Scaleway | DEV1-M | ~€15/month |
| GCP | e2-medium | ~$25/month |
| AWS | t3.medium | ~$30/month |

Tip: Stop the VM when not in use to save costs.
