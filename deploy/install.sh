#!/usr/bin/env bash
# Install the trading-agent systemd service.
# Run as root: sudo bash deploy/install.sh
set -euo pipefail

SERVICE_NAME="trading-agent"
SERVICE_FILE="$(dirname "$0")/${SERVICE_NAME}.service"
SYSTEMD_DIR="/etc/systemd/system"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo)."
    exit 1
fi

if [[ ! -f "$SERVICE_FILE" ]]; then
    echo "ERROR: Service file not found at $SERVICE_FILE"
    exit 1
fi

echo "Installing ${SERVICE_NAME}.service ..."
cp "$SERVICE_FILE" "${SYSTEMD_DIR}/${SERVICE_NAME}.service"
chmod 644 "${SYSTEMD_DIR}/${SERVICE_NAME}.service"

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"

echo ""
echo "Service installed and enabled. To start it:"
echo "  sudo systemctl start ${SERVICE_NAME}"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status ${SERVICE_NAME}     # live status"
echo "  sudo journalctl -u ${SERVICE_NAME} -f     # tail logs"
echo "  sudo systemctl stop ${SERVICE_NAME}       # graceful stop"
echo "  sudo systemctl restart ${SERVICE_NAME}    # restart"
echo ""
echo "NOTE: The instance lock on port 47293 prevents a second process from"
echo "  starting before the first fully exits. RestartSec=30 in the unit file"
echo "  is intentional — it gives the old process time to release the lock."
