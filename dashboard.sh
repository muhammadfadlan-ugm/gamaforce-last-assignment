#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
if [ -d "$SCRIPT_DIR/gamaforce-last-assignment" ]; then
    REPO_DIR="$SCRIPT_DIR/gamaforce-last-assignment"
else
    REPO_DIR="$SCRIPT_DIR"
fi

source /opt/ros/jazzy/setup.bash
source "$HOME/venv-ardupilot/bin/activate"

echo "=========================================================="
echo "      Menjalankan Web Dashboard Telemetri & Kamera        "
echo "=========================================================="
echo "[+] URL Dashboard: http://localhost:5000"
echo "----------------------------------------------------------"

cd "$REPO_DIR"
python3 "$REPO_DIR/scripts/baca_semua_data.py" "$@"

