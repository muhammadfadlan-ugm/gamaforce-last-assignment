#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
if [ -d "$SCRIPT_DIR/gamaforce-last-assignment" ]; then
    REPO_DIR="$SCRIPT_DIR/gamaforce-last-assignment"
else
    REPO_DIR="$SCRIPT_DIR"
fi

echo "=========================================================="
echo "    [Terminal 4] Eksekusi Misi Drone (misi_gabungan.py)   "
echo "=========================================================="

# 1. Muat environment ROS 2 & pymavlink (venv)
echo "[1/3] Memuat environment ROS 2 & ArduPilot venv..."
source /opt/ros/jazzy/setup.bash
source "$HOME/venv-ardupilot/bin/activate"

# 2. Menunggu topik sensor aktif
echo "[2/3] Menunggu sensor aktif di ROS 2 (topik /lidar_front)..."
until ros2 topic list 2>/dev/null | grep -q '/lidar_front'; do
    sleep 1
done
echo "[OK] Sensor terdeteksi di ROS 2!"

# 3. Jalankan script misi
echo "[3/3] Menjalankan scripts/misi_gabungan.py..."
cd "$REPO_DIR"
python3 "$REPO_DIR/scripts/misi_gabungan.py"

