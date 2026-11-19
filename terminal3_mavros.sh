#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
if [ -d "$SCRIPT_DIR/gamaforce-last-assignment" ]; then
    REPO_DIR="$SCRIPT_DIR/gamaforce-last-assignment"
else
    REPO_DIR="$SCRIPT_DIR"
fi

echo "=========================================================="
echo "    [Terminal 3] Bridge Sensor (LiDAR, Kamera) & MAVROS   "
echo "=========================================================="

# 1. Muat environment ROS 2
echo "[1/2] Memuat environment ROS 2 Jazzy..."
source /opt/ros/jazzy/setup.bash

# 2. Jalankan launch file
echo "[2/2] Menjalankan sensor bridge & MAVROS node..."
ros2 launch "$REPO_DIR/scripts/sensors_mavros.launch.py"

