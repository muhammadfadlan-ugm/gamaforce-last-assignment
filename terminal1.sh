#!/usr/bin/env bash
set -e

# Path root repositori
REPO_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

echo "=========================================================="
echo "    [Terminal 1] Randomizer & Gazebo Sim (GUI)            "
echo "=========================================================="

# 1. Acak target ArUco & update tekstur
echo "[1/3] Menjalankan randomize target..."
python3 "$REPO_DIR/worlds/world_oprec/randomize.py" --model-dir "$REPO_DIR/models/Part/obstacles"

# 2. Muat environment ROS 2 & Gazebo ArduPilot
echo "[2/3] Memuat environment ROS 2 & Gazebo..."
source /opt/ros/jazzy/setup.bash
source "$HOME/sim/ardu_ws/install/setup.bash"

export GZ_VERSION=harmonic
export GZ_SIM_RESOURCE_PATH="$HOME/sim/ardu_ws/install/ardupilot_gazebo/share:$REPO_DIR/models/Part:$REPO_DIR/models:${GZ_SIM_RESOURCE_PATH:-}"

# 3. Jalankan Gazebo Sim dengan GUI
echo "[3/3] Membuka Gazebo Sim GUI..."
cd "$REPO_DIR/worlds/world_oprec"
gz sim -r "$REPO_DIR/worlds/world_oprec/world_oprec.world"

