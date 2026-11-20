#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

echo "=========================================================="
echo "    [Terminal 2] ArduPilot SITL (Flight Controller)       "
echo "=========================================================="

# 1. Aktifkan virtual environment ArduPilot
echo "[1/2] Mengaktifkan virtual environment ArduPilot..."
source "$HOME/venv-ardupilot/bin/activate"
export PATH="$PATH:$HOME/sim/ardu_ws/src/ardupilot/Tools/autotest"

# 2. Jalankan simulasi SITL
# Catatan: Port 14550 untuk MAVROS, port 14560 (--out) untuk misi_gabungan.py
echo "[2/2] Menjalankan SITL ArduCopter..."
echo "----------------------------------------------------------"
echo " PENTING: Tunggu hingga muncul 'EKF3 IMU origin set' dan   "
echo " status di console berwarna HIJAU sebelum menjalankan MAVROS!"
echo "----------------------------------------------------------"
cd "$REPO_DIR"
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console --out 127.0.0.1:14560

