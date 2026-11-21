#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

echo "=========================================================="
echo "  Menjalankan Seluruh Lingkungan Simulasi Gamaforce        "
echo "                  (JALUR KIRI)                             "
echo "=========================================================="

# 1. Buka Terminal 2 (SITL ArduPilot) di jendela baru
echo "[+] Membuka jendela Terminal 2 (SITL ArduCopter)..."
gnome-terminal --title="Terminal 2 - SITL ArduCopter" -- bash -c "
cd '$REPO_DIR'
'$REPO_DIR/terminal2.sh'
exec bash
"

# 2. Buka Terminal 3 (Bridge Sensor & MAVROS) di jendela baru
echo "[+] Membuka jendela Terminal 3 (ROS 2 Bridge & MAVROS)..."
gnome-terminal --title="Terminal 3 - ROS 2 MAVROS" -- bash -c "
cd '$REPO_DIR'
echo ''
echo '========================================================================='
echo ' PERHATIAN: Tunggu sampai status di Terminal 2 berwarna HIJAU'
echo ' dan muncul pesan \"EKF3 IMU origin set\" sebelum melanjutkan.'
echo '========================================================================='
read -rp 'Tekan [ENTER] jika Terminal 2 sudah siap (hijau) untuk memulai MAVROS...'
echo ''
'$REPO_DIR/terminal3.sh'
exec bash
"

# 3. Buka Terminal 4 (Misi Drone - JALUR KIRI) di jendela baru
echo "[+] Membuka jendela Terminal 4 (Misi Drone - Jalur Kiri)..."
gnome-terminal --title="Terminal 4 - Misi Drone (KIRI)" -- bash -c "
cd '$REPO_DIR'
echo ''
echo '========================================================================='
echo ' Menunggu topik sensor dari Terminal 3 aktif...'
echo '========================================================================='
source /opt/ros/jazzy/setup.bash
until ros2 topic list 2>/dev/null | grep -q '/lidar_front'; do
    sleep 1
done
echo '[OK] Sensor terdeteksi!'
read -rp 'Tekan [ENTER] untuk menjalankan scripts/scripts_kiri/main.py...'
'$REPO_DIR/terminal4_kiri.sh'
exec bash
"

# 4. Jalankan Terminal 1 (Randomizer & Gazebo Sim) di jendela saat ini
echo "[+] Menjalankan Terminal 1 (Gazebo Sim) di terminal ini..."
exec "$REPO_DIR/terminal1.sh"
