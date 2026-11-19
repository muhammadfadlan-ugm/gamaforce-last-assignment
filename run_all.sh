#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
if [ -d "$SCRIPT_DIR/gamaforce-last-assignment" ]; then
    REPO_DIR="$SCRIPT_DIR/gamaforce-last-assignment"
else
    REPO_DIR="$SCRIPT_DIR"
fi

echo "=========================================================="
echo "    Menjalankan Seluruh Lingkungan Simulasi Gamaforce     "
echo "=========================================================="

# 1. Buka Terminal 2 (SITL ArduPilot) di jendela baru
echo "[+] Membuka jendela Terminal 2 (SITL ArduCopter)..."
gnome-terminal --title="Terminal 2 - SITL ArduCopter" -- bash -c "
cd '$REPO_DIR'
'$REPO_DIR/terminal2_sitl.sh'
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
'$REPO_DIR/terminal3_mavros.sh'
exec bash
"

# 3. Buka Terminal 4 (Misi Gabungan) di jendela baru
echo "[+] Membuka jendela Terminal 4 (Misi Drone)..."
gnome-terminal --title="Terminal 4 - Misi Drone" -- bash -c "
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
read -rp 'Tekan [ENTER] untuk menjalankan scripts/misi_gabungan.py...'
'$REPO_DIR/terminal4_misi.sh'
exec bash
"

# 4. Jalankan Terminal 1 (Randomizer & Gazebo Sim) di jendela saat ini
echo "[+] Menjalankan Terminal 1 (Gazebo Sim) di terminal ini..."
exec "$REPO_DIR/terminal1_gazebo.sh"

