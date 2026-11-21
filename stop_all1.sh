#!/usr/bin/env bash

echo "=========================================================="
echo "       Menghentikan Semua Proses Simulasi Drone           "
echo "=========================================================="

echo "[+] Mematikan proses Gazebo..."
killall -9 gz-sim gz-sim-server gz-sim-gui 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true

echo "[+] Mematikan proses ArduPilot SITL & MAVProxy..."
killall -9 arducopter mavproxy.py 2>/dev/null || true
pkill -9 -f "sim_vehicle.py" 2>/dev/null || true

echo "[+] Mematikan proses ROS 2, Bridge, & MAVROS..."
killall -9 parameter_bridge mavros_node 2>/dev/null || true
pkill -9 -f "sensors_mavros.launch.py" 2>/dev/null || true
pkill -9 -f "misi_gabungan.py" 2>/dev/null || true
pkill -9 -f "scripts_kanan/main.py" 2>/dev/null || true
pkill -9 -f "scripts_kiri/main.py" 2>/dev/null || true
pkill -9 -f "baca_semua_data.py" 2>/dev/null || true

# ==============================================================
# TUTUP JENDELA TERMINAL (Terminal 1, 2, 3, 4)
# ==============================================================
# Judul jendela dicocokkan SEBAGIAN, jadi "Terminal 4 - Misi Drone"
# maupun "Terminal 4 - Misi Drone (KIRI)" akan sama-sama kena.
echo "[+] Menutup jendela-jendela terminal..."
if command -v wmctrl >/dev/null 2>&1; then
    wmctrl -c "Terminal 2 - SITL ArduCopter" 2>/dev/null || true
    wmctrl -c "Terminal 3 - ROS 2 MAVROS" 2>/dev/null || true
    wmctrl -c "Terminal 4 - Misi Drone" 2>/dev/null || true
    sleep 0.5
else
    echo "    [!] 'wmctrl' belum terinstal -> jendela tidak bisa ditutup otomatis."
    echo "        Install dulu dengan: sudo apt install wmctrl"
fi

echo "[OK] Semua proses simulasi berhasil dihentikan!"
