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
pkill -9 -f "baca_semua_data.py" 2>/dev/null || true

echo "[OK] Semua proses simulasi berhasil dihentikan!"

