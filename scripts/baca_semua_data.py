#!/usr/bin/env python3
"""
================================================================================
GAMAFORCE - WEB DASHBOARD TELEMETRI & KAMERA DRONE TERPADU
================================================================================
Dashboard real-time terpadu untuk memantau:
- 2 Kamera: /camera_front, /camera_down (beserta /annotated stream visual OpenCV)
- 6 LiDAR 1D: /lidar_front, /lidar_back, /lidar_left, /lidar_right, /lidar_up, /lidar_down
- Flight Status: /mavros/state (Connection, Arming, Flight Mode)
- Posisi & Attitude: /mavros/local_position/pose (Altitude Z, X, Y, Roll, Pitch, Yaw)
- Kecepatan: /mavros/local_position/velocity_body & velocity_local
- Baterai: /mavros/battery
- Radar Hambatan 2D real-time

Dapat diakses di browser: http://localhost:5000
================================================================================
"""

import sys
import os
import time
import math
import json
import threading
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
import cv2

# ROS 2 & QoS Profiles
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, LaserScan, BatteryState
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State

LIDAR_NAMES = ['front', 'back', 'left', 'right', 'up', 'down']


# ==============================================================================
# DATA HUB (Thread-Safe Telemetry & Vision Store)
# ==============================================================================
class TelemetryDataHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True

        # Buffer JPEG Kamera
        self.jpeg_front = self._create_placeholder_jpeg("Kamera Depan Menunggu Feed...")
        self.jpeg_down = self._create_placeholder_jpeg("Kamera Bawah Menunggu Feed...")
        self.has_front = False
        self.has_down = False
        self.t_last_ann_front = 0.0
        self.t_last_ann_down = 0.0

        # Status & Telemetri Terpadu
        self.telemetry = {
            "connected": False,
            "sim_active": False,
            "armed": False,
            "guided": False,
            "mode": "DISCONNECTED",
            "system_status": 0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "battery_volt": 0.0,
            "battery_pct": 0.0,
            "lidar": {name: None for name in LIDAR_NAMES},
            "lidar_status": {name: "Menunggu..." for name in LIDAR_NAMES},
            "last_update": time.time(),
            "cam_front_active": False,
            "cam_down_active": False
        }

    def _create_placeholder_jpeg(self, label: str) -> bytes:
        img = np.zeros((360, 480, 3), dtype=np.uint8)
        img[:] = (18, 22, 28)
        cv2.rectangle(img, (10, 10), (470, 350), (45, 55, 72), 2)
        cv2.putText(img, label, (35, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 175, 195), 2, cv2.LINE_AA)
        cv2.putText(img, "Menunggu topic aktif dari sensor bridge...", (65, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (105, 115, 130), 1, cv2.LINE_AA)
        _, jpeg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return jpeg.tobytes()

    def update_camera(self, cam_type: str, img_msg: Image):
        """Konversi citra ROS 2 (RGB/BGR/Mono/RGBA) ke JPEG untuk MJPEG streaming."""
        try:
            is_ann = "ann" in cam_type
            now = time.time()

            # Jika frame biasa datang tetapi frame annotated baru saja diperbarui (<0.8s lalu), prioritaskan annotated
            if not is_ann:
                if 'front' in cam_type and (now - self.t_last_ann_front < 0.8):
                    return
                if 'down' in cam_type and (now - self.t_last_ann_down < 0.8):
                    return

            encoding = img_msg.encoding.lower() if img_msg.encoding else 'bgr8'
            if encoding in ('bgr8', 'rgb8'):
                n_channels = 3
                dtype = np.uint8
            elif encoding in ('mono8', '8uc1'):
                n_channels = 1
                dtype = np.uint8
            elif encoding in ('rgba8', 'bgra8', '8uc4'):
                n_channels = 4
                dtype = np.uint8
            else:
                n_channels = 3
                dtype = np.uint8

            img_arr = np.frombuffer(img_msg.data, dtype=dtype)
            if n_channels == 3:
                img_arr = img_arr.reshape((img_msg.height, img_msg.width, 3))
                bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR) if encoding == 'rgb8' else img_arr
            elif n_channels == 4:
                img_arr = img_arr.reshape((img_msg.height, img_msg.width, 4))
                bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR) if encoding in ('rgba8', '8uc4') else cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)
            else:
                img_arr = img_arr.reshape((img_msg.height, img_msg.width))
                bgr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)

            ret, jpeg = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if ret:
                with self.lock:
                    if 'front' in cam_type:
                        self.jpeg_front = jpeg.tobytes()
                        self.has_front = True
                        self.telemetry['cam_front_active'] = True
                        if is_ann:
                            self.t_last_ann_front = now
                    else:
                        self.jpeg_down = jpeg.tobytes()
                        self.has_down = True
                        self.telemetry['cam_down_active'] = True
                        if is_ann:
                            self.t_last_ann_down = now
                    self.telemetry['sim_active'] = True
                    self.telemetry['last_update'] = now
        except Exception:
            pass

    def update_lidar(self, name: str, scan_msg: LaserScan):
        """Pembaruan jarak 6 LiDAR dengan penyaringan data valid."""
        with self.lock:
            if scan_msg.ranges:
                # Saring pembacaan numerik valid
                valid = [r for r in scan_msg.ranges if not math.isnan(r) and not math.isinf(r) and 0.05 < r < 25.0]
                if valid:
                    r = min(valid)
                    self.telemetry['lidar'][name] = round(float(r), 2)
                    self.telemetry['lidar_status'][name] = f"{r:4.2f} m"
                else:
                    # Tidak ada objek terdeteksi di dalam jarak jangkau sensor (bebas/aman)
                    max_r = scan_msg.range_max if (scan_msg.range_max and scan_msg.range_max > 0) else 10.0
                    self.telemetry['lidar'][name] = round(float(max_r), 1)
                    self.telemetry['lidar_status'][name] = f"> {max_r:.1f} m (Bebas)"
            self.telemetry['sim_active'] = True
            self.telemetry['last_update'] = time.time()

    def update_state(self, state_msg: State):
        """Pembaruan status koneksi dan flight mode dari MAVROS."""
        with self.lock:
            self.telemetry['connected'] = state_msg.connected
            self.telemetry['sim_active'] = True
            self.telemetry['armed'] = state_msg.armed
            self.telemetry['guided'] = state_msg.guided
            self.telemetry['mode'] = state_msg.mode if state_msg.mode else "DISCONNECTED"
            self.telemetry['system_status'] = state_msg.system_status
            self.telemetry['last_update'] = time.time()

    def update_pose(self, pose_msg: PoseStamped):
        """Pembaruan posisi EKF (x, y, z) dan orientasi sudut Euler (roll, pitch, yaw)."""
        p = pose_msg.pose.position
        q = pose_msg.pose.orientation
        roll, pitch, yaw = self._quaternion_to_euler(q.x, q.y, q.z, q.w)

        with self.lock:
            self.telemetry['x'] = round(float(p.x), 2)
            self.telemetry['y'] = round(float(p.y), 2)
            self.telemetry['z'] = round(float(p.z), 2)
            self.telemetry['roll'] = round(roll, 1)
            self.telemetry['pitch'] = round(pitch, 1)
            self.telemetry['yaw'] = round(yaw, 1)
            self.telemetry['sim_active'] = True
            self.telemetry['last_update'] = time.time()

    def update_velocity(self, vel_msg: TwistStamped):
        """Pembaruan kecepatan linier translasi."""
        v = vel_msg.twist.linear
        with self.lock:
            self.telemetry['vx'] = round(float(v.x), 2)
            self.telemetry['vy'] = round(float(v.y), 2)
            self.telemetry['vz'] = round(float(v.z), 2)
            self.telemetry['sim_active'] = True
            self.telemetry['last_update'] = time.time()

    def update_battery(self, bat_msg: BatteryState):
        """Pembaruan status baterai."""
        with self.lock:
            self.telemetry['battery_volt'] = round(float(bat_msg.voltage), 1)
            pct = float(bat_msg.percentage) * 100.0 if bat_msg.percentage <= 1.0 else float(bat_msg.percentage)
            self.telemetry['battery_pct'] = round(max(0.0, min(100.0, pct)), 0)
            self.telemetry['last_update'] = time.time()

    def get_telemetry_json(self) -> str:
        with self.lock:
            # Periksa apakah data telemetri telah basi (> 4.5 detik tanpa pesan apapun)
            if time.time() - self.telemetry['last_update'] > 4.5:
                self.telemetry['sim_active'] = False
            return json.dumps(self.telemetry)

    def get_jpeg(self, cam_type: str) -> bytes:
        with self.lock:
            return self.jpeg_front if 'front' in cam_type else self.jpeg_down

    @staticmethod
    def _quaternion_to_euler(x, y, z, w):
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


data_hub = TelemetryDataHub()


# ==============================================================================
# ROS 2 SUBSCRIBER NODE (Sensor Data QoS Compatibility)
# ==============================================================================
class TelemetryListenerNode(Node):
    def __init__(self):
        super().__init__('baca_semua_data_node')

        # QoS sensor_data (BEST_EFFORT + VOLATILE) untuk kamera, lidar, dan pose
        qos_sensor = qos_profile_sensor_data

        # 1. Subscriber Kamera Depan & Bawah (Raw & Olah Citra Vision)
        self.sub_cam_front = self.create_subscription(
            Image, '/camera_front', lambda m: data_hub.update_camera('front', m), qos_sensor
        )
        self.sub_cam_down = self.create_subscription(
            Image, '/camera_down', lambda m: data_hub.update_camera('down', m), qos_sensor
        )
        self.sub_cam_front_ann = self.create_subscription(
            Image, '/camera_front/annotated', lambda m: data_hub.update_camera('front_ann', m), qos_sensor
        )
        self.sub_cam_down_ann = self.create_subscription(
            Image, '/camera_down/annotated', lambda m: data_hub.update_camera('down_ann', m), qos_sensor
        )

        # 2. Subscriber 6 LiDAR 1D
        self.sub_lidars = []
        for name in LIDAR_NAMES:
            topic = f'/lidar_{name}'
            sub = self.create_subscription(
                LaserScan, topic, self._make_lidar_cb(name), qos_sensor
            )
            self.sub_lidars.append(sub)

        # 3. Subscriber MAVROS State & Posisi
        # /mavros/state biasanya latched dengan TRANSIENT_LOCAL
        qos_state_rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos_state_be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub_state_rel = self.create_subscription(
            State, '/mavros/state', data_hub.update_state, qos_state_rel
        )
        self.sub_state_be = self.create_subscription(
            State, '/mavros/state', data_hub.update_state, qos_state_be
        )

        self.sub_pose = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', data_hub.update_pose, qos_sensor
        )
        self.sub_vel_body = self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_body', data_hub.update_velocity, qos_sensor
        )
        self.sub_vel_local = self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_local', data_hub.update_velocity, qos_sensor
        )
        self.sub_bat = self.create_subscription(
            BatteryState, '/mavros/battery', data_hub.update_battery, qos_sensor
        )

        self.get_logger().info("TelemetryListenerNode aktif dengan QoS sensor_data untuk menjamin penerimaan data.")

    def _make_lidar_cb(self, name):
        return lambda msg: data_hub.update_lidar(name, msg)


# ==============================================================================
# WEB SERVER & RESPONSIVE DASHBOARD UI
# ==============================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gamaforce Drone Telemetry & Vision Dashboard</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --accent: #3b82f6;
      --accent-glow: rgba(59, 130, 246, 0.3);
      --green: #10b981;
      --yellow: #f59e0b;
      --red: #ef4444;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 16px; min-height: 100vh; }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 20px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      margin-bottom: 16px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand h1 { font-size: 1.25rem; font-weight: 700; letter-spacing: 0.5px; }
    .brand span { color: var(--accent); }
    .badges { display: flex; gap: 10px; align-items: center; }
    .badge {
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.25s ease;
    }
    .badge-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .bg-red { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid var(--red); }
    .bg-green { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid var(--green); }
    .bg-yellow { background: rgba(245, 158, 11, 0.2); color: #fde68a; border: 1px solid var(--yellow); }
    .bg-blue { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid var(--accent); }

    /* Grid Layout */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }
    @media (max-width: 900px) {
      .dashboard-grid { grid-template-columns: 1fr; }
    }

    /* Cards */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    }
    .card-header {
      padding: 10px 16px;
      background: rgba(255,255,255,0.02);
      border-bottom: 1px solid var(--card-border);
      font-size: 0.9rem;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .card-body { padding: 14px 16px; }

    /* Video Feeds */
    .video-container {
      position: relative;
      background: #000;
      width: 100%;
      height: 320px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .video-container img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .video-overlay {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(0,0,0,0.65);
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-family: monospace;
      color: #38bdf8;
    }

    /* Telemetry Stats Grid */
    .telemetry-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .stat-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px 14px;
      text-align: center;
    }
    .stat-title { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }
    .stat-value { font-size: 1.4rem; font-weight: 700; color: #fff; font-family: monospace; }
    .stat-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }

    /* LiDAR & Radar Section */
    .lidar-section {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 16px;
    }
    @media (max-width: 900px) {
      .lidar-section { grid-template-columns: 1fr; }
    }
    .lidar-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }
    .lidar-card {
      background: rgba(255,255,255,0.02);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px;
      text-align: center;
      transition: all 0.2s ease;
    }
    .lidar-name { font-size: 0.75rem; font-weight: 700; color: var(--accent); margin-bottom: 4px; text-transform: uppercase; }
    .lidar-val { font-size: 1.25rem; font-family: monospace; font-weight: 700; margin-bottom: 6px; }
    .lidar-bar-bg { height: 6px; background: #1f2937; border-radius: 3px; overflow: hidden; }
    .lidar-bar { height: 100%; width: 0%; transition: width 0.15s ease, background-color 0.15s ease; }

    /* Radar Widget */
    .radar-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    canvas#radarCanvas {
      background: #0b1120;
      border: 1px solid var(--card-border);
      border-radius: 50%;
    }
  </style>
</head>
<body>

  <!-- Header Status -->
  <header>
    <div class="brand">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
      <h1>GAMAFORCE <span>GCS TELEMETRY</span></h1>
    </div>
    <div class="badges">
      <div id="badge-conn" class="badge bg-red"><span class="badge-dot" style="background:#ef4444"></span><span id="txt-conn">MENUNGGU FEED</span></div>
      <div id="badge-arm" class="badge bg-red"><span id="txt-arm">DISARMED</span></div>
      <div id="badge-mode" class="badge bg-blue"><span id="txt-mode">MODE: -</span></div>
    </div>
  </header>

  <!-- Video Feeds: Kamera Depan & Kamera Bawah -->
  <div class="dashboard-grid">
    <div class="card">
      <div class="card-header">
        <span>Kamera Depan (/camera_front &amp; Vision HUD)</span>
        <span id="cam-front-stat" style="color:var(--text-muted); font-size:0.8rem">STANDBY</span>
      </div>
      <div class="video-container">
        <span class="video-overlay" id="overlay-front">FRONT 640x480</span>
        <img id="img-front" src="/stream/camera_front" alt="Kamera Depan" onerror="setTimeout(() => { this.src = '/stream/camera_front?t=' + Date.now(); }, 1000)">
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <span>Kamera Bawah (/camera_down &amp; Vision HUD)</span>
        <span id="cam-down-stat" style="color:var(--text-muted); font-size:0.8rem">STANDBY</span>
      </div>
      <div class="video-container">
        <span class="video-overlay" id="overlay-down">DOWN 640x480</span>
        <img id="img-down" src="/stream/camera_down" alt="Kamera Bawah" onerror="setTimeout(() => { this.src = '/stream/camera_down?t=' + Date.now(); }, 1000)">
      </div>
    </div>
  </div>

  <!-- Telemetry Row: Altitude, Attitude RPY, Posisi XY, Kecepatan, Baterai -->
  <div class="telemetry-row">
    <div class="stat-card">
      <div class="stat-title">Ketinggian (Z)</div>
      <div class="stat-value" id="val-alt">0.00 m</div>
      <div class="stat-sub" id="val-vz">Vz: 0.0 m/s</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">Posisi X / Y</div>
      <div class="stat-value" id="val-xy">0.00 / 0.00</div>
      <div class="stat-sub">Koordinat lokal (m)</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">Heading (Yaw)</div>
      <div class="stat-value" id="val-yaw">0.0°</div>
      <div class="stat-sub" id="val-rp">R: 0.0° | P: 0.0°</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">Kecepatan Vx / Vy</div>
      <div class="stat-value" id="val-vxy">0.00 m/s</div>
      <div class="stat-sub">Horizontal Speed</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">Baterai</div>
      <div class="stat-value" id="val-bat">-- V</div>
      <div class="stat-sub" id="val-bat-pct">-- %</div>
    </div>
  </div>

  <!-- LiDAR 6 Sensor & 2D Obstacle Radar -->
  <div class="lidar-section">
    <div class="card">
      <div class="card-header">
        <span>Data 6 Sensor LiDAR (Jarak Hambatan)</span>
        <span style="font-size:0.8rem; color:var(--text-muted);">Threshold Bahaya: &lt; 0.8m</span>
      </div>
      <div class="card-body">
        <div class="lidar-grid">
          <div class="lidar-card" id="card-lidar-front">
            <div class="lidar-name">Depan</div>
            <div class="lidar-val" id="val-lidar-front">--</div>
            <div class="lidar-bar-bg"><div class="lidar-bar" id="bar-lidar-front"></div></div>
          </div>
          <div class="lidar-card" id="card-lidar-back">
            <div class="lidar-name">Belakang</div>
            <div class="lidar-val" id="val-lidar-back">--</div>
            <div class="lidar-bar-bg"><div class="lidar-bar" id="bar-lidar-back"></div></div>
          </div>
          <div class="lidar-card" id="card-lidar-left">
            <div class="lidar-name">Kiri</div>
            <div class="lidar-val" id="val-lidar-left">--</div>
            <div class="lidar-bar-bg"><div class="lidar-bar" id="bar-lidar-left"></div></div>
          </div>
          <div class="lidar-card" id="card-lidar-right">
            <div class="lidar-name">Kanan</div>
            <div class="lidar-val" id="val-lidar-right">--</div>
            <div class="lidar-bar-bg"><div class="lidar-bar" id="bar-lidar-right"></div></div>
          </div>
          <div class="lidar-card" id="card-lidar-up">
            <div class="lidar-name">Atas (Plafon)</div>
            <div class="lidar-val" id="val-lidar-up">--</div>
            <div class="lidar-bar-bg"><div class="lidar-bar" id="bar-lidar-up"></div></div>
          </div>
          <div class="lidar-card" id="card-lidar-down">
            <div class="lidar-name">Bawah (Lantai)</div>
            <div class="lidar-val" id="val-lidar-down">--</div>
            <div class="lidar-bar-bg"><div class="lidar-bar" id="bar-lidar-down"></div></div>
          </div>
        </div>
      </div>
    </div>

    <!-- 2D Obstacle Radar Canvas -->
    <div class="card">
      <div class="card-header">
        <span>Radar 2D Hambatan Sekitar</span>
      </div>
      <div class="card-body radar-container">
        <canvas id="radarCanvas" width="200" height="200"></canvas>
      </div>
    </div>
  </div>

  <script>
    const LIDAR_NAMES = ['front', 'back', 'left', 'right', 'up', 'down'];

    // Polling Telemetri secara periodik (~100ms)
    async function updateTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        if (!res.ok) throw new Error('Network error');
        const d = await res.json();

        // 1. Connection & Mode Status
        const bConn = document.getElementById('badge-conn');
        const tConn = document.getElementById('txt-conn');
        if (d.connected || d.sim_active) {
          bConn.className = 'badge bg-green';
          tConn.innerText = d.connected ? 'CONNECTED (MAVROS)' : 'SIM ACTIVE (BRIDGED)';
        } else {
          bConn.className = 'badge bg-red';
          tConn.innerText = 'DISCONNECTED';
        }

        // Arming
        const bArm = document.getElementById('badge-arm');
        const tArm = document.getElementById('txt-arm');
        if (d.armed) {
          bArm.className = 'badge bg-green';
          tArm.innerText = 'ARMED';
        } else {
          bArm.className = 'badge bg-red';
          tArm.innerText = 'DISARMED';
        }

        // Mode
        document.getElementById('txt-mode').innerText = 'MODE: ' + (d.mode || '-');

        // Status Kamera Live
        const statF = document.getElementById('cam-front-stat');
        if (d.cam_front_active) {
          statF.innerText = 'LIVE';
          statF.style.color = 'var(--green)';
        }
        const statD = document.getElementById('cam-down-stat');
        if (d.cam_down_active) {
          statD.innerText = 'LIVE';
          statD.style.color = 'var(--green)';
        }

        // 2. Stats Numerik
        const alt = typeof d.z === 'number' ? d.z.toFixed(2) : '0.00';
        const vz = typeof d.vz === 'number' ? d.vz.toFixed(2) : '0.00';
        const px = typeof d.x === 'number' ? d.x.toFixed(2) : '0.00';
        const py = typeof d.y === 'number' ? d.y.toFixed(2) : '0.00';
        const yaw = typeof d.yaw === 'number' ? d.yaw.toFixed(1) : '0.0';
        const roll = typeof d.roll === 'number' ? d.roll.toFixed(1) : '0.0';
        const pitch = typeof d.pitch === 'number' ? d.pitch.toFixed(1) : '0.0';

        document.getElementById('val-alt').innerText = alt + ' m';
        document.getElementById('val-vz').innerText = 'Vz: ' + vz + ' m/s';
        document.getElementById('val-xy').innerText = px + ' / ' + py;
        document.getElementById('val-yaw').innerText = yaw + '°';
        document.getElementById('val-rp').innerText = 'R: ' + roll + '° | P: ' + pitch + '°';

        const vx = typeof d.vx === 'number' ? d.vx : 0;
        const vy = typeof d.vy === 'number' ? d.vy : 0;
        const speedXY = Math.sqrt(vx * vx + vy * vy);
        document.getElementById('val-vxy').innerText = speedXY.toFixed(2) + ' m/s';

        if (d.battery_volt && d.battery_volt > 0) {
          document.getElementById('val-bat').innerText = d.battery_volt.toFixed(1) + ' V';
          document.getElementById('val-bat-pct').innerText = (d.battery_pct || 0).toFixed(0) + ' %';
        }

        // 3. Update 6 LiDAR Bars
        if (d.lidar && d.lidar_status) {
          LIDAR_NAMES.forEach(name => {
            const valEl = document.getElementById('val-lidar-' + name);
            const barEl = document.getElementById('bar-lidar-' + name);
            const cardEl = document.getElementById('card-lidar-' + name);
            if (!valEl || !barEl || !cardEl) return;

            const val = d.lidar[name];
            const statText = d.lidar_status[name] || '--';
            valEl.innerText = statText;

            if (val === null || isNaN(val)) {
              barEl.style.width = '0%';
              cardEl.style.borderColor = 'var(--card-border)';
            } else {
              const pct = Math.min(100, Math.max(6, (val / 10.0) * 100));
              barEl.style.width = pct + '%';

              if (val < 0.8) {
                barEl.style.backgroundColor = 'var(--red)';
                cardEl.style.borderColor = 'var(--red)';
              } else if (val < 1.5) {
                barEl.style.backgroundColor = 'var(--yellow)';
                cardEl.style.borderColor = 'var(--yellow)';
              } else {
                barEl.style.backgroundColor = 'var(--green)';
                cardEl.style.borderColor = 'var(--card-border)';
              }
            }
          });
        }

        // 4. Render 2D Radar
        drawRadar(d.lidar);

      } catch (e) {
        console.warn('Telemetry fetch error:', e);
      } finally {
        setTimeout(updateTelemetry, 100);
      }
    }

    // Gambar Radar 2D Lingkaran
    const canvas = document.getElementById('radarCanvas');
    const ctx = canvas ? canvas.getContext('2d') : null;
    const cx = canvas ? canvas.width / 2 : 100;
    const cy = canvas ? canvas.height / 2 : 100;
    const maxRadius = 85;

    function drawRadar(lidar) {
      if (!ctx || !lidar) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Lingkaran konsentris jarak (2m, 5m, 10m)
      [0.2, 0.5, 1.0].forEach(factor => {
        ctx.beginPath();
        ctx.arc(cx, cy, maxRadius * factor, 0, 2 * Math.PI);
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Salib Tengah
      ctx.beginPath();
      ctx.moveTo(cx, cy - maxRadius); ctx.lineTo(cx, cy + maxRadius);
      ctx.moveTo(cx - maxRadius, cy); ctx.lineTo(cx + maxRadius, cy);
      ctx.strokeStyle = '#1e293b';
      ctx.stroke();

      // Drone Icon di Pusat
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      ctx.fillStyle = '#38bdf8';
      ctx.fill();

      // Rintangan (Depan, Belakang, Kiri, Kanan)
      const dirs = [
        { name: 'front', dx: 0, dy: -1 },
        { name: 'back',  dx: 0, dy: 1 },
        { name: 'left',  dx: -1, dy: 0 },
        { name: 'right', dx: 1, dy: 0 }
      ];

      dirs.forEach(target => {
        const dist = lidar[target.name];
        if (dist !== null && !isNaN(dist) && dist < 10.0) {
          const r = Math.min(maxRadius, (dist / 10.0) * maxRadius);
          const px = cx + target.dx * r;
          const py = cy + target.dy * r;

          ctx.beginPath();
          ctx.arc(px, py, dist < 0.8 ? 6 : 4, 0, 2 * Math.PI);
          ctx.fillStyle = dist < 0.8 ? '#ef4444' : (dist < 1.5 ? '#f59e0b' : '#10b981');
          ctx.fill();
        }
      });
    }

    // Mulai polling
    updateTelemetry();
  </script>
</body>
</html>
"""


# ==============================================================================
# HTTP HANDLER (REST API & MJPEG VIDEO STREAM)
# ==============================================================================
class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Mute log default terminal agar output terminal tetap bersih
        return

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()

    def do_GET(self):
        parsed = self.path.split('?')[0]

        if parsed in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache, no-store')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            self.wfile.flush()

        elif parsed == '/api/telemetry':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            self.wfile.write(data_hub.get_telemetry_json().encode('utf-8'))
            self.wfile.flush()

        elif parsed in ('/stream/camera_front', '/stream/camera_down'):
            cam_type = 'front' if 'front' in parsed else 'down'
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                while data_hub.running:
                    jpeg = data_hub.get_jpeg(cam_type)
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
                    self.wfile.flush()
                    time.sleep(0.04)  # ~25 FPS
            except (BrokenPipeError, ConnectionResetError, Exception):
                pass
        else:
            self.send_response(404)
            self.end_headers()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Web Dashboard Telemetri Drone Gamaforce")
    parser.add_argument('--port', type=int, default=5000, help="Port Web Server (default: 5000)")
    args, _ = parser.parse_known_args()

    import signal

    # 1. Inisialisasi ROS 2
    rclpy.init()
    node = TelemetryListenerNode()

    # 2. Spin ROS 2 di background thread dengan SingleThreadedExecutor terisolasi
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin_worker():
        try:
            executor.spin()
        except (rclpy.executors.ExternalShutdownException, Exception):
            pass

    ros_thread = threading.Thread(target=spin_worker, daemon=True)
    ros_thread.start()

    # 3. Jalankan HTTP Web Server
    server_address = ('0.0.0.0', args.port)
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer(server_address, DashboardHTTPRequestHandler)

    def shutdown_signal(sig, frame):
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown_signal)
    signal.signal(signal.SIGTERM, shutdown_signal)

    print("\n" + "=" * 65)
    print("   GAMAFORCE DRONE TELEMETRY & VISION WEB DASHBOARD")
    print("=" * 65)
    print(f"[OK] Dashboard aktif dan mendengarkan pada seluruh interface!")
    print(f"[+] Buka di browser: http://localhost:{args.port}")
    print(f"[+] Menerima stream 2 kamera, 6 LiDAR, dan telemetry MAVROS.")
    print("    Tekan Ctrl+C di terminal ini untuk keluar.")
    print("=" * 65 + "\n")

    try:
        httpd.serve_forever()
    except Exception:
        pass
    finally:
        data_hub.running = False
        try:
            httpd.server_close()
        except Exception:
            pass
        try:
            executor.cancel()
            ros_thread.join(timeout=1.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        print("[OK] Selesai.")


if __name__ == '__main__':
    main()
