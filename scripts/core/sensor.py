#!/usr/bin/env python3
"""
================================================================================
GAMAFORCE CORE - UNIFIED SENSOR & VISION LISTENER (ROS 2)
================================================================================
Modul sensor tunggal yang mencakup:
- 6 LiDAR 1D: Front, Back, Left, Right, Up, Down (LaserScan)
- 2 Kamera: Camera Front, Camera Down (Image)
- Integrasi pipeline pengolahan citra OpenCV (Pad Hijau, Donat Merah, Bulls Eye)
- Meneruskan data telemetri & stream video hasil pengolahan citra OpenCV
  ke Web Dashboard (baca_semua_data.py) secara real-time.
================================================================================
"""

import sys
import os
import time
import math
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, LaserScan

# Pastikan modul deteksi terdeteksi
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CORE_DIR)
DETEKSI_DIR = os.path.join(SCRIPTS_DIR, "deteksi")
for p in (SCRIPTS_DIR, DETEKSI_DIR, CORE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from deteksi import (
    GreenRingDetector,
    GreenCenteringResult,
    RedDonutDetector,
    RedDonutResult,
    KamikazeDetector,
    KamikazeResult,
    DetectedTarget
)

# Integrasi opsional ke Web Dashboard baca_semua_data.py
try:
    from baca_semua_data import data_hub
except Exception:
    data_hub = None


class SensorListener:
    """
    Listener thread-safe untuk seluruh sensor LiDAR dan Kamera ROS 2.
    """
    def __init__(self, target_digit: int = 2):
        self.lock = threading.Lock()
        self.target_digit = target_digit

        # Inisialisasi Detektor Vision OpenCV
        self.green_detector = GreenRingDetector(min_area=180.0, min_circularity=0.35, tolerance_px=22)
        self.donut_detector = RedDonutDetector(tolerance_px=25)
        self.kamikaze_detector = KamikazeDetector(target_digit=target_digit, tolerance_px=22)

        # Buffer Frame Kamera Mentah
        self.frame_front: Optional[np.ndarray] = None
        self.frame_down: Optional[np.ndarray] = None

        # Jarak LiDAR (6 Arah Lengkap)
        self.lidar_front: float = 99.0
        self.lidar_back: float = 99.0
        self.lidar_left: float = 99.0
        self.lidar_right: float = 99.0
        self.lidar_up: float = 99.0
        self.lidar_down: float = 99.0

        # Hasil Deteksi Terkini
        self.res_green: Optional[GreenCenteringResult] = None
        self.res_donut: Optional[RedDonutResult] = None
        self.res_kamikaze: Optional[KamikazeResult] = None

        # Mode Deteksi Kamera Depan: "DONUT", "KAMIKAZE", "ALL"
        self.mode: str = "DONUT"

    def set_mode(self, mode: str):
        """Mengganti mode deteksi kamera depan secara eksklusif."""
        with self.lock:
            self.mode = mode
            if mode == "DONUT":
                self.res_kamikaze = None
            elif mode == "KAMIKAZE":
                self.res_donut = None

    def _imgmsg_to_bgr(self, msg: Image) -> Optional[np.ndarray]:
        try:
            if msg.encoding in ('bgr8', 'rgb8'):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else arr
            elif msg.encoding == 'mono8':
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        except Exception:
            pass
        return None

    def on_camera_front(self, msg: Image):
        bgr = self._imgmsg_to_bgr(msg)
        if bgr is None:
            return

        with self.lock:
            cur_mode = self.mode

        res_d = None
        res_k = None
        annotated_out = bgr

        if cur_mode == "KAMIKAZE":
            res_k = self.kamikaze_detector.detect(bgr, target_digit=self.target_digit, draw=True)
            if res_k and res_k.annotated_frame is not None:
                annotated_out = res_k.annotated_frame
        else:
            # Mode DONUT / GREEN: HANYA fokus pada donat merah
            res_d = self.donut_detector.detect(bgr, draw=True)
            if res_d and res_d.annotated_frame is not None:
                annotated_out = res_d.annotated_frame

        with self.lock:
            self.frame_front = bgr
            self.res_donut = res_d
            self.res_kamikaze = res_k

        # Teruskan stream hasil olah citra OpenCV ke Web Dashboard (baca_semua_data.py)
        if data_hub is not None and annotated_out is not None:
            try:
                ret, jpeg = cv2.imencode('.jpg', annotated_out, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    with data_hub.lock:
                        data_hub.jpeg_front = jpeg.tobytes()
                        data_hub.has_front = True
                        data_hub.telemetry['cam_front_active'] = True
            except Exception:
                pass

    def on_camera_down(self, msg: Image):
        bgr = self._imgmsg_to_bgr(msg)
        if bgr is None:
            return

        res_g = self.green_detector.detect(bgr, draw=True)
        annotated_out = res_g.annotated_frame if (res_g and res_g.annotated_frame is not None) else bgr

        with self.lock:
            self.frame_down = bgr
            self.res_green = res_g

        # Teruskan stream hasil olah citra OpenCV ke Web Dashboard (baca_semua_data.py)
        if data_hub is not None and annotated_out is not None:
            try:
                ret, jpeg = cv2.imencode('.jpg', annotated_out, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    with data_hub.lock:
                        data_hub.jpeg_down = jpeg.tobytes()
                        data_hub.has_down = True
                        data_hub.telemetry['cam_down_active'] = True
            except Exception:
                pass

    def on_lidar_front(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 25.0]
            val = min(valid) if valid else 99.0
            with self.lock:
                self.lidar_front = val
            if data_hub is not None:
                data_hub.update_lidar('front', msg)

    def on_lidar_back(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            val = min(valid) if valid else 99.0
            with self.lock:
                self.lidar_back = val
            if data_hub is not None:
                data_hub.update_lidar('back', msg)

    def on_lidar_left(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            val = min(valid) if valid else 99.0
            with self.lock:
                self.lidar_left = val
            if data_hub is not None:
                data_hub.update_lidar('left', msg)

    def on_lidar_right(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            val = min(valid) if valid else 99.0
            with self.lock:
                self.lidar_right = val
            if data_hub is not None:
                data_hub.update_lidar('right', msg)

    def on_lidar_up(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            val = min(valid) if valid else 99.0
            with self.lock:
                self.lidar_up = val
            if data_hub is not None:
                data_hub.update_lidar('up', msg)

    def on_lidar_down(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            val = min(valid) if valid else 99.0
            with self.lock:
                self.lidar_down = val
            if data_hub is not None:
                data_hub.update_lidar('down', msg)

    # --------------------------------------------------------------------------
    # GETTERS DATA SENSOR & HASIL VISION
    # --------------------------------------------------------------------------
    def get_green(self) -> Optional[GreenCenteringResult]:
        with self.lock:
            return self.res_green

    def get_donut(self) -> Optional[RedDonutResult]:
        with self.lock:
            return self.res_donut

    def get_kamikaze(self) -> Optional[KamikazeResult]:
        with self.lock:
            return self.res_kamikaze

    def get_lidar_front(self) -> float:
        with self.lock:
            return self.lidar_front

    def get_lidar_back(self) -> float:
        with self.lock:
            return self.lidar_back

    def get_lidar_left(self) -> float:
        with self.lock:
            return self.lidar_left

    def get_lidar_right(self) -> float:
        with self.lock:
            return self.lidar_right

    def get_lidar_up(self) -> float:
        with self.lock:
            return self.lidar_up

    def get_lidar_down(self) -> float:
        with self.lock:
            return self.lidar_down

    def get_frame_front(self) -> Optional[np.ndarray]:
        """Mengambil salinan frame mentah terbaru dari kamera depan."""
        with self.lock:
            return self.frame_front.copy() if self.frame_front is not None else None

    def get_frame_down(self) -> Optional[np.ndarray]:
        """Mengambil salinan frame mentah terbaru dari kamera bawah."""
        with self.lock:
            return self.frame_down.copy() if self.frame_down is not None else None

    def get_annotated_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            f_down = self.res_green.annotated_frame if self.res_green is not None else self.frame_down
            if self.mode == "KAMIKAZE" and self.res_kamikaze is not None and self.res_kamikaze.annotated_frame is not None:
                f_front = self.res_kamikaze.annotated_frame
            elif self.res_donut is not None and self.res_donut.annotated_frame is not None:
                f_front = self.res_donut.annotated_frame
            else:
                f_front = self.frame_front
            return f_down, f_front


# ------------------------------------------------------------------------------
# SHARED INSTANCE & ROS 2 NODE MANAGER (SINGLETON)
# ------------------------------------------------------------------------------
_global_sensor: Optional[SensorListener] = None
_global_node: Optional[Node] = None
_global_executor: Optional[SingleThreadedExecutor] = None
_global_thread: Optional[threading.Thread] = None

def get_sensor(target_digit: int = 2) -> SensorListener:
    """Mengembalikan instance SensorListener tunggal dan mengaktifkan node ROS 2 di background."""
    global _global_sensor, _global_node, _global_executor, _global_thread
    if not rclpy.ok():
        rclpy.init()

    if _global_sensor is None:
        _global_sensor = SensorListener(target_digit=target_digit)
        _global_node = Node("core_sensor_manager_node")

        _global_node.create_subscription(Image, "/camera_down", _global_sensor.on_camera_down, 10)
        _global_node.create_subscription(Image, "/camera_front", _global_sensor.on_camera_front, 10)
        _global_node.create_subscription(LaserScan, "/lidar_front", _global_sensor.on_lidar_front, 10)
        _global_node.create_subscription(LaserScan, "/lidar_back", _global_sensor.on_lidar_back, 10)
        _global_node.create_subscription(LaserScan, "/lidar_left", _global_sensor.on_lidar_left, 10)
        _global_node.create_subscription(LaserScan, "/lidar_right", _global_sensor.on_lidar_right, 10)
        _global_node.create_subscription(LaserScan, "/lidar_up", _global_sensor.on_lidar_up, 10)
        _global_node.create_subscription(LaserScan, "/lidar_down", _global_sensor.on_lidar_down, 10)

        _global_executor = SingleThreadedExecutor()
        _global_executor.add_node(_global_node)
        _global_thread = threading.Thread(target=_global_executor.spin, daemon=True)
        _global_thread.start()
        print("[OK] Core SensorListener ROS 2 aktif (6 LiDAR + 2 Kamera OpenCV Hub).")

    return _global_sensor
