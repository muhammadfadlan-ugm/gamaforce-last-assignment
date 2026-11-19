#!/usr/bin/env python3
"""
MISI 3: NAVIGASI LENGKAP & KAMIKAZE TARGET NOMOR 2 (HIGH-SPEED & ZERO TIMEOUT)
=============================================================================
Implementasi algoritma misi 3 berdasarkan skrip_misi3.md dan revisi operasional:

1. Gerak Aktif ke Kiri (Sumbu X Negatif) pada Tahap 1:
   - Setelah keluar terowongan, drone meluncur aktif ke kiri (vy = -1.80 m/s)
     menuju zona lingkaran hijau (sumbu X negatif).
   - Memindai kamera bawah secara kontinu hingga lingkaran hijau terdeteksi.

2. Kecepatan Tinggi (3x Lipat):
   - Gerak lateral cari hijau: vy = -1.80 m/s.
   - Servoing centering: kecepatan max s.d. 1.0 m/s.
   - Menembus donat merah: sprint maju vx = 4.0 m/s.
   - Yaw alignment: laju putar hingga 1.8 rad/s (~100 deg/s).
   - Kamikaze charge: sprint tabrak vx = 7.0 m/s.

3. Zero Timeout (Deteksi Berkelanjutan Tanpa Batas):
   - Tidak ada batasan waktu yang membatalkan atau melompati tahap.
   - Drone memindai dan menyelesaikan rintangan secara kontinu sampai berhasil.

4. Kunci Sasaran Nomor 2 Setelah Donat Merah:
   - Selesai menembus donat, drone LANGSUNG menghadap ke nomor 2 (Yaw 90° lurus ke depan).
   - Kamera depan mendeteksi sasaran nomor 2 (HOG+SVM, Red Bullseye, Template Matching).
   - Centering Yaw & Pitch tepat ke titik tengah angka 2.
   - Drone terbang cepat meluncur menabrak angka nomor 2 (Kamikaze).

Cara Menjalankan:
  python3 scripts/misi3.py
  python3 scripts/misi3.py --step 3 --teleport
  python3 scripts/misi3.py --step 5 --teleport
"""

import sys
import os
import time
import math
import argparse
import threading
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

# Otomatis gunakan venv ArduPilot jika modul pymavlink belum terinstal di system
try:
    from pymavlink import mavutil
except ModuleNotFoundError:
    venv_py = os.path.expanduser('~/venv-ardupilot/bin/python3')
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise

# Impor ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from deteksi import (
    GreenRingDetector,
    GreenCenteringResult,
    RedDonutDetector,
    RedDonutResult,
    KamikazeDetector,
    KamikazeResult,
    DetectedTarget
)

mavlink = mavutil.mavlink
DEFAULT_KONEKSI = 'udpin:127.0.0.1:14560'
TINGGI_DEFAULT = 1.35


# ==============================================================================
# 1. SUBSISTEM SENSOR & VISION LISTENER (ROS 2)
# ==============================================================================
class MissionVisionListener:
    """
    Listener thread-safe untuk kamera dan LiDAR ROS 2 dengan deteksi real-time berkelanjutan.
    """
    def __init__(self, target_digit: int = 2):
        self.lock = threading.Lock()
        self.target_digit = target_digit

        # Inisialisasi Detektor Vision
        self.green_detector = GreenRingDetector(min_area=180.0, min_circularity=0.35, tolerance_px=22)
        self.donut_detector = RedDonutDetector(tolerance_px=25)
        self.kamikaze_detector = KamikazeDetector(target_digit=target_digit, tolerance_px=22)

        # Buffer Frame
        self.frame_front: Optional[np.ndarray] = None
        self.frame_down: Optional[np.ndarray] = None

        # Jarak LiDAR
        self.lidar_front: float = 99.0
        self.lidar_down: float = 99.0

        # Hasil Deteksi Terkini
        self.res_green: Optional[GreenCenteringResult] = None
        self.res_donut: Optional[RedDonutResult] = None
        self.res_kamikaze: Optional[KamikazeResult] = None

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

        res_d = self.donut_detector.detect(bgr, draw=True)
        res_k = self.kamikaze_detector.detect(bgr, target_digit=self.target_digit, draw=True)

        with self.lock:
            self.frame_front = bgr
            self.res_donut = res_d
            self.res_kamikaze = res_k

    def on_camera_down(self, msg: Image):
        bgr = self._imgmsg_to_bgr(msg)
        if bgr is None:
            return

        res_g = self.green_detector.detect(bgr, draw=True)

        with self.lock:
            self.frame_down = bgr
            self.res_green = res_g

    def on_lidar_front(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 25.0]
            with self.lock:
                self.lidar_front = min(valid) if valid else 99.0

    def on_lidar_down(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            with self.lock:
                self.lidar_down = min(valid) if valid else 99.0

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

    def get_annotated_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            f_down = self.res_green.annotated_frame if self.res_green is not None else self.frame_down
            f_front = self.res_kamikaze.annotated_frame if self.res_kamikaze is not None else self.frame_front
            return f_down, f_front


# ==============================================================================
# 2. SUBSISTEM KONTROL DRONE AUTOPILOT (HIGH SPEED)
# ==============================================================================
class DroneController:
    """
    Kontrol manuver drone lincah dan berkecepatan tinggi berbasis MAVLink.
    """
    def __init__(self, koneksi: str = DEFAULT_KONEKSI):
        print("=" * 60)
        print(f"[+] Menghubungkan ke SITL Autopilot: {koneksi} ...")
        self.m = mavutil.mavlink_connection(koneksi)
        self.m.mav.heartbeat_send(0, 0, 0, 0, 0)

        print("[*] Menunggu heartbeat autopilot...")
        self.m.wait_heartbeat(timeout=10)
        if self.m.target_system == 0:
            raise RuntimeError(
                "Gagal terhubung ke autopilot! Pastikan Terminal 2 (SITL) berjalan dengan opsi --out 127.0.0.1:14560."
            )

        print(f"[OK] Terhubung ke Drone Iris (System ID: {self.m.target_system})")
        self.pos_x: float = 0.0
        self.pos_y: float = 0.0
        self.pos_z: float = 0.0
        self.yaw_deg: float = 90.0
        self.armed: bool = False
        self.flightmode: str = "UNKNOWN"

        self._minta_stream(32, 25)  # LOCAL_POSITION_NED
        self._minta_stream(30, 25)  # ATTITUDE
        time.sleep(0.8)
        self.update()

    def _minta_stream(self, msg_id: int, hz: int):
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0
        )

    def update(self):
        """Membaca telemetri drone non-blocking."""
        msg = self.m.recv_match(blocking=False)
        while msg is not None:
            t = msg.get_type()
            if t == 'LOCAL_POSITION_NED':
                self.pos_x = msg.x
                self.pos_y = msg.y
                self.pos_z = msg.z
            elif t == 'ATTITUDE':
                deg = math.degrees(msg.yaw)
                if deg < 0:
                    deg += 360.0
                self.yaw_deg = deg
            elif t == 'HEARTBEAT' and msg.get_srcSystem() == self.m.target_system:
                self.armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.flightmode = self.m.flightmode
            msg = self.m.recv_match(blocking=False)

    def set_mode(self, mode: str, timeout: float = 5.0) -> bool:
        self.m.set_mode(mode)
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.update()
            if self.m.flightmode == mode:
                return True
            time.sleep(0.05)
        return False

    def send_velocity_body(self, vx: float, vy: float, vz_up: float, yaw_rate: float = 0.0):
        """
        Mengirim kecepatan translasi BODY FRAME:
        vx > 0: Maju,      vx < 0: Mundur
        vy > 0: Kanan,     vy < 0: Kiri (sumbu X negatif saat yaw=90°)
        vz_up > 0: Naik,   vz_up < 0: Turun
        """
        vz_ned = -vz_up
        # type_mask 0b010111000111 (1479): Enable vx, vy, vz; Ignore px, py, pz, ax, ay, az, yaw
        type_mask = 0b010111000111
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            mavlink.MAV_FRAME_BODY_OFFSET_NED,
            type_mask,
            0, 0, 0,
            vx, vy, vz_ned,
            0, 0, 0,
            0, yaw_rate
        )

    def move_timed(self, vx: float, vy: float, vz_up: float, detik: float, yaw_rate: float = 0.0):
        t_end = time.time() + detik
        while time.time() < t_end:
            self.send_velocity_body(vx, vy, vz_up, yaw_rate)
            self.update()
            time.sleep(0.04)
        self.send_velocity_body(0, 0, 0, 0)

    def brake(self, detik: float = 0.4):
        self.move_timed(0, 0, 0, detik)

    def takeoff_if_needed(self, tinggi: float = TINGGI_DEFAULT):
        self.update()
        if not self.armed or (-self.pos_z < 0.35):
            print(f"[*] Drone belum di udara. Takeoff ke {tinggi:.2f} m...")
            self.set_mode('GUIDED')
            for _ in range(5):
                self.m.mav.command_long_send(
                    self.m.target_system, self.m.target_component,
                    mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                    1, 21196, 0, 0, 0, 0, 0
                )
                time.sleep(0.3)
                self.update()
                if self.armed:
                    break

            self.m.mav.command_long_send(
                self.m.target_system, self.m.target_component,
                mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                0, 0, 0, 0, 0, 0, tinggi
            )

            while -self.pos_z < tinggi * 0.85:
                self.update()
                time.sleep(0.1)
            print(f"[OK] Ketinggian take off tercapai ({-self.pos_z:.2f} m).")
            self.brake(0.8)

    def set_yaw(self, target_yaw_deg: float, speed_deg_s: float = 75.0, timeout: float = 3.0):
        """Memutar hadap yaw drone ke sudut target secara aman menggunakan MAV_CMD_CONDITION_YAW."""
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_CONDITION_YAW,
            0,
            target_yaw_deg, speed_deg_s, 0, 0, 0, 0, 0
        )
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.update()
            err = (target_yaw_deg - self.yaw_deg + 180.0) % 360.0 - 180.0
            if abs(err) <= 5.0:
                break
            time.sleep(0.05)
        self.brake(0.2)

    def land(self):
        print("\n[*] Mendaratkan drone...")
        self.set_mode('LAND')


# ==============================================================================
# 3. IMPLEMENTASI 10 TAHAP ALGORITMA (EVENT-DRIVEN CALLBACK, KECEPATAN TINGGI)
# ==============================================================================

def tahap_1_cari_lingkaran_hijau(drone: DroneController, sensor: MissionVisionListener):
    """
    TAHAP 1:
    Setelah keluar terowongan, drone melaju ke kiri (sumbu X negatif: vy = -1.25 m/s)
    sambil memproses callback kamera bawah (/camera_down) secara kontinu hingga
    lingkaran hijau tertangkap oleh detektor (tanpa timer buta atau estimasi meter).
    """
    print("\n" + "=" * 65)
    print(">>> [TAHAP 1] BERGERAK KE KIRI (SUMBU X NEGATIF) MENCARI LINGKARAN HIJAU")
    print(">>> (Tracking Murni Berbasis Callback Deteksi Kamera Bawah | Sumbu -X)")
    print("=" * 65)

    drone.update()
    drone.brake(0.3)

    print("[*] Melaju lurus murni ke KIRI (sumbu X negatif, vy = -1.25 m/s)...")
    print("[*] Memindai feed /camera_down dari callback secara kontinu...")

    step_cnt = 0
    while rclpy.ok():
        drone.update()
        green = sensor.get_green()

        # Deteksi lingkaran hijau secara reaktif langsung dari callback
        if green is not None and green.detected and green.circularity >= 0.35 and green.area >= 180.0:
            print(f"\n[OK] LINGKARAN HIJAU TERDETEKSI OLEH CALLBACK!")
            print(f"     -> Posisi: ({green.cx}, {green.cy}) | Area: {green.area:.0f} px | Sirkularitas: {green.circularity:.2f}")
            drone.brake(0.3)
            break

        # Pertahankan gerak murni ke kiri (sumbu X negatif) tanpa memutar yaw
        drone.send_velocity_body(vx=0.0, vy=-1.25, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)
        step_cnt += 1
        if step_cnt % 25 == 0:
            print(f"  [*] Mencari lingkaran hijau ke kiri... (Posisi drone: X={drone.pos_x:.2f}, Y={drone.pos_y:.2f})")


def tahap_2_centering_lingkaran_hijau(drone: DroneController, sensor: MissionVisionListener):
    """
    TAHAP 2:
    Centering posisi X dan Y terhadap lingkaran hijau berbasis visual servoing
    (umpan balik piksel dx, dy langsung dari callback kamera bawah).
    """
    print("\n" + "=" * 65)
    print(">>> [TAHAP 2] CENTERING POSISI X DAN Y TERHADAP LINGKARAN HIJAU")
    print(">>> (Visual Servoing Closed-Loop: P-Controller Reaktif | Toleransi 22px)")
    print("=" * 65)

    steady_count = 0

    while rclpy.ok():
        drone.update()
        green = sensor.get_green()

        if green is None or not green.detected:
            # Jika sesaat hilang dari pandangan, pertahankan posisi (hover)
            drone.send_velocity_body(0.0, 0.0, 0.0, yaw_rate=0.0)
            time.sleep(0.04)
            continue

        dx = green.dx
        dy = green.dy

        # Visual Servoing P-Controller:
        # Kamera bawah: dy > 0 berarti target di bawah frame (perlu mundur / vx < 0)
        #               dx > 0 berarti target di kanan frame (perlu geser kanan / vy > 0)
        vx_cmd = np.clip(-0.0055 * dy, -0.85, 0.85)
        vy_cmd = np.clip( 0.0055 * dx, -0.85, 0.85)

        if abs(dx) <= 22 and abs(dy) <= 22:
            steady_count += 1
            drone.send_velocity_body(0.0, 0.0, 0.0, yaw_rate=0.0)
            if steady_count >= 5:  # Stabil di tengah selama ~0.2 detik
                print(f"[OK] Centering Lingkaran Hijau SELESAI! (dx={dx:+d}, dy={dy:+d})")
                drone.brake(0.4)
                break
        else:
            steady_count = 0
            drone.send_velocity_body(vx_cmd, vy_cmd, 0.0, yaw_rate=0.0)

        time.sleep(0.04)


def tahap_3_deteksi_dan_centering_donat_merah(drone: DroneController, sensor: MissionVisionListener):
    """
    TAHAP 3:
    Mendeteksi lingkaran merah dan centering titik tengah lubang donat:
    - Drone melayang di tempat (vx = 0.0) dan melakukan tracking ketinggian naik/turun
      terlebih dahulu hingga donat merah tertangkap jelas di kamera depan.
    - Melakukan visual servoing centering (mengatur ketinggian Z dan lateral Y)
      hingga titik tengah kamera depan sejajar sempurna dengan titik tengah donat merah.
    """
    print("\n" + "=" * 65)
    print(">>> [TAHAP 3] TRACKING NAIK-TURUN & CENTERING TITIK PUSAT DONAT MERAH")
    print(">>> (Hover di Tempat vx=0.0 | Tracking Ketinggian Naik-Turun | Centering Lubang)")
    print("=" * 65)

    drone.brake(0.4)

    steady_count = 0
    search_phase = 0
    search_timer = time.time()

    print("[*] Melayang di tempat (vx=0.0). Memulai tracking naik-turun mencari donat merah...")

    while rclpy.ok():
        drone.update()
        donut = sensor.get_donut()

        if donut is not None and donut.detected:
            dx = donut.dx
            dy = donut.dy

            # Visual Servoing Centering:
            # dy > 0: donat di bawah tengah kamera -> turun (vz_up < 0)
            # dy < 0: donat di atas tengah kamera  -> naik  (vz_up > 0)
            # dx > 0: donat di kanan kamera       -> geser kanan (vy > 0)
            # dx < 0: donat di kiri kamera        -> geser kiri  (vy < 0)
            vy_cmd = np.clip( 0.0045 * dx, -0.65, 0.65)
            vz_cmd = np.clip(-0.0055 * dy, -0.75, 0.75)

            # PENTING: vx = 0.0 (jangan maju dulu sebelum tengah kamera sejajar dengan donat!)
            if abs(dx) <= 22 and abs(dy) <= 22:
                steady_count += 1
                drone.send_velocity_body(0.0, 0.0, 0.0, yaw_rate=0.0)
                if steady_count >= 5:  # Stabil di tengah selama ~0.2 detik
                    hole_str = "Lubang Tembus" if donut.has_hole else "Pusat Donat"
                    print(f"\n[OK] CENTERING DONAT MERAH SELESAI!")
                    print(f"     -> Status: {hole_str} Terkunci | Offset: dx={dx:+d}, dy={dy:+d} | Ketinggian: {-drone.pos_z:.2f} m")
                    drone.brake(0.4)
                    break
            else:
                steady_count = 0
                drone.send_velocity_body(vx=0.0, vy=vy_cmd, vz_up=vz_cmd, yaw_rate=0.0)

            if steady_count == 0 and int(time.time() * 5) % 5 == 0:
                print(f"  [TRACKING DONAT] dx={dx:+3d}, dy={dy:+3d} | vz_cmd={vz_cmd:+.2f} m/s, vy_cmd={vy_cmd:+.2f} m/s | Alt={-drone.pos_z:.2f}m")

        else:
            steady_count = 0
            alt = -drone.pos_z

            # Tracking ketinggian naik-turun mencari donat merah (donat berada di Z ~ 2.6 m)
            # Jika drone masih rendah (Z < 2.5 m), naik ke atas mencari donat
            if alt < 2.70:
                vz_search = 0.55
                search_label = "Naik vertikal mencari donat (Z ~ 2.6 m)"
            else:
                vz_search = -0.45
                search_label = "Turun vertikal mencari donat"

            now = time.time()
            if now - search_timer > 0.8:
                print(f"  [*] {search_label}... (Ketinggian saat ini: {alt:.2f} m)")
                search_timer = now

            # Pastikan drone tidak melaju maju (vx = 0.0) saat mencari
            drone.send_velocity_body(vx=0.0, vy=0.0, vz_up=vz_search, yaw_rate=0.0)

        time.sleep(0.04)


def tahap_4_menembus_donat_merah(drone: DroneController):
    """
    TAHAP 4:
    Melaju cepat ke depan tepat di tengah lubang donat merah (vx = 3.5 m/s),
    lalu melambat dan berhenti setelah melewati donat untuk tracking sasaran Bulls Eye.
    """
    print("\n" + "=" * 65)
    print(">>> [TAHAP 4] MELAJU KEDEPAN MENEMBUS LUBANG DONAT MERAH")
    print(">>> (Kecepatan Tinggi: vx = 3.5 m/s | Berhenti Setelah Melewati Donat)")
    print("=" * 65)

    print("[*] SPRINT MAJU menembus lubang donat merah...")
    # Menempuh jarak ~6.8 meter pada 3.5 m/s (dari Y ~ 12.0 ke seberang donat di Y ~ 18.8)
    t_end = time.time() + 2.0
    while time.time() < t_end and rclpy.ok():
        drone.send_velocity_body(vx=3.5, vy=0.0, vz_up=0.0, yaw_rate=0.0)
        drone.update()
        time.sleep(0.03)

    # Melambat dan berhenti sempurna setelah menembus donat sesuai permintaan algoritma
    print("[*] Selesai melewati donat merah. Melambat dan berhenti total...")
    drone.brake(1.0)
    print("[OK] Drone telah melewati donat merah dengan aman dan berhenti.")

    # Menyesuaikan ketinggian turun sedikit agar sejajar dengan target Bulls Eye (Z ~ 1.5 m)
    print("[*] Menyesuaikan ketinggian ke level target Bulls Eye (Z ~ 1.5 m)...")
    t_descend = time.time() + 1.8
    while time.time() < t_descend and rclpy.ok():
        drone.update()
        if -drone.pos_z <= 1.55:
            break
        drone.send_velocity_body(vx=0.0, vy=0.0, vz_up=-0.50, yaw_rate=0.0)
        time.sleep(0.04)

    drone.brake(0.4)
    print(f"[OK] Posisi stabil di seberang donat merah (Ketinggian: {-drone.pos_z:.2f} m).")


def tahap_5_hingga_10_kunci_dan_kamikaze_nomor_2(
    drone: DroneController,
    sensor: MissionVisionListener,
    target_digit: int = 2
):
    """
    TAHAP 5 s.d. 10:
    Setelah menyelesaikan obstacle donat merah:
    - Drone LANGSUNG menghadap ke angka nomor 2 (Yaw 90° lurus ke depan tempat target 2 berada).
    - Kamera depan mendeteksi nomor 2 secara kontinu tanpa timeout.
    - Centering Yaw & Pitch tepat ke titik tengah angka 2.
    - Kamikaze Sprint: melaju cepat 3x lipat (vx = 7.0 m/s) menabrak nomor 2.
    """
    print("\n" + "=" * 65)
    print(f">>> [TAHAP 5 s.d. 10] MENGHADAP & KAMIKAZE TARGET NOMOR {target_digit}")
    print(">>> (Hadap Langsung ke Sasaran 2 | Zero Timeout | Kecepatan Tabrak 7.0 m/s)")
    print("=" * 65)

    # 1. POSISIKAN DRONE MENGHADAP LURUS KE DEPAN (Arah Sasaran Bulls Eye #2)
    drone.brake(0.4)

    # 2. DETEKSI BERKELANJUTAN TANPA TIMEOUT HINGGA NOMOR 2 TERKUNCI
    print(f"[*] Mendeteksi target #{target_digit} secara kontinu di kamera depan...")
    step_cnt = 0

    while rclpy.ok():
        drone.update()
        res_k = sensor.get_kamikaze()

        if res_k is not None:
            # Cari sasaran nomor 2 dari deteksi HOG+SVM atau Bullseye Merah
            matching = [t for t in res_k.all_targets if t.digit == target_digit]
            if res_k.locked_target and res_k.locked_target.digit == target_digit:
                matching.append(res_k.locked_target)

            if matching:
                best = max(matching, key=lambda t: t.area)
                print(f"[OK] TARGET #{target_digit} TERDETEKSI & TERKUNCI!")
                print(f"     -> Koordinat: ({best.cx}, {best.cy}) | Offset: dx={best.dx:+d}, dy={best.dy:+d}")
                break

        # Jika belum tampak jelas, lakukan penyelarasan maju perlahan
        step_cnt += 1
        if step_cnt % 30 == 0:
            print(f"  [*] Memindai kamera depan mencari Bulls Eye #{target_digit}...")
        drone.send_velocity_body(vx=0.15, vy=0.0, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)

    # 3. CENTERING YAW & PITCH/KETINGGIAN TEPAT DI TITIK TENGAH NOMOR 2
    print(f"\n[*] Menyelaraskan Yaw & Ketinggian presisi ke titik tengah nomor {target_digit}...")
    steady_aim = 0

    while rclpy.ok():
        drone.update()
        res_k = sensor.get_kamikaze()

        target_now = None
        if res_k is not None:
            matching = [t for t in res_k.all_targets if t.digit == target_digit]
            if res_k.locked_target and res_k.locked_target.digit == target_digit:
                matching.append(res_k.locked_target)
            if matching:
                target_now = max(matching, key=lambda t: t.area)

        if target_now is not None:
            dx = target_now.dx
            dy = target_now.dy

            yaw_rate_cmd = np.clip(0.006 * dx, -1.2, 1.2)
            vz_cmd = np.clip(-0.005 * dy, -0.6, 0.6)

            if abs(dx) <= 20 and abs(dy) <= 20:
                steady_aim += 1
                if steady_aim >= 4:
                    print(f"[OK] RETICLE TERKUNCI SEMPURNA PADA TARGET #{target_digit}! (dx={dx:+d}, dy={dy:+d})")
                    break
            else:
                steady_aim = 0
            drone.send_velocity_body(vx=0.15, vy=0, vz_up=vz_cmd, yaw_rate=yaw_rate_cmd)
        else:
            drone.send_velocity_body(vx=0.10, vy=0.0, vz_up=0.0, yaw_rate=0.0)

        time.sleep(0.03)

    drone.brake(0.3)

    # 4. TAHAP 10: KAMIKAZE CHARGE (TERBANG CEPAT 7.0 M/S HINGGA MENABRAK NOMOR 2)
    print("\n" + "=" * 65)
    print(f"🚀🚀 KAMIKAZE SPRINT: TERBANG PENUH KILAT MENABRAK NOMOR {target_digit}! 🚀🚀")
    print(">>> Kecepatan: vx = 7.0 m/s (3x Lebih Cepat) | Tracking Aktif Menuju Tabrakan")
    print("=" * 65)

    KECEPATAN_TABRAK = 7.0  # 3x lebih cepat dari sebelumnya (2.4 m/s -> 7.0 m/s)

    while rclpy.ok():
        drone.update()
        jarak_lidar = sensor.get_lidar_front()

        # Deteksi tabrakan via sensor jarak LiDAR depan
        if jarak_lidar <= 0.45:
            print(f"\n💥💥💥 IMPACT TABRAKAN TERJADI! Jarak LiDAR depan: {jarak_lidar:.2f} m 💥💥💥")
            drone.send_velocity_body(vx=KECEPATAN_TABRAK, vy=0, vz_up=0)
            time.sleep(0.2)
            break

        # Visual tracking aktif agar hidung drone tetap presisi mengarah ke nomor 2 saat melesat
        res_k = sensor.get_kamikaze()
        yaw_trim = 0.0
        vz_trim = 0.0

        if res_k is not None:
            matching = [t for t in res_k.all_targets if t.digit == target_digit]
            if res_k.locked_target and res_k.locked_target.digit == target_digit:
                matching.append(res_k.locked_target)
            if matching:
                t_cur = max(matching, key=lambda t: t.area)
                yaw_trim = np.clip(0.0035 * t_cur.dx, -0.8, 0.8)
                vz_trim = np.clip(-0.0030 * t_cur.dy, -0.5, 0.5)

        drone.send_velocity_body(vx=KECEPATAN_TABRAK, vy=0, vz_up=vz_trim, yaw_rate=yaw_trim)
        time.sleep(0.03)

    drone.brake(0.5)
    print("\n" + "=" * 65)
    print(f"🎉🎉 MISI 3 SUKSES PENUH! DRONE BERHASIL MENABRAK SASARAN NOMOR {target_digit}! 🎉🎉")
    print("=" * 65)


# ==============================================================================
# 4. FUNGSI UTAMA & EKSEKUSI CLI
# ==============================================================================
def run_teleport_preset(preset_key: str):
    teleport_py = os.path.join(SCRIPTS_DIR, "teleport.py")
    if not os.path.exists(teleport_py):
        return

    coords_map = {
        "1": (13.0, 5.0, 1.3, 90.0),
        "2": (3.3, 11.5, 1.3, 90.0),
        "3": (3.3, 12.0, 1.4, 90.0),   # Di atas lingkaran hijau, siap tracking donat merah
        "4": (3.3, 14.0, 2.6, 90.0),   # Tepat di depan lubang donat merah (Z ~ 2.6 m)
        "5": (3.3, 18.5, 1.5, 90.0),   # Tepat setelah donat merah, siap tracking target 2
    }
    if preset_key in coords_map:
        x, y, z, yaw = coords_map[preset_key]
        import subprocess
        print(f"[*] Teleport otomatis ke tahap {preset_key}: X={x}, Y={y}, Z={z}, Yaw={yaw}°...")
        subprocess.run([sys.executable, teleport_py, str(x), str(y), str(z), str(yaw)])
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Misi 3: Navigasi Cepat & Kamikaze Target 2 (Zero Timeout)")
    parser.add_argument("--koneksi", default=DEFAULT_KONEKSI, help="Koneksi MAVLink (default: udpin:127.0.0.1:14560)")
    parser.add_argument("--step", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Mulai dari tahap tertentu: 1=Terowongan, 2=Lingkaran Hijau, 3=Donat Merah, 4=Tembus Donat, 5=Kamikaze Bulls Eye")
    parser.add_argument("--target", type=int, default=2, help="Nomor target Bulls Eye sasaran (default: 2)")
    parser.add_argument("--teleport", action="store_true", help="Teleport otomatis ke posisi awal tahap yang dipilih")
    parser.add_argument("--headless", action="store_true", help="Jalankan tanpa window visual GUI OpenCV")
    args = parser.parse_args()

    print("=================================================================")
    print("  GAMAFORCE - MISI 3: HIGH-SPEED NAVIGATION & BULLSEYE KAMIKAZE  ")
    print("=================================================================")
    print(f"[*] Target Sasaran Kamikaze: Nomor {args.target}")
    print(f"[*] Tahap Awal Eksekusi    : Tahap {args.step}")

    if args.teleport:
        run_teleport_preset(str(args.step))

    rclpy.init()
    listener = MissionVisionListener(target_digit=args.target)
    node = Node("misi3_executive_node")

    node.create_subscription(Image, "/camera_down", listener.on_camera_down, 10)
    node.create_subscription(Image, "/camera_front", listener.on_camera_front, 10)
    node.create_subscription(LaserScan, "/lidar_front", listener.on_lidar_front, 10)
    node.create_subscription(LaserScan, "/lidar_down", listener.on_lidar_down, 10)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    print("[OK] Subsistem ROS 2 Kamera & LiDAR aktif.")

    drone = DroneController(koneksi=args.koneksi)
    drone.takeoff_if_needed(TINGGI_DEFAULT)

    has_display = bool(os.environ.get("DISPLAY")) and not args.headless
    gui_running = [True]

    def gui_loop():
        while gui_running[0] and rclpy.ok():
            f_down, f_front = listener.get_annotated_frames()
            if f_down is not None:
                cv2.imshow("Misi 3 - Kamera Bawah (Lingkaran Hijau)", f_down)
            if f_front is not None:
                cv2.imshow("Misi 3 - Kamera Depan (Donat & Bulls Eye)", f_front)
            if f_down is not None or f_front is not None:
                if cv2.waitKey(20) & 0xFF in (27, ord('q')):
                    break
            else:
                time.sleep(0.04)
        cv2.destroyAllWindows()

    if has_display:
        gui_th = threading.Thread(target=gui_loop, daemon=True)
        gui_th.start()

    try:
        # Eksekusi tahap berurutan tanpa timeout
        if args.step <= 1:
            tahap_1_cari_lingkaran_hijau(drone, listener)

        if args.step <= 2:
            tahap_2_centering_lingkaran_hijau(drone, listener)

        if args.step <= 3:
            tahap_3_deteksi_dan_centering_donat_merah(drone, listener)

        if args.step <= 4:
            tahap_4_menembus_donat_merah(drone)

        if args.step <= 5:
            tahap_5_hingga_10_kunci_dan_kamikaze_nomor_2(drone, listener, target_digit=args.target)

    except KeyboardInterrupt:
        print("\n[!] Misi dihentikan oleh pengguna (Ctrl+C).")
        drone.brake(0.5)
    except Exception as e:
        print(f"\n[!] Terjadi error dalam eksekusi: {e}")
        import traceback
        traceback.print_exc()
        drone.brake(0.5)
    finally:
        gui_running[0] = False
        print("\n[*] Menyelesaikan sesi...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
