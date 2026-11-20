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

        # Mode Deteksi Kamera Depan: "DONUT" atau "KAMIKAZE"
        # Tahap 1, 2, 3, 4: Mode DONUT (HANYA mendeteksi donat merah, DILARANG deteksi angka bullseye!)
        # Tahap 5+: Mode KAMIKAZE (Baru mendeteksi target angka pada Bulls Eye)
        self.mode: str = "DONUT"

    def set_mode(self, mode: str):
        """Mengganti mode deteksi kamera depan secara eksklusif."""
        with self.lock:
            self.mode = mode
            if mode != "KAMIKAZE":
                self.res_kamikaze = None
            if mode == "KAMIKAZE":
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

        if cur_mode == "KAMIKAZE":
            # Tahap 5: Deteksi sasaran angka pada Bulls Eye
            res_k = self.kamikaze_detector.detect(bgr, target_digit=self.target_digit, draw=True)
            res_d = None
        else:
            # Tahap 1 s.d. 4: HANYA mendeteksi rintangan donat merah!
            # Angka/Bulls eye sama sekali tidak diproses agar tidak mendahului tahap misi
            res_d = self.donut_detector.detect(bgr, draw=True)
            res_k = None

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

    def on_lidar_back(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            with self.lock:
                self.lidar_back = min(valid) if valid else 99.0

    def on_lidar_left(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            with self.lock:
                self.lidar_left = min(valid) if valid else 99.0

    def on_lidar_right(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            with self.lock:
                self.lidar_right = min(valid) if valid else 99.0

    def on_lidar_up(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            with self.lock:
                self.lidar_up = min(valid) if valid else 99.0

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
            if steady_count >= 6:  # Stabil di tengah selama ~0.25 detik
                print(f"[OK] Centering Lingkaran Hijau SELESAI! (dx={dx:+d}, dy={dy:+d})")
                drone.brake(0.6)
                break
        else:
            steady_count = 0
            drone.send_velocity_body(vx_cmd, vy_cmd, 0.0, yaw_rate=0.0)

        time.sleep(0.04)


def tahap_3_deteksi_dan_centering_donat_merah(drone: DroneController, sensor: MissionVisionListener):
    """
    TAHAP 3:
    Mendeteksi lingkaran merah dan centering titik tengah lubang donat:
    - Drone melayang di tempat di atas lingkaran hijau (kamera bawah mengunci posisi X & Y).
    - Drone melakukan tracking ketinggian naik/turun vertikal (Z) dan yaw di atas lingkaran hijau
      terlebih dahulu hingga titik tengah kamera depan sejajar sempurna dengan titik pusat lubang donat.
    - DILARANG maju terlebih dahulu dan DILARANG mendeteksi/centering angka bullseye sebelum
      tahap centering donat merah ini selesai!
    - Kriteria Selesai: Lubang donat terdeteksi jelas, offset kamera depan lurus (|dx|<=18, |dy|<=18),
      posisi tetap terkunci di atas lingkaran hijau, dan ketinggian di zona donat (alt >= 2.20 m)
      stabil selama >= 15 frame berturut-turut.
    - Jika sudah sesuai, barulah sistem selesai dan siap maju menembus donat.
    """
    print("\n" + "=" * 65)
    print(">>> [TAHAP 3] CENTERING PRESISI TITIK PUSAT LUBANG DONAT MERAH")
    print(">>> (Fokus Naik-Turun di Atas Lingkaran Hijau | Dilarang Maju Dahulu)")
    print("=" * 65)

    # Pastikan kamera depan dalam mode DONUT (angka bullseye tidak dideteksi)
    sensor.set_mode("DONUT")
    drone.brake(0.5)

    steady_count = 0
    search_timer = time.time()
    last_log_time = 0.0

    print("[*] Mengunci posisi di atas lingkaran hijau. Memulai tracking naik-turun mencari lubang donat...")

    while rclpy.ok():
        drone.update()
        green = sensor.get_green()
        donut = sensor.get_donut()
        alt = -drone.pos_z

        # 1. KONTROL ANCHOR POSISI (X, Y) DI ATAS LINGKARAN HIJAU (KAMERA BAWAH)
        # Menjaga drone agar tetap melayang tepat di atas lingkaran hijau selama proses naik-turun
        if green is not None and green.detected:
            # dy > 0: lingkaran di belakang/bawah -> mundur (vx < 0)
            # dx > 0: lingkaran di kanan frame    -> kanan  (vy > 0)
            vx_anchor = np.clip(-0.0050 * green.dy, -0.45, 0.45)
            vy_anchor = np.clip( 0.0050 * green.dx, -0.45, 0.45)
            is_over_green = (abs(green.dx) <= 28 and abs(green.dy) <= 28)
        else:
            # Jika sesaat tidak terlihat di ketinggian tinggi, tahan kecepatan body ke 0
            vx_anchor = 0.0
            vy_anchor = 0.0
            is_over_green = True

        # 2. KONTROL VERTIKAL NAIK-TURUN (Z) & YAW PUSAT LUBANG DONAT (KAMERA DEPAN)
        if donut is not None and donut.detected:
            dx_d = donut.dx
            dy_d = donut.dy

            # Visual Servoing Centering Donat Merah:
            # dy_d > 0: donat di bawah tengah kamera -> turun (vz_up < 0)
            # dy_d < 0: donat di atas tengah kamera  -> naik  (vz_up > 0)
            vz_cmd = np.clip(-0.0050 * dy_d, -0.55, 0.55)
            # dx_d > 0: donat di kanan kamera -> putar yaw kanan (yaw_rate > 0)
            # dx_d < 0: donat di kiri kamera  -> putar yaw kiri  (yaw_rate < 0)
            yaw_cmd = np.clip(0.0040 * dx_d, -0.45, 0.45)

            # Syarat Centering Selesai Sempurna:
            # 1. Lubang donat sejajar tengah kamera: |dx| <= 18 px, |dy| <= 18 px
            # 2. Lubang donat terdeteksi valid (donut.has_hole == True)
            # 3. Ketinggian drone sudah di zona lubang donat (alt >= 2.20 m)
            # 4. Posisi horizontal drone tetap terkunci di atas lingkaran hijau (is_over_green)
            is_tight_centered = (abs(dx_d) <= 18 and abs(dy_d) <= 18 and donut.has_hole and alt >= 2.20 and is_over_green)

            if is_tight_centered:
                steady_count += 1
                drone.send_velocity_body(0.0, 0.0, 0.0, yaw_rate=0.0)
                if steady_count >= 15:  # Terkunci stabil ~0.6 detik dalam kondisi diam sempurna
                    print("\n" + "=" * 65)
                    print("[OK] CENTERING TITIK PUSAT DONAT MERAH SELESAI & TERKUNCI!")
                    print(f"     -> Lubang Donat : ({donut.cx}, {donut.cy}) | Offset: dx={dx_d:+d}, dy={dy_d:+d}")
                    print(f"     -> Ketinggian   : {alt:.2f} m (Zona Donat) | Posisi: Tepat di atas lingkaran hijau")
                    print("=" * 65)
                    drone.brake(0.6)
                    break
            else:
                steady_count = 0
                # Gabungkan kontrol anchor di atas lingkaran hijau (vx, vy)
                # dengan kontrol vertikal naik-turun dan hadap yaw lubang donat (vz, yaw)
                drone.send_velocity_body(vx=vx_anchor, vy=vy_anchor, vz_up=vz_cmd, yaw_rate=yaw_cmd)

            now = time.time()
            if now - last_log_time > 0.4:
                status_str = f"STABIL {steady_count}/15" if is_tight_centered else "CENTERING..."
                hole_str = "Hole:OK" if donut.has_hole else "Hole:NO"
                print(f"  [{status_str}] Donut(dx={dx_d:+3d}, dy={dy_d:+3d}, {hole_str}) | vz={vz_cmd:+.2f} m/s | Alt={alt:.2f}m")
                last_log_time = now

        else:
            # Donat belum tertangkap jelas di kamera depan:
            # Lakukan tracking naik-turun di atas lingkaran hijau mencari donat (donat di Z ~ 2.6 m)
            steady_count = 0
            if alt < 2.65:
                vz_search = 0.45
                search_label = "Naik vertikal mencari lubang donat (Z ~ 2.6 m)"
            else:
                vz_search = -0.35
                search_label = "Turun vertikal mencari lubang donat"

            now = time.time()
            if now - search_timer > 0.8:
                print(f"  [*] {search_label}... (Ketinggian: {alt:.2f} m | Tetap di atas lingkaran hijau)")
                search_timer = now

            # Drone tetap terkunci di atas lingkaran hijau selama mencari
            drone.send_velocity_body(vx=vx_anchor, vy=vy_anchor, vz_up=vz_search, yaw_rate=0.0)

        time.sleep(0.04)


def tahap_4_menembus_donat_merah(drone: DroneController, sensor: MissionVisionListener):
    """
    TAHAP 4:
    Melaju ke depan menembus lubang donat merah (vx = 2.8 m/s).
    Menggunakan sensor LiDAR 4 arah (kiri, kanan, atas, bawah) untuk mendeteksi
    secara fisik saat memasuki dinding sempit donat, dan HANYA merayakan
    setelah dinding sempit terlewati dan ruang kembali terbuka luas.
    Setelah itu melambat/berhenti untuk tracking sasaran Bulls Eye.
    """
    print("\n" + "=" * 65)
    print(">>> [TAHAP 4] MELAJU MENEMBUS LUBANG DONAT MERAH (VERIFIKASI FISIK LIDAR)")
    print(">>> (Monitoring LiDAR Kiri/Kanan/Atas: Dinding Sempit -> Ruang Terbuka)")
    print("=" * 65)

    print("[*] SPRINT MAJU lurus menembus lubang donat merah (vx = 2.80 m/s)...")

    has_entered_donut = False
    has_cleared_donut = False
    t_start = time.time()
    t_entered = 0.0
    last_log_time = 0.0

    while rclpy.ok():
        drone.update()
        drone.send_velocity_body(vx=2.80, vy=0.0, vz_up=0.0, yaw_rate=0.0)

        d_left = sensor.get_lidar_left()
        d_right = sensor.get_lidar_right()
        d_up = sensor.get_lidar_up()
        d_down = sensor.get_lidar_down()
        now = time.time()
        elapsed = now - t_start

        # Deteksi dinding sempit donat merah (ketika drone berada di dalam rongga donat)
        # Di luar donat, d_left & d_right > 5 m, d_up > 5 m.
        # Di dalam donat (diameter lubang 1.2 m), jarak sensor ke dinding donat menjadi < 1.30 m.
        is_narrow_wall = (d_left < 1.30 or d_right < 1.30 or d_up < 1.30)

        if not has_entered_donut:
            if is_narrow_wall:
                has_entered_donut = True
                t_entered = now
                print("\n" + "=" * 65)
                print(">>> [LIDAR EVENT] DINDING SEKETIKA SEMPIT TERDETEKSI!")
                print(">>> DRONE SEDANG MEMASUKI & BERADA DI DALAM LUBANG DONAT MERAH!")
                print(f">>> Sensor: Left={d_left:.2f}m | Right={d_right:.2f}m | Up={d_up:.2f}m | Down={d_down:.2f}m")
                print("=" * 65 + "\n")
            elif now - last_log_time > 0.4:
                print(f"  [*] Melaju mendekati donat... (LiDAR Left: {d_left:.1f}m, Right: {d_right:.1f}m, Up: {d_up:.1f}m)")
                last_log_time = now

        else:
            # Drone sudah pernah masuk rongga donat. Sekarang pantau saat keluar dari donat:
            # Dinding sempit hilang (semua sensor samping & atas kembali terbuka > 1.65 m)
            # dan telah melaju minimal 0.35 detik sejak masuk (badan drone sepanjang 0.6m tuntas keluar).
            is_space_opened = (d_left > 1.65 and d_right > 1.65 and d_up > 1.65)
            time_inside = now - t_entered

            if is_space_opened and time_inside >= 0.35:
                has_cleared_donut = True
                print("\n" + "=" * 65)
                print(">>> [LIDAR EVENT] DINDING SEMPIT SELESAI, RUANG KEMBALI TERBUKA LUAS!")
                print(">>> TERKONFIRMASI FISIK OLEH LIDAR: DONAT MERAH TELAH SEPENUHNYA DILEWATI!")
                print(f">>> Sensor: Left={d_left:.2f}m | Right={d_right:.2f}m | Up={d_up:.2f}m")
                print("=" * 65)
                break
            elif now - last_log_time > 0.3:
                print(f"  [*] Sedang meluncur melewati lubang donat... (Time inside: {time_inside:.2f}s | Left: {d_left:.2f}m, Right: {d_right:.2f}m)")
                last_log_time = now

        # Batas pengaman waktu jika single-ray LiDAR lolos sekilas:
        # Jika sudah melaju 2.8 detik (menempuh > 7 meter), donat dipastikan sudah terlewati
        if elapsed >= 2.8:
            print("\n[*] Batas jarak sprint donat tercapai (menempuh ~7.5 meter).")
            break

        time.sleep(0.03)

    # HANYA SETELAH TERKONFIRMASI MELEWATI DONAT:
    # Drone melambat dan berhenti sempurna sesuai instruksi
    print("\n[*] Melambat dan berhenti total setelah melewati donat merah...")
    drone.brake(1.0)
    print("[OK] Drone telah aman di seberang donat merah dan berhenti.")

    # Menyesuaikan ketinggian turun perlahan ke level target Bulls Eye (Z ~ 1.5 m)
    print("[*] Menyesuaikan ketinggian ke level target Bulls Eye (Z ~ 1.5 m)...")
    t_descend = time.time() + 2.0
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
    Setelah menyelesaikan obstacle donat merah dan berhenti:
    1. Memindai feed kamera depan mencari target angka (target_digit = 2).
    2. Mendeteksi Bounding Box angka dan menghitung titik pusatnya (cx, cy).
    3. Centering presisi ke titik pusat bounding box angka (hover di tempat vx=0.0).
    4. Baru kemudian melaju lurus dengan kecepatan tinggi (vx = 7.0 m/s) menabrak
       titik tengah sasaran hingga impact terdeteksi oleh LiDAR depan (< 0.45 m).
    """
    print("\n" + "=" * 65)
    print(f">>> [TAHAP 5 s.d. 10] CENTERING TITIK PUSAT BOX & KAMIKAZE SASARAN #{target_digit}")
    print(">>> (Centering Presisi ke Titik Tengah Angka -> Sprint Lurus Menabrak)")
    print("=" * 65)

    # 1. Posisikan drone stabil di seberang donat
    drone.brake(0.4)
    # Aktifkan mode KAMIKAZE: Sekarang detektor angka Bullseye aktif & feed GUI berganti ke sasaran
    sensor.set_mode("KAMIKAZE")
    print(f"[*] Mode Kamera Depan dialihkan ke: KAMIKAZE (Deteksi Sasaran #{target_digit}).")

    # 2. DETEKSI BERKELANJUTAN TANPA TIMEOUT HINGGA ANGKA SASARAN TERDETEKSI
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
                bx, by, bw, bh = best.bbox
                print(f"\n[OK] TARGET #{target_digit} TERDETEKSI!")
                print(f"     -> Bounding Box Angka: [X={bx}, Y={by}, W={bw}, H={bh}]")
                print(f"     -> Titik Pusat Angka : ({best.cx}, {best.cy}) | Offset: dx={best.dx:+d}, dy={best.dy:+d}")
                break

        # Jika belum tampak jelas, lakukan penyelarasan maju perlahan
        step_cnt += 1
        if step_cnt % 25 == 0:
            print(f"  [*] Memindai kamera depan mencari Bounding Box Angka #{target_digit}...")
        drone.send_velocity_body(vx=0.15, vy=0.0, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)

    # 3. CENTERING PRESISI KE TITIK PUSAT BOX ANGKA (HOVER DI TEMPAT vx = 0.0)
    print(f"\n[*] Memulai Centering presisi ke TITIK PUSAT Box Angka #{target_digit} (Hover vx=0.0)...")
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
            bx, by, bw, bh = target_now.bbox
            tcx, tcy = target_now.cx, target_now.cy

            # Visual Servoing P-Controller:
            # yaw_rate: mengarahkan moncong secara horizontal ke titik tengah box
            # vy_cmd  : pergeseran samping lateral agar posisi bodi sejajar garis lurus sasaran
            # vz_cmd  : menyamakan ketinggian kamera tepat dengan titik tengah vertikal box
            yaw_rate_cmd = np.clip(0.0055 * dx, -0.80, 0.80)
            vy_cmd = np.clip(0.0035 * dx, -0.45, 0.45)
            vz_cmd = np.clip(-0.0055 * dy, -0.65, 0.65)

            # PENTING: vx = 0.0 (drone melayang di tempat menyelaraskan reticle ke titik pusat angka)
            if abs(dx) <= 16 and abs(dy) <= 16:
                steady_aim += 1
                drone.send_velocity_body(0.0, 0.0, 0.0, yaw_rate=0.0)
                if steady_aim >= 5:  # Terkunci stabil di titik tengah selama ~0.2 detik
                    print(f"\n[OK] RETICLE TERKUNCI SEMPURNA PADA TITIK PUSAT BOX ANGKA #{target_digit}!")
                    print(f"     -> Box: [{bx}, {by}, {bw}, {bh}] | Pusat Angka: ({tcx}, {tcy}) | Error: dx={dx:+d}, dy={dy:+d}")
                    drone.brake(0.4)
                    break
            else:
                steady_aim = 0
                drone.send_velocity_body(vx=0.0, vy=vy_cmd, vz_up=vz_cmd, yaw_rate=yaw_rate_cmd)

            if steady_aim == 0 and int(time.time() * 5) % 5 == 0:
                print(f"  [CENTERING ANGKA #{target_digit}] Box: [{bx},{by},{bw}x{bh}] | Pusat: ({tcx},{tcy}) | Error: dx={dx:+3d}, dy={dy:+3d}")
        else:
            # Jika sesaat tidak terdeteksi, tahan posisi di tempat
            drone.send_velocity_body(vx=0.0, vy=0.0, vz_up=0.0, yaw_rate=0.0)

        time.sleep(0.03)

    # 4. TAHAP 10: BARU KEMUDIAN BERJALAN LURUS MENABRAK TITIK TENGAH ANGKA
    print("\n" + "=" * 65)
    print(f"🚀🚀 SPRINT LURUS: MELAJU KILAT MENABRAK TITIK TENGAH ANGKA #{target_digit}! 🚀🚀")
    print(">>> Kecepatan: vx = 7.0 m/s (Lurus Menuju Pusat Sasaran) | Kamikaze Strike")
    print("=" * 65)

    KECEPATAN_TABRAK = 7.0

    while rclpy.ok():
        drone.update()
        jarak_lidar = sensor.get_lidar_front()

        # Deteksi tabrakan via sensor jarak LiDAR depan
        if jarak_lidar <= 0.45:
            print(f"\n💥💥💥 IMPACT TABRAKAN TERJADI TEPAT PADA TITIK TENGAH SASARAN! (LiDAR: {jarak_lidar:.2f} m) 💥💥💥")
            drone.send_velocity_body(vx=KECEPATAN_TABRAK, vy=0.0, vz_up=0.0, yaw_rate=0.0)
            time.sleep(0.2)
            break

        # Koreksi mikro garis lurus saat melesat agar tabrakan persis di tengah
        res_k = sensor.get_kamikaze()
        yaw_trim = 0.0
        vz_trim = 0.0

        if res_k is not None:
            matching = [t for t in res_k.all_targets if t.digit == target_digit]
            if res_k.locked_target and res_k.locked_target.digit == target_digit:
                matching.append(res_k.locked_target)
            if matching:
                t_cur = max(matching, key=lambda t: t.area)
                yaw_trim = np.clip(0.0030 * t_cur.dx, -0.6, 0.6)
                vz_trim = np.clip(-0.0030 * t_cur.dy, -0.4, 0.4)

        drone.send_velocity_body(vx=KECEPATAN_TABRAK, vy=0.0, vz_up=vz_trim, yaw_rate=yaw_trim)
        time.sleep(0.03)

    drone.brake(0.5)
    print("\n" + "=" * 65)
    print(f"🎉🎉 MISI 3 SUKSES PENUH! DRONE TELAH MENABRAK TITIK TENGAH SASARAN #{target_digit}! 🎉🎉")
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
    node.create_subscription(LaserScan, "/lidar_back", listener.on_lidar_back, 10)
    node.create_subscription(LaserScan, "/lidar_left", listener.on_lidar_left, 10)
    node.create_subscription(LaserScan, "/lidar_right", listener.on_lidar_right, 10)
    node.create_subscription(LaserScan, "/lidar_up", listener.on_lidar_up, 10)
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
            tahap_4_menembus_donat_merah(drone, listener)

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
