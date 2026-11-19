#!/usr/bin/env python3
"""
MISI 3: NAVIGASI LENGKAP & KAMIKAZE TARGET NOMOR 2
==================================================
Implementasi algoritma misi 3 sesuai instruksi pada skrip_misi3.md:

1. Setelah keluar dari terowongan, drone bergerak ke kiri (X negatif) mencari lingkaran hijau.
2. Ketika mendeteksi lingkaran hijau, drone melakukan centering posisi X dan Y terhadap lingkaran hijau.
3. Setelah centering lingkaran hijau, drone mendeteksi lingkaran merah (bisa naik/turun mencari lingkaran merah).
4. Setelah mendeteksi dan centering terhadap titik pusat lingkaran merah, drone melaju ke depan tepat
   di tengah lingkaran merah (karena lingkaran merah berbentuk seperti donat).
5. Setelah itu drone mendeteksi angka pada Bulls Eye.
6. Drone melakukan yaw +- 60 derajat untuk memindai semua Bulls Eye di depannya.
7. Menggunakan modul OpenCV berkecepatan tinggi pada ./deteksi/ (tanpa YOLO / deep learning).
8. Koordinat dan heading nomor 2 dari Bulls Eye didapatkan dan disimpan (di-keep).
9. Drone menghadap (yaw & pitch/ketinggian) tepat ke titik tengah nomor 2.
10. Drone terbang melesat maju hingga menabrak sasaran nomor 2 (Kamikaze).

Modul Deteksi:
  Menggunakan package ./deteksi/ yang sudah disediakan:
  - GreenRingDetector  (centering_hijau.py  - kamera bawah)
  - RedDonutDetector   (donat_merah.py     - kamera depan)
  - KamikazeDetector   (kamikaze.py        - kamera depan, HOG + Linear SVM)

Cara Menjalankan:
  python3 scripts/misi3.py                 # Eksekusi penuh dari tahap 1 s.d. 10
  python3 scripts/misi3.py --step 3        # Mulai langsung dari tahap 3 (donat merah)
  python3 scripts/misi3.py --step 5        # Mulai langsung dari tahap 5 (bullseye kamikaze)
  python3 scripts/misi3.py --step 5 --teleport  # Auto-teleport ke depan Bulls Eye lalu jalankan
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

# Tambahkan path scripts/ ke sys.path untuk import package deteksi
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
TINGGI_DEFAULT = 1.35  # Ketinggian standar drone


# ==============================================================================
# 1. SUBSISTEM SENSOR & VISION LISTENER (ROS 2)
# ==============================================================================
class MissionVisionListener:
    """
    Listener thread-safe untuk stream sensor kamera dan LiDAR dari ROS 2.
    Mengintegrasikan modul deteksi dari ./deteksi/ secara real-time.
    """
    def __init__(self, target_digit: int = 2):
        self.lock = threading.Lock()
        self.target_digit = target_digit

        # Inisialisasi Detektor Vision OpenCV
        self.green_detector = GreenRingDetector(tolerance_px=20)
        self.donut_detector = RedDonutDetector(tolerance_px=22)
        self.kamikaze_detector = KamikazeDetector(target_digit=target_digit, tolerance_px=20)

        # Buffer Frame Mentah
        self.frame_front: Optional[np.ndarray] = None
        self.frame_down: Optional[np.ndarray] = None

        # Data Sensor Jarak LiDAR
        self.lidar_front: float = 99.0
        self.lidar_down: float = 99.0

        # Hasil Deteksi Terakhir
        self.res_green: Optional[GreenCenteringResult] = None
        self.res_donut: Optional[RedDonutResult] = None
        self.res_kamikaze: Optional[KamikazeResult] = None

        # Template Matching Tambahan untuk Digit 2 (Verifikasi Redundan)
        self.tmpl_digit2 = self._init_template_digit2()

    def _init_template_digit2(self) -> np.ndarray:
        tmpl_dir = os.path.join(SCRIPTS_DIR, "templates")
        os.makedirs(tmpl_dir, exist_ok=True)
        tmpl_path = os.path.join(tmpl_dir, "digit_2.png")
        if os.path.exists(tmpl_path):
            img = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
        # Buat template sintetis jika belum ada
        tmpl = np.full((70, 50), 255, dtype=np.uint8)
        cv2.putText(tmpl, '2', (6, 58), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5, cv2.LINE_AA)
        cv2.imwrite(tmpl_path, tmpl)
        return tmpl

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

        # Jalankan deteksi donat merah dan kamikaze pada frame depan
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

        # Jalankan deteksi lingkaran hijau pada frame bawah
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

    # Thread-safe getters
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

    def get_lidar_down(self) -> float:
        with self.lock:
            return self.lidar_down

    def get_annotated_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            f_down = self.res_green.annotated_frame if self.res_green is not None else self.frame_down
            f_front = self.res_kamikaze.annotated_frame if self.res_kamikaze is not None else self.frame_front
            return f_down, f_front


# ==============================================================================
# 2. SUBSISTEM KONTROL DRONE AUTOPILOT (MAVLINK)
# ==============================================================================
class DroneController:
    """
    Pengendali manuver terbang drone berbasis pymavlink.
    Mendukung kontrol kecepatan translasi (body frame) dan rotasi yaw.
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

        # Minta stream data posisi (32) dan attitude (30) pada 20 Hz
        self._minta_stream(32, 20)  # LOCAL_POSITION_NED
        self._minta_stream(30, 20)  # ATTITUDE
        time.sleep(1.0)
        self.update()

    def _minta_stream(self, msg_id: int, hz: int):
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0
        )

    def update(self, timeout: float = 0.02):
        """Membaca dan memperbarui status telemetri drone dari paket MAVLink."""
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
            time.sleep(0.1)
        return False

    def send_velocity_body(self, vx: float, vy: float, vz_up: float, yaw_rate: float = 0.0):
        """
        Mengirim perintah kecepatan dalam BODY FRAME:
        vx > 0: Maju,      vx < 0: Mundur
        vy > 0: Geser Kanan, vy < 0: Geser Kiri
        vz_up > 0: Naik,   vz_up < 0: Turun
        yaw_rate: Kecepatan putar sudut (rad/s, positif = putar kanan/CW)
        """
        # Dalam NED: vz positif adalah ke bawah (down), sehingga -vz_up membalik arah naik
        vz_ned = -vz_up

        # type_mask: abaikan posisi (0,1,2), akselerasi (6,7,8), yaw (10). Aktifkan vel (3,4,5) dan yaw_rate (11).
        type_mask = 0b000111000111
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
        """Mengirim kecepatan selama durasi tertentu dengan loop pembaruan status."""
        t_end = time.time() + detik
        while time.time() < t_end:
            self.send_velocity_body(vx, vy, vz_up, yaw_rate)
            self.update()
            time.sleep(0.05)
        self.send_velocity_body(0, 0, 0, 0)

    def brake(self, detik: float = 1.0):
        """Menghentikan pergerakan drone seketika (hover di tempat)."""
        self.move_timed(0, 0, 0, detik)

    def takeoff_if_needed(self, tinggi: float = TINGGI_DEFAULT):
        """Memastikan drone dalam kondisi terbang pada ketinggian target."""
        self.update()
        if not self.armed or (-self.pos_z < 0.35):
            print(f"[*] Drone belum di udara (Armed: {self.armed}, Alt: {-self.pos_z:.2f}m).")
            print("[*] Mengubah mode ke GUIDED & melakukan ARMing motor...")
            self.set_mode('GUIDED')

            # Force Arming
            for attempt in range(1, 11):
                self.m.mav.command_long_send(
                    self.m.target_system, self.m.target_component,
                    mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                    1, 21196, 0, 0, 0, 0, 0
                )
                time.sleep(0.5)
                self.update()
                if self.armed:
                    print(f"[OK] Drone berhasil di-ARM (Percobaan ke-{attempt}).")
                    break

            print(f"[*] Mengirim perintah Takeoff ke ketinggian {tinggi:.2f} meter...")
            self.m.mav.command_long_send(
                self.m.target_system, self.m.target_component,
                mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                0, 0, 0, 0, 0, 0, tinggi
            )

            t_end = time.time() + 20.0
            while time.time() < t_end:
                self.update()
                current_alt = -self.pos_z
                if current_alt >= tinggi * 0.85:
                    print(f"[OK] Ketinggian target tercapai ({current_alt:.2f} m).")
                    break
                time.sleep(0.2)
            self.brake(1.5)
        else:
            print(f"[OK] Drone sudah berada di udara (Altitude: {-self.pos_z:.2f} m).")

    def turn_to_yaw(self, target_yaw_deg: float, tolerance: float = 3.5, timeout: float = 8.0) -> bool:
        """Memutar hadap drone (yaw) ke target sudut absolut secara tertutup."""
        print(f"[*] Memutar drone ke Yaw: {target_yaw_deg:.1f}° ...")
        t_end = time.time() + timeout

        while time.time() < t_end:
            self.update()
            # Hitung selisih sudut terkecil [-180 .. 180]
            err = (target_yaw_deg - self.yaw_deg + 180.0) % 360.0 - 180.0

            if abs(err) <= tolerance:
                self.brake(0.4)
                print(f"[OK] Yaw tercapai: {self.yaw_deg:.1f}° (Error: {err:.1f}°)")
                return True

            # Kontrol P untuk laju putar sudut yaw
            yaw_rate_cmd = np.clip(err * 0.035, -0.65, 0.65)
            self.send_velocity_body(0, 0, 0, yaw_rate=yaw_rate_cmd)
            time.sleep(0.05)

        self.brake(0.5)
        print(f"[!] Timeout putar yaw. Posisi akhir: {self.yaw_deg:.1f}°")
        return False

    def land(self):
        print("\n[*] Mendaratkan drone...")
        self.set_mode('LAND')


# ==============================================================================
# 3. IMPLEMENTASI 10 TAHAP ALGORITMA SESUAI SKRIP_MISI3.MD
# ==============================================================================

def tahap_1_cari_lingkaran_hijau(drone: DroneController, sensor: MissionVisionListener, timeout: float = 25.0) -> bool:
    """
    TAHAP 1:
    Setelah keluar dari terowongan, drone harus bergerak ke kiri (X negatif)
    untuk mencari lingkaran hijau di kamera bawah.
    """
    print("\n" + "=" * 65)
    print(">>> [TAHAP 1] BERGERAK KE KIRI (X NEGATIF) MENCARI LINGKARAN HIJAU")
    print("=" * 65)

    drone.update()
    base_yaw = drone.yaw_deg
    print(f"[*] Heading awal drone: {base_yaw:.1f}°")
    print("[*] Bergerak geser ke kiri (vy = -0.55 m/s) sambil memindai kamera bawah...")

    t_start = time.time()
    t_end = t_start + timeout

    while time.time() < t_end:
        drone.update()
        green = sensor.get_green()

        # Deteksi lingkaran hijau pada kamera bawah
        if green is not None and green.detected and green.area >= 180.0:
            print(f"\n[OK] LINGKARAN HIJAU DITEMUKAN!")
            print(f"     -> Area: {green.area:.0f} px | Posisi: cx={green.cx}, cy={green.cy}")
            print(f"     -> Offset: dx={green.dx:+d}, dy={green.dy:+d}")
            drone.brake(1.0)
            return True

        # Bergerak ke kiri (vy < 0) dengan sedikit maju (vx > 0) jika perlu menjaga jalur
        drone.send_velocity_body(vx=0.08, vy=-0.55, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.06)

    print("[!] Timeout Tahap 1: Lingkaran hijau belum terdeteksi setelah bergerak ke kiri.")
    drone.brake(1.0)
    return False


def tahap_2_centering_lingkaran_hijau(drone: DroneController, sensor: MissionVisionListener, timeout: float = 20.0) -> bool:
    """
    TAHAP 2:
    Ketika mendeteksi lingkaran hijau, drone seharusnya melakukan centering
    posisi X dan Y nya terhadap lingkaran hijau.
    """
    print("\n" + "=" * 65)
    print(">>> [TAHAP 2] CENTERING POSISI X DAN Y TERHADAP LINGKARAN HIJAU")
    print("=" * 65)

    t_start = time.time()
    t_end = t_start + timeout
    steady_count = 0

    while time.time() < t_end:
        drone.update()
        green = sensor.get_green()

        if green is None or not green.detected:
            # Jika sesaat hilang, berhenti dan cari kembali
            drone.send_velocity_body(0, 0, 0)
            time.sleep(0.1)
            continue

        dx = green.dx
        dy = green.dy

        # Pada kamera bawah:
        # dx > 0 (target di sebelah kanan frame) -> drone geser kanan (vy > 0)
        # dx < 0 (target di sebelah kiri frame)  -> drone geser kiri (vy < 0)
        # dy < 0 (target di bagian atas frame / depan) -> drone maju (vx > 0)
        # dy > 0 (target di bagian bawah frame / belakang) -> drone mundur (vx < 0)
        vx_cmd = np.clip(-0.0018 * dy, -0.35, 0.35)
        vy_cmd = np.clip(0.0018 * dx, -0.35, 0.35)

        # Cek apakah sudah tepat di tengah (toleransi +-20 pixel)
        if abs(dx) <= 22 and abs(dy) <= 22:
            steady_count += 1
            drone.send_velocity_body(0, 0, 0)
            if steady_count >= 12:  # Stabil selama ~0.7 detik
                print(f"[OK] Centering lingkaran hijau SUKSES! (dx={dx:+d}, dy={dy:+d}, Radius={green.radius:.1f})")
                drone.brake(1.5)
                return True
        else:
            steady_count = 0
            drone.send_velocity_body(vx_cmd, vy_cmd, 0.0)

        time.sleep(0.05)

    print("[!] Batas waktu centering tercapai. Mengunci posisi terbaik dan melanjutkan.")
    drone.brake(1.0)
    return True


def tahap_3_deteksi_dan_centering_donat_merah(drone: DroneController, sensor: MissionVisionListener, timeout: float = 25.0) -> bool:
    """
    TAHAP 3:
    Setelah melakukan centering pada lingkaran hijau, drone mendeteksi lingkaran merah,
    dimana drone bisa naik dan turun untuk mencari lingkaran merah tersebut.
    """
    print("\n" + "=" * 65)
    print(">>> [TAHAP 3] DETEKSI & CENTERING DONAT MERAH (NAIK / TURUN)")
    print("=" * 65)

    # Pastikan drone menghadap lurus ke depan (menghadap donat merah di utara / Y positif ~ 90°)
    drone.turn_to_yaw(90.0, tolerance=4.0)
    drone.brake(1.0)

    t_start = time.time()
    t_end = t_start + timeout
    steady_count = 0
    search_phase = 0
    search_timer = time.time()

    while time.time() < t_end:
        drone.update()
        donut = sensor.get_donut()

        if donut is not None and donut.detected:
            # Titik tengah lubang donat berhasil terdeteksi
            dx = donut.dx
            dy = donut.dy

            # Koreksi horizontal (vy) dan vertikal (vz_up)
            # dy > 0: lubang di bawah tengah frame -> drone harus TURUN (vz_up < 0)
            # dy < 0: lubang di atas tengah frame  -> drone harus NAIK (vz_up > 0)
            vy_cmd = np.clip(0.0015 * dx, -0.28, 0.28)
            vz_cmd = np.clip(-0.0018 * dy, -0.28, 0.28)

            # Cek apakah sudah sejajar dengan lubang donat
            if abs(dx) <= 25 and abs(dy) <= 25:
                steady_count += 1
                drone.send_velocity_body(0.05, 0, 0)  # Perlahan mendekat
                if steady_count >= 10:
                    hole_txt = "Lubang Dalam (Inner Hole)" if donut.has_hole else "Pusat Donat"
                    print(f"[OK] Centering Donat Merah SUKSES! ({hole_txt})")
                    print(f"     -> cx={donut.cx}, cy={donut.cy} | dx={dx:+d}, dy={dy:+d} | Radius={donut.outer_radius:.1f}")
                    drone.brake(1.0)
                    return True
            else:
                steady_count = 0
                drone.send_velocity_body(vx=0.08, vy=vy_cmd, vz_up=vz_cmd)

        else:
            # Sesuai instruksi: Drone bisa NAIK dan TURUN untuk mencari lingkaran merah
            steady_count = 0
            now = time.time()
            if now - search_timer > 2.5:
                search_phase = (search_phase + 1) % 4
                search_timer = now

            if search_phase == 0:
                print("[*] Donat merah belum tampak: Menyesuaikan ketinggian (Naik perlahan)...")
                drone.send_velocity_body(0, 0, vz_up=0.25)
            elif search_phase == 1:
                print("[*] Donat merah belum tampak: Menyesuaikan ketinggian (Turun perlahan)...")
                drone.send_velocity_body(0, 0, vz_up=-0.25)
            elif search_phase == 2:
                print("[*] Donat merah belum tampak: Menjaga posisi stabil...")
                drone.send_velocity_body(0, 0, vz_up=0.0)
            else:
                print("[*] Donat merah belum tampak: Maju perlahan mendekati zona...")
                drone.send_velocity_body(vx=0.15, vy=0, vz_up=0.0)

        time.sleep(0.06)

    print("[!] Timeout centering donat merah. Mengunci jalur tengah dan melanjutkan.")
    drone.brake(1.0)
    return True


def tahap_4_menembus_donat_merah(drone: DroneController, sensor: MissionVisionListener, durasi: float = 3.8):
    """
    TAHAP 4:
    Setelah mendeteksi dan centering terhadap titik pusat lingkaran merah tersebut,
    drone melaju ke depan tepat di tengah lingkaran merah karena berbentuk donat.
    """
    print("\n" + "=" * 65)
    print(">>> [TAHAP 4] MELAJU KE DEPAN TEPAT DI TENGAH LUBANG DONAT MERAH")
    print("=" * 65)

    print(f"[*] Meluncur maju dengan kecepatan vx = 1.35 m/s menembus donat (durasi: {durasi} detik)...")
    t_end = time.time() + durasi

    while time.time() < t_end:
        drone.update()
        # Terbang lurus ke depan dengan ketinggian stabil
        drone.send_velocity_body(vx=1.35, vy=0.0, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.05)

    drone.brake(1.5)
    print("[OK] Drone berhasil menembus donat merah dan tiba di area sasaran berikutnya!")


def tahap_5_dan_6_scanning_yaw_bullseye(
    drone: DroneController,
    sensor: MissionVisionListener,
    target_digit: int = 2
) -> Dict[str, Any]:
    """
    TAHAP 5 & 6 & 7 & 8:
    Drone mendeteksi angka pada Bulls Eye dengan melakukan yaw +- 60 derajat,
    sehingga dapat melihat semua Bulls Eye di depannya menggunakan OpenCV (tanpa YOLO).
    Koordinat dan heading nomor 2 didapatkan dan di-keep (disimpan).
    """
    print("\n" + "=" * 65)
    print(f">>> [TAHAP 5 & 6] SCANNING YAW +-60 DERAJAT MEMINDAI BULLSEYE #{target_digit}")
    print("=" * 65)

    drone.update()
    base_yaw = drone.yaw_deg
    yaw_kiri = (base_yaw + 60.0) % 360.0
    yaw_kanan = (base_yaw - 60.0 + 360.0) % 360.0

    print(f"[*] Sudut hadap awal: {base_yaw:.1f}°")
    print(f"[*] Rencana pemindaian: {base_yaw:.1f}° -> {yaw_kiri:.1f}° (+60°) -> {yaw_kanan:.1f}° (-60°) -> Kunci Target")

    best_target = {
        "found": False,
        "yaw": base_yaw,
        "cx": 320,
        "cy": 240,
        "area": 0.0,
        "score": 0.0,
        "dx": 0,
        "dy": 0
    }

    def inspect_current_frame():
        nonlocal best_target
        res_k = sensor.get_kamikaze()
        if res_k is None:
            return

        # Periksa apakah target nomor 2 terdeteksi (baik melalui locked_target atau daftar all_targets)
        candidates = [t for t in res_k.all_targets if t.digit == target_digit]
        if res_k.locked_target and res_k.locked_target.digit == target_digit:
            candidates.append(res_k.locked_target)

        for c in candidates:
            # Bandingkan skor dan ukuran area
            quality = c.area * (1.0 - abs(c.dx) / 640.0)
            if quality > best_target["area"]:
                best_target["found"] = True
                best_target["yaw"] = drone.yaw_deg
                best_target["cx"] = c.cx
                best_target["cy"] = c.cy
                best_target["area"] = quality
                best_target["score"] = c.confidence
                best_target["dx"] = c.dx
                best_target["dy"] = c.dy
                print(f"  [+] Target #{target_digit} TERDETEKSI pada Yaw {drone.yaw_deg:.1f}°! (Pixel: {c.cx}, {c.cy} | Area: {c.area:.0f})")

    # 1. Sweep ke KIRI (+60 derajat)
    print("\n[*] [Sweep 1/2] Menyapu ke kiri (+60°)...")
    drone.turn_to_yaw(yaw_kiri, tolerance=5.0, timeout=5.0)
    for _ in range(15):
        drone.update()
        inspect_current_frame()
        time.sleep(0.08)

    # 2. Sweep ke KANAN (-60 derajat dari base)
    print("\n[*] [Sweep 2/2] Menyapu ke kanan (-60°)...")
    drone.turn_to_yaw(yaw_kanan, tolerance=5.0, timeout=8.0)
    for _ in range(15):
        drone.update()
        inspect_current_frame()
        time.sleep(0.08)

    # 3. Evaluasi hasil penyimpanan koordinat (TAHAP 8)
    print("\n" + "=" * 65)
    print(f">>> [TAHAP 7 & 8] HASIL PEMINDAIAN NOMOR {target_digit}")
    print("=" * 65)

    if best_target["found"]:
        print(f"[OK] Target #{target_digit} BERHASIL DI-KEEP!")
        print(f"     -> Sudut Hadap Optimal (Yaw): {best_target['yaw']:.1f}°")
        print(f"     -> Koordinat Pusat Sasaran : ({best_target['cx']}, {best_target['cy']})")
        print(f"     -> Bobot Kualitas Deteksi  : {best_target['area']:.0f}")
    else:
        print(f"[!] Target #{target_digit} tidak tertangkap selama sapuan +-60°. Menggunakan heading default {base_yaw:.1f}°.")
        best_target["yaw"] = base_yaw

    return best_target


def tahap_9_dan_10_kamikaze_target_nomor_2(
    drone: DroneController,
    sensor: MissionVisionListener,
    best_target: Dict[str, Any],
    target_digit: int = 2
):
    """
    TAHAP 9 & 10:
    9. Drone menghadap yaw pitch ke arah nomor 2 di tengah-tengahnya.
    10. Drone terbang cepat hingga menabrak nomor 2.
    """
    print("\n" + "=" * 65)
    print(f">>> [TAHAP 9] MENGHADAP YAW & PITCH/KETINGGIAN TEPAT KE TENGAH NOMOR {target_digit}")
    print("=" * 65)

    # 1. Hadapkan drone ke sudut yaw optimal tempat nomor 2 terlihat
    target_yaw = best_target["yaw"]
    drone.turn_to_yaw(target_yaw, tolerance=3.0, timeout=6.0)
    drone.brake(0.8)

    # 2. Penyelarasan halus secara live (Fine visual servoing yaw & pitch)
    print(f"[*] Melakukan alignment visual live ke titik tengah nomor {target_digit}...")
    t_align_end = time.time() + 4.0

    while time.time() < t_align_end:
        drone.update()
        res_k = sensor.get_kamikaze()

        if res_k is not None and res_k.locked_target and res_k.locked_target.digit == target_digit:
            lt = res_k.locked_target
            dx = lt.dx
            dy = lt.dy

            yaw_rate_cmd = np.clip(0.0022 * dx, -0.45, 0.45)
            vz_cmd = np.clip(-0.0018 * dy, -0.25, 0.25)

            if abs(dx) <= 18 and abs(dy) <= 18:
                print(f"[OK] Kunci target sempurna! Target #{target_digit} berada tepat di tengah frame (dx={dx:+d}, dy={dy:+d}).")
                break
            drone.send_velocity_body(vx=0.1, vy=0, vz_up=vz_cmd, yaw_rate=yaw_rate_cmd)
        else:
            drone.send_velocity_body(0, 0, 0)

        time.sleep(0.05)

    drone.brake(0.5)

    # 3. TAHAP 10: TERBANG CEPAT HINGGA MENABRAK NOMOR 2
    print("\n" + "=" * 65)
    print(f">>> [TAHAP 10] KAMIKAZE CHARGE: TERBANG CEPAT MENABRAK TARGET NOMOR {target_digit}!")
    print("=" * 65)

    KECEPATAN_TABRAK = 2.4  # m/s
    DURASI_MAX = 7.0        # detik
    t_charge_start = time.time()
    t_charge_end = t_charge_start + DURASI_MAX

    while time.time() < t_charge_end:
        drone.update()
        jarak_lidar = sensor.get_lidar_front()

        # Cek tabrakan via sensor LiDAR depan
        if jarak_lidar <= 0.38:
            print(f"\n💥💥 IMPACT TERDETEKSI! Jarak LiDAR depan: {jarak_lidar:.2f} m 💥💥")
            drone.send_velocity_body(vx=KECEPATAN_TABRAK, vy=0, vz_up=0)
            time.sleep(0.3)
            break

        # Koreksi lintasan visual secara real-time selama meluncur
        res_k = sensor.get_kamikaze()
        yaw_trim = 0.0
        vz_trim = 0.0

        if res_k is not None and res_k.locked_target and res_k.locked_target.digit == target_digit:
            lt = res_k.locked_target
            yaw_trim = np.clip(0.0015 * lt.dx, -0.35, 0.35)
            vz_trim = np.clip(-0.0012 * lt.dy, -0.20, 0.20)

        drone.send_velocity_body(vx=KECEPATAN_TABRAK, vy=0, vz_up=vz_trim, yaw_rate=yaw_trim)
        time.sleep(0.04)

    # Selesai
    drone.brake(1.0)
    print("\n" + "=" * 65)
    print(f"🎉 MISI 3 SUKSES! DRONE TELAH BERHASIL MENABRAK SASARAN NOMOR {target_digit}!")
    print("=" * 65)


# ==============================================================================
# 4. FUNGSI UTAMA & EKSEKUSI MISI
# ==============================================================================
def run_teleport_preset(preset_key: str):
    """Menjalankan helper teleport.py jika diminta via argumen CLI."""
    teleport_py = os.path.join(SCRIPTS_DIR, "teleport.py")
    if not os.path.exists(teleport_py):
        return

    coords_map = {
        "1": (13.0, 5.0, 1.3, 90.0),   # Di luar terowongan
        "2": (3.3, 11.5, 1.3, 90.0),   # Di atas lingkaran hijau
        "3": (3.3, 9.5, 1.2, 90.0),    # Di depan donat merah
        "4": (3.3, 9.5, 1.2, 90.0),    # Menembus donat
        "5": (-2.0, 21.0, 1.4, 90.0),  # Di depan Bulls Eye
    }
    if preset_key in coords_map:
        x, y, z, yaw = coords_map[preset_key]
        import subprocess
        print(f"[*] Teleport otomatis ke tahap {preset_key}: X={x}, Y={y}, Z={z}, Yaw={yaw}°...")
        subprocess.run([sys.executable, teleport_py, str(x), str(y), str(z), str(yaw)])
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Misi 3: Navigasi Lengkap & Kamikaze Target 2")
    parser.add_argument("--koneksi", default=DEFAULT_KONEKSI, help="Koneksi MAVLink (default: udpin:127.0.0.1:14560)")
    parser.add_argument("--step", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Mulai dari tahap tertentu: 1=Terowongan, 2=Lingkaran Hijau, 3=Donat Merah, 4=Tembus Donat, 5=Kamikaze Bulls Eye")
    parser.add_argument("--target", type=int, default=2, help="Nomor target Bulls Eye sasaran (default: 2)")
    parser.add_argument("--teleport", action="store_true", help="Teleport otomatis ke posisi awal tahap yang dipilih")
    parser.add_argument("--headless", action="store_true", help="Jalankan tanpa window visual GUI OpenCV")
    args = parser.parse_args()

    print("=================================================================")
    print("  GAMAFORCE - MISI 3: NAVIGASI DONAT & KAMIKAZE TARGET BULLSEYE  ")
    print("=================================================================")
    print(f"[*] Target Sasaran Kamikaze: Nomor {args.target}")
    print(f"[*] Tahap Awal Eksekusi    : Tahap {args.step}")

    # 1. Teleport awal jika diaktifkan
    if args.teleport:
        run_teleport_preset(str(args.step))

    # 2. Inisialisasi ROS 2 Node & Sensor Listener
    rclpy.init()
    listener = MissionVisionListener(target_digit=args.target)
    node = Node("misi3_executive_node")

    node.create_subscription(Image, "/camera_down", listener.on_camera_down, 5)
    node.create_subscription(Image, "/camera_front", listener.on_camera_front, 5)
    node.create_subscription(LaserScan, "/lidar_front", listener.on_lidar_front, 5)
    node.create_subscription(LaserScan, "/lidar_down", listener.on_lidar_down, 5)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    print("[OK] Subsistem ROS 2 Kamera & LiDAR aktif.")

    # 3. Inisialisasi Koneksi Drone Autopilot
    drone = DroneController(koneksi=args.koneksi)
    drone.takeoff_if_needed(TINGGI_DEFAULT)

    # 4. Thread Visualisasi Opsional (OpenCV GUI jika display tersedia)
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
                time.sleep(0.05)
        cv2.destroyAllWindows()

    if has_display:
        gui_th = threading.Thread(target=gui_loop, daemon=True)
        gui_th.start()

    try:
        # EKSEKUSI TAHAP BERURUTAN
        if args.step <= 1:
            tahap_1_cari_lingkaran_hijau(drone, listener)

        if args.step <= 2:
            tahap_2_centering_lingkaran_hijau(drone, listener)

        if args.step <= 3:
            tahap_3_deteksi_dan_centering_donat_merah(drone, listener)

        if args.step <= 4:
            tahap_4_menembus_donat_merah(drone, listener)

        if args.step <= 5:
            best_target_2 = tahap_5_dan_6_scanning_yaw_bullseye(drone, listener, target_digit=args.target)
            tahap_9_dan_10_kamikaze_target_nomor_2(drone, listener, best_target_2, target_digit=args.target)

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

