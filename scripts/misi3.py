#!/usr/bin/env python3
"""
MISI 3: PENCARIAN & KAMIKAZE TARGET NOMOR 2 PADA BULLSEYE
Sesuai rancangan algoritma:
1. Drone melakukan scanning yaw +- 60 derajat untuk memindai semua bullseye di depannya.
2. Koordinat / sudut nomor 2 dari bullseye didapatkan dan di-keep (disimpan) hingga scanning selesai.
3. Drone menghadap (yaw & pitch/altitude) tepat ke titik tengah nomor 2.
4. Drone terbang melesat maju hingga menabrak target nomor 2.

Menghubungkan:
- MAVLink SITL (port 14560) untuk kontrol drone.
- ROS 2 (/camera_front & /lidar_front) untuk deteksi visi dan deteksi tabrakan.

Cara menjalankan:
  python3 scripts/misi3.py
"""

import os
import sys
import time
import math
import threading
import traceback
import numpy as np
import cv2

# Otomatis gunakan venv ArduPilot jika modul pymavlink belum aktif
try:
    from pymavlink import mavutil
except ModuleNotFoundError:
    venv_py = os.path.expanduser('~/venv-ardupilot/bin/python3')
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise

# ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan

mavlink = mavutil.mavlink
KONEKSI_DEFAULT = 'udpin:127.0.0.1:14560'
TINGGI_TARGET = 1.35  # Ketinggian sejajar dengan titik tengah Bulls Eye

# ==============================================================================
# 1. SUBSYSTEM ROS 2: DETEKSI KAMERA & SENSOR LIDAR
# ==============================================================================
class VisionSensorListener:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.lidar_front = 99.0

        # Data deteksi target nomor 2 yang tersimpan (di-keep)
        self.target2_detected = False
        self.target2_cx = 320  # Pusat pixel horizontal (default tengah)
        self.target2_cy = 240  # Pusat pixel vertikal
        self.target2_area = 0
        self.target2_heading_saved = None
        self.target2_confidence = 0

    def on_camera_front(self, msg: Image):
        try:
            if msg.encoding in ('rgb8', 'bgr8'):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                if msg.encoding == 'rgb8':
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                else:
                    bgr = arr
            else:
                return

            with self.lock:
                self.latest_frame = bgr
                self._process_bullseye_target(bgr)
        except Exception:
            pass

    def on_lidar_front(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            with self.lock:
                self.lidar_front = min(valid) if valid else 99.0

    def _process_bullseye_target(self, frame):
        """Mendeteksi target nomor 2 di antara target 1, 2, 3 pada bullseye."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape

        # Thresholding adaptif untuk mengisolasi angka dan kontur gelap pada bullseye
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for c in contours:
            area = cv2.contourArea(c)
            # Filter luas kontur angka
            if 250 < area < 15000:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(h) / max(w, 1)

                # Angka 2 memiliki aspect ratio proporsional ~1.1 - 2.1
                if 1.1 < aspect_ratio < 2.2:
                    roi = thresh[y:y+h, x:x+w]
                    score = self._classify_digit_two(roi)
                    if score > 0.4:
                        cx = x + w // 2
                        cy = y + h // 2
                        candidates.append((score, cx, cy, area))

        # Jika kandidat terdeteksi, pilih skor terbaik
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            best_score, cx, cy, area = candidates[0]
            self.target2_detected = True
            self.target2_cx = cx
            self.target2_cy = cy
            self.target2_area = area
            self.target2_confidence = best_score
        else:
            # Metode Alternatif: Deteksi lingkaran bullseye warna
            # Target 2 berada di posisi tengah di antara target 1 dan 3
            self._detect_center_bullseye(frame)

    def _classify_digit_two(self, roi):
        """Memvalidasi bentuk angka 2: garis horizontal bawah tebal."""
        h, w = roi.shape
        if h < 10 or w < 8:
            return 0.0

        # Cek kepadatan piksel di bagian bawah (alas angka 2 horizontal penuh)
        bottom_strip = roi[int(h * 0.72):, :]
        bottom_density = np.sum(bottom_strip > 0) / float(bottom_strip.size + 1e-5)

        # Cek kepadatan piksel di bagian atas (lengkung atas)
        top_strip = roi[:int(h * 0.35), :]
        top_density = np.sum(top_strip > 0) / float(top_strip.size + 1e-5)

        if bottom_density > 0.40 and top_density > 0.25:
            return bottom_density
        return 0.0

    def _detect_center_bullseye(self, frame):
        """Fallback: Deteksi lingkaran sasaran bullseye konsentris di tengah."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Masking warna merah/biru lingkaran bullseye
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 100, 100]), np.array([180, 255, 255]))
        mask_red = mask1 | mask2

        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(c)
                ratio = float(w) / max(h, 1)
                if 0.7 < ratio < 1.4:
                    circles.append((x + w // 2, y + h // 2, area))

        # Jika ada beberapa lingkaran sasaran, pilih yang paling tengah (nomor 2)
        if len(circles) >= 2:
            circles.sort(key=lambda c: c[0])  # Urutkan dari kiri ke kanan
            # Pilih yang tengah (elemen index tengah)
            mid_idx = len(circles) // 2
            self.target2_detected = True
            self.target2_cx = circles[mid_idx][0]
            self.target2_cy = circles[mid_idx][1]
            self.target2_area = circles[mid_idx][2]
            self.target2_confidence = 0.5


# ==============================================================================
# 2. SUBSYSTEM MAVLINK: KONTROL PENERBANGAN DRONE
# ==============================================================================
class DroneController:
    def __init__(self, koneksi=KONEKSI_DEFAULT):
        print(f"[+] Menghubungkan ke autopilot: {koneksi} ...")
        self.m = mavutil.mavlink_connection(koneksi)
        self.m.mav.heartbeat_send(0, 0, 0, 0, 0)

        print("[*] Menunggu heartbeat dari autopilot...")
        self.m.wait_heartbeat(timeout=10)
        if self.m.target_system == 0:
            raise RuntimeError("Gagal terhubung. Pastikan SITL berjalan.")

        print(f"[OK] Terhubung ke drone (System ID: {self.m.target_system})")
        self.pos_z = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.yaw_deg = 90.0
        self.armed = False

        # Request telemetry stream (LOCAL_POSITION_NED & ATTITUDE)
        self._minta_stream(32, 20)  # LOCAL_POSITION_NED
        self._minta_stream(30, 20)  # ATTITUDE

        time.sleep(1.0)
        self.update()

    def _minta_stream(self, msg_id, hz):
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0
        )

    def update(self, timeout=0.05):
        msg = self.m.recv_match(blocking=True, timeout=timeout)
        while msg is not None:
            t = msg.get_type()
            if t == 'LOCAL_POSITION_NED':
                self.pos_x = msg.x
                self.pos_y = msg.y
                self.pos_z = msg.z  # Ketinggian: -z
            elif t == 'ATTITUDE':
                self.yaw_deg = math.degrees(msg.yaw)
                if self.yaw_deg < 0:
                    self.yaw_deg += 360.0
            elif t == 'HEARTBEAT' and msg.get_srcSystem() == self.m.target_system:
                self.armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            msg = self.m.recv_match(blocking=False)

    def set_mode(self, mode):
        self.m.set_mode(mode)
        t_end = time.time() + 5.0
        while time.time() < t_end:
            self.update()
            if self.m.flightmode == mode:
                print(f"[OK] Mode berubah ke {mode}")
                return True
            time.sleep(0.1)
        return False

    def send_velocity_body(self, vx, vy, vz, yaw_rate=0.0):
        """
        Mengirim kecepatan dalam Body Frame (FRD):
        vx : Maju (+)/Mundur (-)
        vy : Kanan (+)/Kiri (-)
        vz : Bawah (+)/Atas (-) -> Di-invers agar vz positif = naik
        yaw_rate : Putar searah jarum jam (+) / berlawanan (-) [rad/s]
        """
        # Type mask: abaikan posisi & akselerasi, gunakan kecepatan (vx, vy, vz) & yaw_rate
        type_mask = 0b000111000111  # bit 11 = yaw_rate aktif
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            mavlink.MAV_FRAME_BODY_OFFSET_NED,
            type_mask,
            0, 0, 0,
            vx, vy, -vz,
            0, 0, 0,
            0, yaw_rate
        )

    def move_timed(self, vx, vy, vz, detik, yaw_rate=0.0):
        t_end = time.time() + detik
        while time.time() < t_end:
            self.send_velocity_body(vx, vy, vz, yaw_rate)
            self.update(0.05)
            time.sleep(0.05)
        # Berhenti
        self.send_velocity_body(0, 0, 0, 0)

    def takeoff_if_needed(self, tinggi=TINGGI_TARGET):
        self.update()
        if not self.armed or (-self.pos_z < 0.3):
            print("[*] Drone belum di udara. Memulai prosedur takeoff otomatis...")
            self.set_mode('GUIDED')

            # Force Arm
            self.m.mav.command_long_send(
                self.m.target_system, self.m.target_component,
                mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1, 21196, 0, 0, 0, 0, 0
            )
            time.sleep(1.0)
            self.m.mav.command_long_send(
                self.m.target_system, self.m.target_component,
                mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                0, 0, 0, 0, 0, 0, tinggi
            )
            print(f"[*] Naik ke ketinggian {tinggi} m...")
            t_end = time.time() + 15.0
            while time.time() < t_end:
                self.update(0.1)
                if -self.pos_z >= tinggi * 0.85:
                    print(f"[OK] Sudah mencapai ketinggian {-self.pos_z:.2f} m.")
                    break
                time.sleep(0.1)

    def land(self):
        print("\n[*] Mendaratkan drone...")
        self.set_mode('LAND')


# ==============================================================================
# 3. ALGORITMA UTAMA MISI 3 (4 TAHAP)
# ==============================================================================
def jalankan_misi_3():
    print("=" * 65)
    print("      MEMULAI MISI 3: PENCARIAN & KAMIKAZE TARGET NOMOR 2      ")
    print("=" * 65)

    # 1. Inisialisasi ROS 2 Node untuk Kamera & LiDAR
    rclpy.init()
    listener = VisionSensorListener()
    node = Node('misi3_vision_node')
    node.create_subscription(Image, '/camera_front', listener.on_camera_front, 5)
    node.create_subscription(LaserScan, '/lidar_front', listener.on_lidar_front, 5)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # 2. Inisialisasi Drone Controller
    drone = DroneController()
    drone.takeoff_if_needed(TINGGI_TARGET)

    # Stabilisasi sejenak di depan donat merah
    print("\n[*] Menstabilkan posisi di ketinggian target...")
    drone.move_timed(0, 0, 0, 2.0)

    drone.update()
    base_yaw = drone.yaw_deg
    print(f"[INFO] Sudut Hadap Awal Drone: {base_yaw:.1f}°")

    # Penyimpan koordinat / heading target nomor 2 (di-keep)
    saved_target_heading = None
    saved_target_info = None

    # ==========================================================================
    # TAHAP 1 & 2: SCANNING YAW +- 60 DERAJAT & KEEP KOORDINAT NOMOR 2
    # ==========================================================================
    print("\n" + "-" * 60)
    print(">>> TAHAP 1: SCANNING YAW +- 60 DERAJAT UNTUK MEMINDAI SEMUA BULLSEYE")
    print(">>> TAHAP 2: SIMPAN (KEEP) KOORDINAT NOMOR 2 SAAT TERDETEKSI")
    print("-" * 60)

    # Kecepatan putar yaw ~20 deg/s (0.35 rad/s)
    yaw_speed_rad = 0.32

    # A. Putar ke KIRI +60 derajat (3 detik)
    print("[*] (1/3) Menyapu Yaw ke KIRI (+60°)...")
    t_start = time.time()
    while time.time() - t_start < 3.2:
        drone.send_velocity_body(0, 0, 0, yaw_rate=-yaw_speed_rad)
        drone.update(0.05)

        # Cek deteksi target nomor 2
        with listener.lock:
            if listener.target2_detected:
                # Hitung estimasi heading target dari pixel error
                err_px = listener.target2_cx - 320
                angle_offset_deg = (err_px / 320.0) * 35.0  # FOV horizontal kamera ~70 deg
                detected_heading = (drone.yaw_deg + angle_offset_deg) % 360.0

                if saved_target_heading is None or listener.target2_confidence > (saved_target_info or {}).get('conf', 0):
                    saved_target_heading = detected_heading
                    saved_target_info = {
                        'conf': listener.target2_confidence,
                        'cx': listener.target2_cx,
                        'cy': listener.target2_cy,
                        'area': listener.target2_area,
                        'drone_pos': (drone.pos_x, drone.pos_y, -drone.pos_z)
                    }
                    print(f"    [!] >>> TARGET NOMOR 2 TERDETEKSI! <<<")
                    print(f"        Heading Target: {saved_target_heading:.1f}° | Pixel: ({listener.target2_cx}, {listener.target2_cy})")
                    print("        Koordinat di-KEEP. Melanjutkan sweep +-60°...")
        time.sleep(0.05)

    # B. Putar menyapu ke KANAN -120 derajat (6.5 detik)
    print("[*] (2/3) Menyapu Yaw ke KANAN (-60° dari titik awal)...")
    t_start = time.time()
    while time.time() - t_start < 6.5:
        drone.send_velocity_body(0, 0, 0, yaw_rate=yaw_speed_rad)
        drone.update(0.05)

        with listener.lock:
            if listener.target2_detected:
                err_px = listener.target2_cx - 320
                angle_offset_deg = (err_px / 320.0) * 35.0
                detected_heading = (drone.yaw_deg + angle_offset_deg) % 360.0

                if saved_target_heading is None or listener.target2_confidence > (saved_target_info or {}).get('conf', 0):
                    saved_target_heading = detected_heading
                    saved_target_info = {
                        'conf': listener.target2_confidence,
                        'cx': listener.target2_cx,
                        'cy': listener.target2_cy,
                        'area': listener.target2_area,
                        'drone_pos': (drone.pos_x, drone.pos_y, -drone.pos_z)
                    }
                    print(f"    [!] >>> TARGET NOMOR 2 TERDETEKSI! <<<")
                    print(f"        Heading Target: {saved_target_heading:.1f}° | Pixel: ({listener.target2_cx}, {listener.target2_cy})")
                    print("        Koordinat di-KEEP. Melanjutkan sweep +-60°...")
        time.sleep(0.05)

    # C. Kembali ke arah tengah
    print("[*] (3/3) Mengembalikan Yaw ke arah tengah...")
    drone.move_timed(0, 0, 0, detik=3.2, yaw_rate=-yaw_speed_rad)

    # Validasi hasil keep koordinat
    if saved_target_heading is None:
        print("[!] Target nomor 2 tidak terdeteksi via OCR. Menggunakan posisi target tengah default...")
        # Default tengah arena target (Target 2 berada tepat di hadapan lurus ~90-95 deg)
        saved_target_heading = base_yaw

    print(f"\n[OK] Selesai scanning +-60°. Target nomor 2 terkunci pada Heading: {saved_target_heading:.1f}°")

    # ==========================================================================
    # TAHAP 3: DRONE MENGHADAP YAW & PITCH/ALTITUDE KE ARAH NOMOR 2
    # ==========================================================================
    print("\n" + "-" * 60)
    print(">>> TAHAP 3: MENGHADAP YAW & SEJAJARKAN POSISI KE TITIK TENGAH NOMOR 2")
    print("-" * 60)

    # Putar drone hingga menghadap persis ke heading target nomor 2 yang disimpan
    t_align = time.time()
    while time.time() - t_align < 4.0:
        drone.update(0.05)
        diff_yaw = (saved_target_heading - drone.yaw_deg + 180) % 360 - 180
        if abs(diff_yaw) < 2.0:
            break
        # Koreksi putaran
        rate_cmd = float(np.clip(diff_yaw * 0.03, -0.4, 0.4))
        drone.send_velocity_body(0, 0, 0, yaw_rate=-rate_cmd)
        time.sleep(0.05)

    drone.move_timed(0, 0, 0, 1.0)
    print("[OK] Drone telah menghadap lurus ke nomor 2!")

    # Penyelarasan visual halus (Fine Alignment via Kamera)
    print("[*] Melakukan fine-tuning visual (pusat target di tengah frame)...")
    t_fine = time.time()
    while time.time() - t_fine < 2.5:
        drone.update(0.05)
        with listener.lock:
            if listener.target2_detected:
                err_x = listener.target2_cx - 320
                err_y = listener.target2_cy - 240
                vy_fine = float(np.clip(err_x * 0.0015, -0.25, 0.25))
                vz_fine = float(np.clip(-err_y * 0.0015, -0.20, 0.20))
                drone.send_velocity_body(0.1, vy_fine, vz_fine, yaw_rate=0)
        time.sleep(0.05)

    drone.move_timed(0, 0, 0, 0.5)

    # ==========================================================================
    # TAHAP 4: TERBANG MENABRAK NOMOR 2 (KAMIKAZE CHARGE)
    # ==========================================================================
    print("\n" + "-" * 60)
    print(">>> TAHAP 4: DRONE TERBANG CEPAT HINGGA MENABRAK TARGET NOMOR 2")
    print("-" * 60)

    print("[🚀] MELUNCUR MENABRAK SASARAN NOMOR 2...")
    t_charge = time.time()
    menabrak = False

    while time.time() - t_charge < 12.0:
        drone.update(0.05)
        with listener.lock:
            d_front = listener.lidar_front
            cx = listener.target2_cx
            cy = listener.target2_cy
            detected = listener.target2_detected

        # Koreksi halus arah selama meluncur
        if detected:
            err_x = cx - 320
            vy_steer = float(np.clip(err_x * 0.002, -0.3, 0.3))
        else:
            vy_steer = 0.0

        # Terbang maju dengan kecepatan tinggi (1.4 m/s)
        drone.send_velocity_body(vx=1.4, vy=vy_steer, vz=0.0, yaw_rate=0.0)

        # Tampilkan status jarak
        sys.stdout.write(f"\r    Jarak ke sasaran: {d_front:.2f} m | Waktu: {time.time() - t_charge:.1f} s")
        sys.stdout.flush()

        # Deteksi tabrakan (jarak LiDAR depan sangat dekat < 0.35m atau -inf)
        if d_front < 0.35 or math.isinf(d_front) and d_front < 0:
            print(f"\n💥 [TABRAKAN TERDETEKSI] Sensor depan: {d_front:.2f} m!")
            menabrak = True
            break

        time.sleep(0.05)

    # Berhenti dan selesaikan misi
    drone.move_timed(0, 0, 0, 1.5)

    print("\n" + "=" * 65)
    if menabrak:
        print("🎯 BERHASIL! Drone sukses menabrak tepat pada sasaran nomor 2!")
    else:
        print("🎯 Misi selesai! Area target nomor 2 telah dicapai.")
    print("=" * 65)

    # Mendarat
    drone.land()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        jalankan_misi_3()
    except KeyboardInterrupt:
        print("\n[!] Misi dihentikan oleh pengguna.")
    except Exception:
        print("\n[!] Terjadi error dalam misi 3:")
        traceback.print_exc()
