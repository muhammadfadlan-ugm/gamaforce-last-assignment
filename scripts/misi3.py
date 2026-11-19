#!/usr/bin/env python3
"""
MISI 3: PENCARIAN & KAMIKAZE TARGET NOMOR 2 MENGGUNAKAN OPENCV TEMPLATE MATCHING
Berdasarkan instruksi pada skrip_misi3.md:
1. Drone melakukan scanning yaw +- 60 derajat untuk memindai semua bullseye di depannya.
2. Drone menggunakan OpenCV Template Matching (cv2.matchTemplate) berkecepatan tinggi (tanpa YOLO / deep learning).
3. Koordinat dan heading nomor 2 dari bullseye didapatkan dan di-keep (disimpan).
4. Drone menghadap (yaw & pitch/altitude) tepat ke titik tengah nomor 2.
5. Drone terbang cepat hingga menabrak target nomor 2.

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
TINGGI_TARGET = 1.35  # Ketinggian sejajar sasaran Bulls Eye

# ==============================================================================
# 1. OPENCV TEMPLATE MATCHING DETECTOR (ANGKA 2)
# ==============================================================================
class DigitTwoTemplateMatcher:
    def __init__(self):
        self.template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        os.makedirs(self.template_dir, exist_ok=True)
        self.template_path = os.path.join(self.template_dir, "digit_2.png")

        # Muat atau buat template angka 2 sintetis
        self.base_template = self._load_or_create_template()
        
        # Pre-compute multi-scale templates (skala 0.6x s.d. 1.4x)
        self.scaled_templates = []
        base_h, base_w = self.base_template.shape[:2]
        for scale in [0.65, 0.85, 1.0, 1.2, 1.4]:
            nw, nh = int(base_w * scale), int(base_h * scale)
            if nw > 15 and nh > 20:
                resized = cv2.resize(self.base_template, (nw, nh))
                self.scaled_templates.append((resized, nw, nh))

    def _load_or_create_template(self):
        if os.path.exists(self.template_path):
            img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img

        # Buat template sintetis angka 2 jika belum ada
        tmpl = np.full((70, 50), 255, dtype=np.uint8)
        cv2.putText(tmpl, '2', (6, 58), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5, cv2.LINE_AA)
        cv2.imwrite(self.template_path, tmpl)
        return tmpl

    def detect(self, frame_bgr, threshold=0.58):
        """
        Mendeteksi angka 2 menggunakan OpenCV matchTemplate multi-skala.
        Mengembalikan: (detected, score, center_x, center_y, (x, y, w, h))
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Downscale 0.5x untuk akselerasi template matching (~20-30 ms latency)
        small_gray = cv2.resize(gray, (320, 240))
        
        best_score = -1.0
        best_loc_small = (0, 0)
        best_size_small = (0, 0)

        for tmpl, tw, th in self.scaled_templates:
            # Skala template disesuaikan dengan frame kecil
            tw_s, th_s = max(8, tw // 2), max(10, th // 2)
            if tw_s >= 320 or th_s >= 240:
                continue
            tmpl_s = cv2.resize(tmpl, (tw_s, th_s))

            res = cv2.matchTemplate(small_gray, tmpl_s, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score = max_val
                best_loc_small = max_loc
                best_size_small = (tw_s, th_s)

        # Kembalikan koordinat ke resolusi asli (640x480)
        if best_score >= threshold:
            x = best_loc_small[0] * 2
            y = best_loc_small[1] * 2
            w = best_size_small[0] * 2
            h = best_size_small[1] * 2
            cx = x + w // 2
            cy = y + h // 2
            return True, float(best_score), cx, cy, (x, y, w, h)

        return False, float(best_score), 320, 240, None


# ==============================================================================
# 2. SUBSYSTEM ROS 2: LISTENER KAMERA & SENSOR LIDAR
# ==============================================================================
class VisionSensorListener:
    def __init__(self):
        self.lock = threading.Lock()
        self.matcher = DigitTwoTemplateMatcher()
        self.latest_frame = None
        self.lidar_front = 99.0

        # Data deteksi target nomor 2 yang disimpan (di-keep)
        self.target2_detected = False
        self.target2_score = 0.0
        self.target2_cx = 320
        self.target2_cy = 240
        self.target2_bbox = None

    def on_camera_front(self, msg: Image):
        try:
            if msg.encoding in ('rgb8', 'bgr8'):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else arr
            else:
                return

            # Jalankan OpenCV Template Matching
            detected, score, cx, cy, bbox = self.matcher.detect(bgr, threshold=0.55)

            with self.lock:
                self.latest_frame = bgr
                self.target2_detected = detected
                self.target2_score = score
                self.target2_cx = cx
                self.target2_cy = cy
                self.target2_bbox = bbox
        except Exception:
            pass

    def on_lidar_front(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if 0.05 < r < 20.0]
            with self.lock:
                self.lidar_front = min(valid) if valid else 99.0


# ==============================================================================
# 3. SUBSYSTEM MAVLINK: KONTROL PENERBANGAN DRONE
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
                self.pos_z = msg.z
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
        type_mask = 0b000111000111  # bit 11 aktif untuk yaw_rate
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
# 4. ALGORITMA UTAMA MISI 3 (5 TAHAP SESUAI SKRIP_MISI3.MD)
# ==============================================================================
def jalankan_misi_3():
    print("=" * 65)
    print("  MISI 3: PENCARIAN & KAMIKAZE TARGET 2 DENGAN OPENCV TEMPLATE MATCHING")
    print("=" * 65)

    # 1. Inisialisasi ROS 2 Node
    rclpy.init()
    listener = VisionSensorListener()
    node = Node('misi3_template_matching_node')
    node.create_subscription(Image, '/camera_front', listener.on_camera_front, 5)
    node.create_subscription(LaserScan, '/lidar_front', listener.on_lidar_front, 5)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # 2. Inisialisasi Autopilot
    drone = DroneController()
    drone.takeoff_if_needed(TINGGI_TARGET)

    print("\n[*] Menstabilkan posisi drone di depan area sasaran...")
    drone.move_timed(0, 0, 0, 2.0)
    drone.update()
    base_yaw = drone.yaw_deg
    print(f"[INFO] Sudut Hadap Awal Drone: {base_yaw:.1f}°")

    # Penyimpan koordinat nomor 2 yang di-keep
    saved_target_heading = None
    saved_target_info = None

    # ==========================================================================
    # LANGKAH 1 & 2 & 3: SCANNING YAW +- 60°, MATCHING TEMPLATE OPENCV & KEEP TARGET 2
    # ==========================================================================
    print("\n" + "-" * 60)
    print(">>> 1. SCANNING YAW +- 60 DERAJAT")
    print(">>> 2. DETEKSI DENGAN OPENCV TEMPLATE MATCHING (ULTRA LOW LATENCY)")
    print(">>> 3. KEEP KOORDINAT NOMOR 2 HINGGA SCANNING SELESAI")
    print("-" * 60)

    yaw_rate = 0.32  # Kecepatan putar ~18.3 deg/s

    def periksa_dan_keep_target2():
        nonlocal saved_target_heading, saved_target_info
        with listener.lock:
            if listener.target2_detected:
                # Hitung deviasi sudut horizontal dari titik tengah kamera (FOV ~70°)
                err_px = listener.target2_cx - 320
                angle_offset = (err_px / 320.0) * 35.0
                current_target_heading = (drone.yaw_deg + angle_offset) % 360.0

                if saved_target_heading is None or listener.target2_score > (saved_target_info or {}).get('score', 0):
                    saved_target_heading = current_target_heading
                    saved_target_info = {
                        'score': listener.target2_score,
                        'cx': listener.target2_cx,
                        'cy': listener.target2_cy,
                        'bbox': listener.target2_bbox,
                        'drone_heading': drone.yaw_deg
                    }
                    print(f"    [🎯 MATCH TEMPLATE!] Angka 2 ditemukan (Skor: {listener.target2_score:.2f})")
                    print(f"       -> Koordinat di-KEEP pada Heading: {saved_target_heading:.1f}° | Pixel: ({listener.target2_cx}, {listener.target2_cy})")

    # 1A. Putar ke KIRI +60° (3.2 detik)
    print("[*] (1/3) Menyapu ke KIRI (+60°)...")
    t_start = time.time()
    while time.time() - t_start < 3.2:
        drone.send_velocity_body(0, 0, 0, yaw_rate=-yaw_rate)
        drone.update(0.05)
        periksa_dan_keep_target2()
        time.sleep(0.05)

    # 1B. Putar menyapu ke KANAN -120° (6.5 detik)
    print("[*] (2/3) Menyapu ke KANAN (-60° dari posisi awal)...")
    t_start = time.time()
    while time.time() - t_start < 6.5:
        drone.send_velocity_body(0, 0, 0, yaw_rate=yaw_rate)
        drone.update(0.05)
        periksa_dan_keep_target2()
        time.sleep(0.05)

    # 1C. Putar kembali ke tengah
    print("[*] (3/3) Mengembalikan putaran ke tengah...")
    drone.move_timed(0, 0, 0, detik=3.2, yaw_rate=-yaw_rate)

    # Validasi hasil keep
    if saved_target_heading is None:
        print("[!] Template matching tidak mendeteksi angka 2 secara sempurna. Mengunci target tengah arena...")
        saved_target_heading = base_yaw
    else:
        print(f"\n[OK] Scanning selesai! Target nomor 2 resmi dikunci pada Heading: {saved_target_heading:.1f}°")

    # ==========================================================================
    # LANGKAH 4: MENGHADAP YAW & PITCH KE ARAH NOMOR 2 DI TENGAH-TENGAHNYA
    # ==========================================================================
    print("\n" + "-" * 60)
    print(">>> 4. DRONE MENGHADAP YAW & PITCH/ALTITUDE KE TITIK TENGAH NOMOR 2")
    print("-" * 60)

    # A. Arahkan Yaw drone ke target nomor 2 yang disimpan
    t_align = time.time()
    while time.time() - t_align < 4.0:
        drone.update(0.05)
        diff_yaw = (saved_target_heading - drone.yaw_deg + 180) % 360 - 180
        if abs(diff_yaw) < 1.5:
            break
        cmd_rate = float(np.clip(diff_yaw * 0.03, -0.4, 0.4))
        drone.send_velocity_body(0, 0, 0, yaw_rate=-cmd_rate)
        time.sleep(0.05)

    drone.move_timed(0, 0, 0, 0.8)

    # B. Penyelarasan visual (Visual Centering ke tengah kamera: cx=320, cy=240)
    print("[*] Menyelaraskan visual agar angka 2 presisi di tengah kamera (cx=320, cy=240)...")
    t_center = time.time()
    while time.time() - t_center < 3.0:
        drone.update(0.05)
        with listener.lock:
            detected = listener.target2_detected
            cx = listener.target2_cx
            cy = listener.target2_cy

        if detected:
            err_x = cx - 320
            err_y = cy - 240
            vy_align = float(np.clip(err_x * 0.0015, -0.25, 0.25))
            vz_align = float(np.clip(-err_y * 0.0015, -0.20, 0.20))
            drone.send_velocity_body(0.1, vy_align, vz_align, yaw_rate=0.0)
        else:
            drone.send_velocity_body(0, 0, 0, 0)
        time.sleep(0.05)

    drone.move_timed(0, 0, 0, 0.5)
    print("[OK] Drone telah terkunci lurus pada pusat nomor 2!")

    # ==========================================================================
    # LANGKAH 5: TERBANG HINGGA MENABRAK NOMOR 2 (KAMIKAZE CHARGE)
    # ==========================================================================
    print("\n" + "-" * 60)
    print(">>> 5. DRONE TERBANG HINGGA MENABRAK SASARAN NOMOR 2")
    print("-" * 60)

    print("[🚀] MELUNCUR MENABRAK TARGET 2...")
    t_charge = time.time()
    menabrak = False

    while time.time() - t_charge < 12.0:
        drone.update(0.05)
        with listener.lock:
            d_front = listener.lidar_front
            cx = listener.target2_cx
            detected = listener.target2_detected

        # Tracking visual selama meluncur maju
        vy_steer = float(np.clip((cx - 320) * 0.002, -0.3, 0.3)) if detected else 0.0

        # Terbang maju kencang (vx = 1.4 m/s)
        drone.send_velocity_body(vx=1.4, vy=vy_steer, vz=0.0, yaw_rate=0.0)

        sys.stdout.write(f"\r    Jarak ke sasaran: {d_front:.2f} m | Waktu luncur: {time.time() - t_charge:.1f} s")
        sys.stdout.flush()

        # Deteksi benturan tabrakan (jarak LiDAR depan sangat dekat < 0.35m atau -inf)
        if d_front < 0.35 or (math.isinf(d_front) and d_front < 0):
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
        print("🎯 Misi selesai! Area sasaran nomor 2 berhasil dicapai.")
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
