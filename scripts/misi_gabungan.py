#!/usr/bin/env python3

import time
import math
import traceback
import threading
import cv2
import numpy as np
from pymavlink import mavutil

# --- IMPORT ROS 2 ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image

# ===================================================================
# ====================== PENGATURAN UTAMA ===========================
# ===================================================================
KONEKSI = 'udpin:127.0.0.1:14560'
TINGGI = 1.3
JARAK_GESER_AWAL = 11.7  # Sesuaikan meter pergeseran awal ke kanan

# Variabel Global Sensor
jarak_depan = 99.0
jarak_kiri = 99.0
jarak_kanan = 99.0
jarak_bawah = 99.0
jarak_atas = 99.0
jarak_belakang = 99.0
frame_kamera = None

# ===================================================================
# ====================== CALLBACK SENSOR ROS 2 ======================
# ===================================================================
def lidar_depan_callback(msg):
    global jarak_depan
    valid = [r for r in msg.ranges if 0.1 < r < 10.0]
    if valid:
        jarak_depan = min(valid)

def lidar_kanan_callback(msg):
    global jarak_kanan
    valid = [r for r in msg.ranges if 0.1 < r < 10.0]
    if valid:
        jarak_kanan = min(valid)

def lidar_kiri_callback(msg):
    global jarak_kiri
    valid = [r for r in msg.ranges if 0.1 < r < 10.0]
    if valid:
        jarak_kiri = min(valid)

def lidar_bawah_callback(msg):
    global jarak_bawah
    valid = [r for r in msg.ranges if 0.1 < r < 10.0]
    if valid:
        jarak_bawah = min(valid)

def lidar_atas_callback(msg):
    global jarak_atas
    valid = [r for r in msg.ranges if 0.1 < r < 10.0]
    if valid:
        jarak_atas = min(valid)

def lidar_belakang_callback(msg):
    global jarak_belakang
    valid = [r for r in msg.ranges if 0.1 < r < 10.0]
    if valid:
        jarak_belakang = min(valid)

def imgmsg_to_cv2(img_msg: Image):
    if img_msg.encoding in ('bgr8', 'rgb8'):
        img_array = np.frombuffer(img_msg.data, dtype=np.uint8)
        img_array = img_array.reshape((img_msg.height, img_msg.width, 3))
        if img_msg.encoding == 'rgb8':
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array
    return None

def camera_callback(msg):
    global frame_kamera
    frame_kamera = imgmsg_to_cv2(msg)

def jalankan_ros():
    rclpy.init()
    node = Node('sensor_listener')
    node.create_subscription(LaserScan, '/lidar_front', lidar_depan_callback, 10)
    node.create_subscription(LaserScan, '/lidar_left', lidar_kiri_callback, 10)
    node.create_subscription(LaserScan, '/lidar_right', lidar_kanan_callback, 10)
    node.create_subscription(LaserScan, '/lidar_down', lidar_bawah_callback, 10)
    node.create_subscription(LaserScan, '/lidar_back', lidar_belakang_callback, 10)
    node.create_subscription(LaserScan, '/lidar_up', lidar_atas_callback, 10)
    node.create_subscription(Image, '/camera_front', camera_callback, 10)
    node.create_subscription(Image, '/camera_down', camera_callback, 10)
    print("[ROS 2] Semua Sensor LiDAR 1D & Kamera siap.")
    rclpy.spin(node)

# ===================================================================
# ====================== LOGIKA MAVLINK =============================
# ===================================================================
mavlink = mavutil.mavlink

class _Drone:
    def __init__(self, koneksi):
        print(f'Menghubungkan ke {koneksi} ...')
        self.m = mavutil.mavlink_connection(koneksi)
        self.m.mav.heartbeat_send(0, 0, 0, 0, 0)
        print('Menunggu heartbeat...')
        self.m.wait_heartbeat(timeout=10)

        if self.m.target_system == 0:
            raise RuntimeError("Gagal terhubung. Pastikan MAVROS mati dan SITL jalan.")

        print(f"Terhubung ke drone (sysid {self.m.target_system})")
        self.n = 0.0
        self.e = 0.0
        self.d = 0.0
        self.yaw = 0.0
        self.yaw_referensi = 0.0
        self.armed = False
        self.dapat_posisi = False

        # Minta stream LOCAL_POSITION_NED (32) dan ATTITUDE (30) pada 20 Hz
        self._minta_data(32, 20)
        self._minta_data(30, 20)

        print('Menunggu data posisi (EKF/GPS siap)...')
        if not self._tunggu(lambda: self.dapat_posisi, 90, 'data posisi'):
            raise RuntimeError('Tidak ada data posisi.')

    def _minta_data(self, msg_id, hz):
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)

    def _update(self, tunggu=0.05):
        if tunggu > 0:
            msg = self.m.recv_match(blocking=True, timeout=tunggu)
        else:
            msg = self.m.recv_match(blocking=False)

        while msg is not None:
            t = msg.get_type()
            if t == 'LOCAL_POSITION_NED':
                self.n = msg.x   # North (m)
                self.e = msg.y   # East (m)
                self.d = msg.z   # Down (m)
                self.dapat_posisi = True
            elif t == 'ATTITUDE':
                self.yaw = msg.yaw
            elif t == 'HEARTBEAT' and msg.get_srcSystem() == self.m.target_system \
                    and msg.type != mavlink.MAV_TYPE_GCS:
                self.armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            msg = self.m.recv_match(blocking=False)

    def _tunggu(self, kondisi, timeout, nama=''):
        t_akhir = time.time() + timeout
        while time.time() < t_akhir:
            self._update(0.05)
            if kondisi():
                return True
        return False

    def _ganti_mode(self, mode):
        self.m.set_mode(mode)
        self._tunggu(lambda: self.m.flightmode == mode, 10, 'mode ' + mode)

    def _kirim_kecepatan_lokal_ned(self, vn, ve, vd, target_yaw=None):
        """Kirim kecepatan dalam koordinat Global NED dengan pengunci sudut yaw."""
        if target_yaw is None:
            type_mask = 0b0000111111000111
            yaw_cmd = 0.0
        else:
            type_mask = 0b0000101111000111
            yaw_cmd = target_yaw

        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0, 0, 0,
            vn, ve, vd,
            0, 0, 0,
            yaw_cmd, 0)

    def _kirim_kecepatan_body_terkunci_yaw(self, vx_maju, vy_kanan, vz_atas, target_yaw):
        """
        Konversi kecepatan bodi (maju, kanan, atas) ke Global NED
        sambil mengunci sudut yaw agar arah hadap drone tidak miring.
        """
        vn = vx_maju * math.cos(target_yaw) - vy_kanan * math.sin(target_yaw)
        ve = vx_maju * math.sin(target_yaw) + vy_kanan * math.cos(target_yaw)
        vd = -vz_atas
        self._kirim_kecepatan_lokal_ned(vn, ve, vd, target_yaw=target_yaw)

    def takeoff(self, tinggi):
        print('Mengubah mode ke GUIDED...')
        self._ganti_mode('GUIDED')

        print('Mencoba ARM...')
        for i in range(30):
            self.m.mav.command_long_send(
                self.m.target_system, self.m.target_component,
                mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1, 21196, 0, 0, 0, 0, 0)

            if self._tunggu(lambda: self.armed, 1, 'arm'):
                print("Drone berhasil di-ARM")
                break
            print(f"Menunggu ARM... ({i+1}/30)")

        if not self.armed:
            raise RuntimeError("Gagal ARM drone setelah 30 kali percobaan.")

        print(f'Mengirim perintah Takeoff ke {tinggi} meter...')
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, tinggi)

        berhasil_naik = self._tunggu(lambda: -self.d >= tinggi * 0.9, 20, 'naik ke ketinggian')
        if berhasil_naik:
            print(f'Sudah di ketinggian {-self.d:.2f} m')
        else:
            print('Peringatan: Drone mungkin belum mencapai ketinggian yang diinginkan.')

        # Simpan yaw awal sebagai patokan arah lurus sepanjang misi
        self._update(0.1)
        self.yaw_referensi = self.yaw

    def gerak_relatif_presisi(self, maju_m=0.0, kanan_m=0.0, naik_m=0.0, kecepatan_maks=1.2, toleransi=0.08):
        """
        Menggerakkan drone ke titik setpoint koordinat EKF tertentu secara akurat
        dengan pengereman P-Controller dan pengunci arah hadap.
        """
        self._update(0.1)
        yaw_kunci = self.yaw_referensi

        target_n = self.n + (maju_m * math.cos(yaw_kunci) - kanan_m * math.sin(yaw_kunci))
        target_e = self.e + (maju_m * math.sin(yaw_kunci) + kanan_m * math.cos(yaw_kunci))
        target_d = self.d - naik_m

        print(f"Setpoint EKF (Maju:{maju_m}m, Kanan:{kanan_m}m) -> Target NED({target_n:.2f}, {target_e:.2f}, {target_d:.2f})")

        dt = 1.0 / 30.0
        kp_xy = 1.2
        kp_z = 1.5

        while True:
            self._update(0.0)
            err_n = target_n - self.n
            err_e = target_e - self.e
            err_d = target_d - self.d

            jarak_sisa = math.sqrt(err_n**2 + err_e**2 + err_d**2)
            if jarak_sisa <= toleransi:
                break

            vn = err_n * kp_xy
            ve = err_e * kp_xy
            vd = max(-0.7, min(0.7, err_d * kp_z))

            kecepatan_xy = math.hypot(vn, ve)
            if kecepatan_xy > kecepatan_maks:
                skala = kecepatan_maks / kecepatan_xy
                vn *= skala
                ve *= skala

            self._kirim_kecepatan_lokal_ned(vn, ve, vd, target_yaw=yaw_kunci)
            time.sleep(dt)

        self.rem_aktif()
        print(f"Tiba di setpoint! Posisi aktual NED({self.n:.2f}, {self.e:.2f}, {self.d:.2f}) | Error: {jarak_sisa:.3f}m")

    def move_at_vel(self, vx_maju, vy_kanan, vz_atas, detik, rate_hz=30.0):
        """Menggerakkan drone berdasarkan kecepatan & waktu dengan yaw terkunci."""
        dt = 1.0 / rate_hz
        t_mulai = time.monotonic()
        t_akhir = t_mulai + detik
        next_tick = t_mulai

        while True:
            sekarang = time.monotonic()
            if sekarang >= t_akhir:
                break

            self._kirim_kecepatan_body_terkunci_yaw(vx_maju, vy_kanan, vz_atas, self.yaw_referensi)
            self._update(0.0)

            next_tick += dt
            sisa_tidur = next_tick - time.monotonic()
            if sisa_tidur > 0:
                time.sleep(min(sisa_tidur, max(0.0, t_akhir - time.monotonic())))
            else:
                next_tick = time.monotonic()

        if vx_maju == 0.0 and vy_kanan == 0.0 and vz_atas == 0.0:
            self.rem_aktif()

    def rem_aktif(self):
        """Menghentikan laju drone seketika di tempat."""
        for _ in range(5):
            self._kirim_kecepatan_lokal_ned(0.0, 0.0, 0.0, target_yaw=self.yaw_referensi)
            time.sleep(0.015)

    def mendarat(self):
        print('Mendarat...')
        self._ganti_mode('LAND')
        self._tunggu(lambda: (not self.armed) or -self.d < 0.15, 60, 'mendarat')

_drone = None

def move_at_vel(vx, vy, vz, detik):
    _drone.move_at_vel(vx, vy, vz, detik)

# ===================================================================
# ====================== MISI UTAMA =================================
# ===================================================================
def misi():
    global jarak_depan, jarak_kiri, jarak_kanan, jarak_atas, jarak_bawah, frame_kamera

    print(">>> MEMULAI MISI HYBRID (SETPOINT EKF + NAVIGASI LIDAR 1D)...")
    time.sleep(2.0)  # Tunggu stabil setelah takeoff

    # ==========================================
    # TAHAP 1: Geser ke KANAN Presisi (Setpoint EKF)
    # ==========================================
    print(f"\n[Tahap 1] Bergerak ke kanan tepat sejauh {JARAK_GESER_AWAL}m (EKF Setpoint)...")
    _drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=JARAK_GESER_AWAL, naik_m=0.0, kecepatan_maks=1.0)
    move_at_vel(0, 0, 0, 1.0)  # Hover stabil 1 detik sebelum masuk lorong

    # ==========================================
    # TAHAP 2: Masuk lorong 1 hingga ujung depan terdeteksi
    # ==========================================
    print("\n[Tahap 2] Terbang maju masuk ke dalam lorong 1...")
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 25.0:
        if jarak_depan < 1.3:
            print(f"uJUNG LORONG 1 TERDETEKSI ({jarak_depan:.2f}m)! Berhenti.")
            move_at_vel(0, 0, 0, 1.5)
            break
        else:
            # Melambat otomatis jika sudah mulai dekat (< 2.5m) agar tidak menabrak dinding
            vx = 0.45 if jarak_depan < 2.5 else 1.0
            move_at_vel(vx, 0, 0, 0.1)

    # ==========================================
    # TAHAP 3: Geser ke kiri di dalam lorong hingga dinding kiri terdeteksi
    # ==========================================
    print("\n[Tahap 3] Geser ke kiri dalam lorong...")
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 25.0:
        if jarak_kiri < 0.8:
            print(f"DINDING KIRI TERDETEKSI ({jarak_kiri:.2f}m)! Berhenti.")
            move_at_vel(0, 0, 0, 1.5)
            break
        else:
            vy = -0.45 if jarak_kiri < 2.0 else -1.0
            move_at_vel(0, vy, 0, 0.1)

    # ==========================================
    # TAHAP 4: Maju lorong 2 hingga ujung depan terdeteksi
    # ==========================================
    print("\n[Tahap 4] Terbang maju masuk ke lorong 2...")
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 25.0:
        if jarak_depan < 1.0:
            print(f"UJUNG DEPAN TERDETEKSI ({jarak_depan:.2f}m)! Berhenti.")
            move_at_vel(0, 0, 0, 1.5)
            break
        else:
            vx = 0.45 if jarak_depan < 2.5 else 1.0
            move_at_vel(vx, 0, 0, 0.1)

    # ==========================================
    # TAHAP 5: Naik ketinggian berdasarkan LiDAR bawah
    # ==========================================
    print("\n[Tahap 5] Naik ketinggian melewati rintangan bawah...")
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 15.0:
        if jarak_bawah > 1.3:
            print(f"KETINGGIAN TERCAPAI (LiDAR bawah: {jarak_bawah:.2f}m)! Berhenti.")
            move_at_vel(0, 0, 0, 1.0)
            break
        else:
            move_at_vel(0, 0, 0.4, 0.05)

    # ==========================================
    # TAHAP 6: Maju melewati rintangan hingga ujung depan
    # ==========================================
    print("\n[Tahap 6] Terbang maju melewati rintangan...")
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 25.0:
        if jarak_depan < 1.3:
            print(f"UJUNG LORONG TERDETEKSI ({jarak_depan:.2f}m)! Berhenti.")
            move_at_vel(0, 0, 0, 1.5)
            break
        else:
            vx = 0.45 if jarak_depan < 2.5 else 1.0
            move_at_vel(vx, 0, 0, 0.1)

    # ==========================================
    # TAHAP 7: Turun kembali ke ketinggian normal
    # ==========================================
    print("\n[Tahap 7] Menurunkan ketinggian...")
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 20.0:
        if 0.6 < jarak_bawah < 0.85:
            print(f"KETINGGIAN NORMAL TERCAPAI (LiDAR bawah: {jarak_bawah:.2f}m)! Berhenti.")
            move_at_vel(0, 0, 0, 1.5)
            break
        else:
            move_at_vel(0, 0, -0.5, 0.05)
    print("\n[Tahap 8] Terbang maju melewati rintangan...")
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 25.0:
        if jarak_depan < 1.2:
            print(f"UJUNG LORONG TERDETEKSI ({jarak_depan:.2f}m)! Berhenti.")
            move_at_vel(0, 0, 0, 1.5)
            break
        else:
            vx = 0.45 if jarak_depan < 2.5 else 1.0
            move_at_vel(vx, 0, 0, 0.1)
    print("\n[Tahap 9] Terbang atas melewati rintangan...")
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 25.0:
            if jarak_bawah > 1.5:
                print(f"UJUNG LORONG TERDETEKSI ({jarak_bawah:.2f}m)! Berhenti.")
                move_at_vel(0, 0, 0, 1.5)
                break
            else:
                vz = 0.4 if jarak_bawah < 2.5 else 1.0
                move_at_vel(0, 0, vz, 0.1)
    print("\n[Tahap 10] Terbang minggir untuk masuk lobang...")
    move_at_vel(0,-0.2,0,3.3)
    waktu_mulai = time.time()   
    while time.time() - waktu_mulai < 25.0:
            if jarak_depan > 1.2:
                print(f"BOLONGAN TERDETEKSI ({jarak_bawah:.2f}m)! Berhenti.")
                move_at_vel(0, 0, 0, 1.5)
                break
            else:
                vy = -0.3 if jarak_depan < 2.5 else 1.0
                move_at_vel(0, vy, 0, 0.1)    
    print("\n[Tahap 11] Terbang maju menuju lobang...")
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 25.0:
            if jarak_depan < 1.45:
                print(f"UJUNG LORONG TERDETEKSI ({jarak_depan:.2f}m)! Berhenti.")
                move_at_vel(0, 0, 0, 1.5)
                break
            else:
                vx = 0.4 if jarak_bawah < 2.5 else 1.0
                move_at_vel(vx, 0, 0, 0.1)
 
    print("\n[Tahap 12] Terbang maju untuk masuk lobang...")
    
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 25.0:
            if jarak_depan < 1.5:
                print(f"tembok TERDETEKSI ({jarak_bawah:.2f}m)! Berhenti.")
                move_at_vel(0, 0, 0, 1.5)
                break
            else:
                vx = 0.4 if jarak_depan < 2.5 else 1.0
                move_at_vel(vx, 0, 0, 0.1)
    print("\n[Tahap 13] Terbang menurun untuk melewati rintangan...")
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 25.0:
            if jarak_depan < 0.4 and jarak_bawah < 0.4:
                print(f"tembok TERDETEKSI ({jarak_bawah:.2f}m)! Berhenti.")
                move_at_vel(0, 0, 0, 1.5)
                break
            else:
                vx = 0.3 if jarak_depan < 2.5 else 1.0
                vz = -0.22 if jarak_bawah < 2.0 else 1.0
                move_at_vel(vx, 0, vz, 0.1)
    print("\n[Tahap 14] Terbang naik untuk melewati rintangan...")
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 25.0:
        if jarak_atas < 1.2:
            print(f"tembok TERDETEKSI ({jarak_bawah:.2f}m)! Berhenti.")
            move_at_vel(0, 0, 0, 1.5)
            break
        else:
            vz = 0.45 if jarak_atas < 2.0 else 1.0
            move_at_vel(0, 0, vz, 0.1)
    print("\n[Tahap 15] Terbang maju untuk keluar labirin...")
        
    waktu_mulai = time.time()
    while time.time() - waktu_mulai < 25.0:
        if jarak_atas > 5.0 :
            print(f"tembok Hilang ({jarak_kiri:.2f}m)! Berhenti.")
            move_at_vel(0, 0, 0, 1.5)
            break
        else:
            vx = 0.4 if jarak_kiri or jarak_kanan < 2.5 else 1.0
            move_at_vel(vx, 0, 0, 0.1)
    
    print("\n>>> MISI SELESAI.")

# ===================================================================
# ====================== EKSEKUSI UTAMA =============================
# ===================================================================
if __name__ == '__main__':
    ros_thread = threading.Thread(target=jalankan_ros, daemon=True)
    ros_thread.start()
    time.sleep(2.0)

    try:
        _drone = _Drone(KONEKSI)
        _drone.takeoff(TINGGI)
        misi()
    except BaseException:
        print('\nMISI BERHENTI:')
        traceback.print_exc()
    finally:
        if _drone:
            _drone.mendarat()