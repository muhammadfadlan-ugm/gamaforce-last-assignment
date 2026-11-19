#!/usr/bin/env python3

import time
import traceback
import threading
import os
import cv2
from pymavlink import mavutil

# --- IMPORT ROS 2 ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image

# ===================================================================
# ====================== PENGATURAN UTAMA ===========================
# ===================================================================
# Koneksi ke SITL via MAVProxy (UDP)
KONEKSI = 'udpin:127.0.0.1:14560'
TINGGI = 1.3                 
THRESHOLD_LIDAR = 2.0             

# Variabel Global
jarak_depan = 99.0
jarak_kiri = 99.0
frame_kamera = None
jarak_kanan = 99.0
jarak_bawah = 99.0


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
# ===================================================================
# ====================== SENSOR ROS 2 ===============================
# ===================================================================
def imgmsg_to_cv2(img_msg: Image):
    import numpy as np
    if img_msg.encoding == 'bgr8':
        dtype, n_channels = np.uint8, 3
    elif img_msg.encoding == 'rgb8':
        dtype, n_channels = np.uint8, 3
    else:
        return None

    img_array = np.frombuffer(img_msg.data, dtype=dtype)
    img_array = img_array.reshape((img_msg.height, img_msg.width, n_channels))
    if img_msg.encoding == 'rgb8':
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_array

def lidar_callback(msg):
    global jarak_depan
    valid = [r for r in msg.ranges if 0.1 < r < 10.0]
    if valid:
        jarak_depan = min(valid)

def camera_callback(msg):
    global frame_kamera
    frame_kamera = imgmsg_to_cv2(msg)

def jalankan_ros():
    rclpy.init()
    node = Node('sensor_listener')
    node.create_subscription(LaserScan, '/lidar_front', lidar_callback, 10)
    node.create_subscription(Image, '/camera_front', camera_callback, 10)
    node.create_subscription(LaserScan, '/lidar_left', lidar_kiri_callback, 10
    )

    node.create_subscription(
        LaserScan, '/lidar_right', lidar_kanan_callback, 10
    )
    node.create_subscription(
            LaserScan, '/lidar_down', lidar_bawah_callback, 10
    )
    node.create_subscription(
                LaserScan, '/lidar_behind', lidar_belakang_callback, 10
        )
    node.create_subscription(
                LaserScan, '/lidar_up', lidar_kanan_callback, 10
        )
    print("[ROS 2] Sensor LiDAR & Kamera siap.")
    rclpy.spin(node)

# ===================================================================
# ====================== LOGIKA MAVLINK =============================
# ===================================================================
mavlink = mavutil.mavlink

class _Drone:
    def __init__(self, koneksi):
        print('Menghubungkan ke %s ...' % koneksi)
        self.m = mavutil.mavlink_connection(koneksi)
        self.m.mav.heartbeat_send(0, 0, 0, 0, 0)
        print('Menunggu heartbeat...')
        self.m.wait_heartbeat(timeout=10) 
        
        if self.m.target_system == 0:
            raise RuntimeError("Gagal terhubung. Pastikan MAVROS mati dan SITL jalan.")
        
        print(f"✅ Terhubung ke drone (sysid {self.m.target_system})")
        self.d = 0.0
        self.armed = False
        self.dapat_posisi = False
        self._minta_data(32, 10)  
        
        print('Menunggu data posisi (EKF/GPS siap)...')
        if not self._tunggu(lambda: self.dapat_posisi, 90, 'data posisi'):
            raise RuntimeError('Tidak ada data posisi.')

    def _minta_data(self, msg_id, hz):
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)

    def _update(self, tunggu=0.1):
        msg = self.m.recv_match(blocking=True, timeout=tunggu)
        while msg is not None:
            t = msg.get_type()
            if t == 'LOCAL_POSITION_NED':
                self.d = msg.z
                self.dapat_posisi = True
            elif t == 'HEARTBEAT' and msg.get_srcSystem() == self.m.target_system \
                    and msg.type != mavlink.MAV_TYPE_GCS:
                self.armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            msg = self.m.recv_match(blocking=False)

    def _tunggu(self, kondisi, timeout, nama=''):
        t_akhir = time.time() + timeout
        while time.time() < t_akhir:
            self._update(0.1)
            if kondisi():
                return True
        return False

    def _ganti_mode(self, mode):
        self.m.set_mode(mode)
        self._tunggu(lambda: self.m.flightmode == mode, 10, 'mode ' + mode)

    def _kirim_kecepatan(self, vx, vy, vz_bawah):
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b010111000111,           
            0, 0, 0, vx, vy, vz_bawah, 0, 0, 0, 0, 0)

    def takeoff(self, tinggi):
        print('Mengubah mode ke GUIDED...')
        self._ganti_mode('GUIDED')
        
        print('Mencoba ARM...')
        for i in range(30):
            # Kirim perintah ARM
            self.m.mav.command_long_send(
                self.m.target_system, self.m.target_component,
                mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 
                1, 21196, 0, 0, 0, 0, 0) # 21196 adalah force arm
            
            if self._tunggu(lambda: self.armed, 1, 'arm'):
                print("✅ Drone berhasil di-ARM")
                break
            print(f"Menunggu ARM... ({i+1}/30)")
        
        if not self.armed:
            raise RuntimeError("Gagal ARM drone setelah 30 kali percobaan.")

        print(f'Mengirim perintah Takeoff ke {tinggi} meter...')
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_NAV_TAKEOFF, 0, 
            0, 0, 0, 0, 0, 0, tinggi)
        
        # Tunggu sampai drone benar-benar naik
        berhasil_naik = self._tunggu(lambda: -self.d >= tinggi * 0.8, 20, 'naik ke ketinggian')
        
        if berhasil_naik:
            print(f'✅ Sudah di ketinggian {tinggi} m')
        else:
            print('⚠️ Peringatan: Drone mungkin belum mencapai ketinggian yang diinginkan.')

    def move_at_vel(self, vx, vy, vz, detik):
        t_akhir = time.time() + detik
        while time.time() < t_akhir:
            self._kirim_kecepatan(vx, vy, -vz)  
            self._update(0.1)

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
    global jarak_depan, frame_kamera,jarak_atas, jarak_bawah
    print(">>> MEMULAI MISI: NAVIGASI BERDASARKAN WAKTU...")
    time.sleep(3) # Tunggu stabil setelah takeoff
    
    # ==========================================
    # TAHAP 1: Terbang ke KANAN (berdasarkan waktu)
    # ==========================================
   
    print("Tahap 1: Bergerak ke kanan selama 34 detik...")
    move_at_vel(0, 1.0, 0, 34.60)

    print("✅ Selesai bergeser ke kanan. Berhenti sejenak.")
    move_at_vel(0, 0, 0, 1.0) # Berhenti 1 detik


    # ==========================================
    # TAHAP 3 & 4: Masuk lorong dan berhenti di ujung
    # ==========================================
    print("Tahap 3: Terbang masuk ke dalam lorong...")
    
    # Beri waktu maksimal 10 detik untuk masuk lorong
    waktu_mulai_masuk = time.time()
    
    while time.time() - waktu_mulai_masuk < 20.0:
        # Cek apakah ujung lorong sudah dekat
        if jarak_depan < 1.3:
            print(f"🛑 UJUNG LORONG TERDETEKSI ({jarak_depan:.2f}m)! Berhenti.")
            
            # Ambil foto di ujung lorong
            move_at_vel(0, 0, 0, 2.0) # Berhenti
            break # Misi selesai
        
        else:
            # Terbang MAJU masuk lorong
            move_at_vel(1, 0, 0, 0.4)
    print("tahap 4 lorong")
    waktu_mulai_zigzag = time.time()
        
    while time.time() - waktu_mulai_zigzag < 20.0:
            # Cek apakah ujung lorong sudah dekat
            if jarak_kiri < 1.0:
                print(f"🛑 UJUNG LORONG TERDETEKSI ({jarak_kiri:.2f}m)! Berhenti.")
                
                # Ambil foto di ujung lorong
                move_at_vel(0, 0, 0, 2.0) # Berhenti
                break # Misi selesai
            
            else:
                # Terbang MAJU masuk lorong
                move_at_vel(0, -1, 0, 0.4)
    waktu_mulai_masuk2 = time.time()
        
    while time.time() - waktu_mulai_masuk2 < 20.0:

            if jarak_depan < 1.3:
                print(f"🛑 UJUNG depan TERDETEKSI ({jarak_depan:.2f}m)! Berhenti.")
                
                
                move_at_vel(0, 0, 0, 2.0) 
                break 
            
            else:
                
                move_at_vel(1, 0, 0, 0.4)
    while time.time() - waktu_mulai_masuk2 < 20.0:
    
                if jarak_bawah > 1.6:
                    print(f"🛑 UJUNG bawah TERDETEKSI ({jarak_bawah:.2f}m)! Berhenti.")
                    
                    
                    move_at_vel(0, 0, 0, 2.0) 
                    break 
                
                else:
                    
                    move_at_vel(0, 0, 1, 0.4)
    while time.time() - waktu_mulai_masuk2 < 20.0:
    
                if jarak_depan < 1.3:
                    print(f"🛑 UJUNG LORONG TERDETEKSI ({jarak_depan:.2f}m)! Berhenti.")
                    
                    
                    move_at_vel(0, 0, 0, 2.0) 
                    break 
                
                else:
                    
                    move_at_vel(1, 0, 0, 0.4)
    
    print(">>> MISI SELESAI.")
    # print(">>> MEMULAI MISI LORONG...")
    # waktu_mulai = time.time()
    
    # BATAS_DEPAN = 0.9     # Jarak mulai menghindar saat ada sekat di depan
    # BATAS_DINDING = 0.55  # Jarak aman dari dinding kiri/kanan lorong


    #     # 2. Navigasi Horizontal (Maju & Menghindar Zig-Zag)
    # if jarak_depan < BATAS_DEPAN:
    #         # Berhenti maju, geser ke sisi yang lebih luas
    #         if jarak_kanan > jarak_kiri:
    #             move_at_vel(0.0, 0.5, 0, 0.15)   # Geser Kanan
    #         else:
    #             move_at_vel(0.0, -0.5, 0, 0.15)  # Geser Kiri
    # else:
    #         # Maju perlahan (vx = 0.5) sambil menjaga jarak aman dari dinding kiri/kanan
    #         vy = 0.0
    #         if jarak_kiri < BATAS_DINDING:
    #             vy = 0.35   # Menjauh dari dinding kiri (ke kanan)
    #         elif jarak_kanan < BATAS_DINDING:
    #             vy = -0.35  # Menjauh dari dinding kanan (ke kiri)

    #         move_at_vel(0.5, vy, 0, 0.1)

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