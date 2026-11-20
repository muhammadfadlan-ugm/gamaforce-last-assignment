#!/usr/bin/env python3
"""
Misi jalur KIRI (Track Vision) - OPREC Gamaforce.

Isi jalur kiri (koordinat dunia, hasil baca world_oprec.world + obstacles.dae):
  - Tanjakan oranye  : x -21.9..-19.4, y -9.0..-5.8, miring sampai z~2.1 (menghalangi jalur lurus)
  - Lengkung merah   : y ~ -1.4, z 1.17..4.19 (menggantung tinggi, drone z~0.6 lewat di bawahnya)
  - Lorong 1 (utara) : x -21.19..-19.70 (lebar 1.49 m), y 2.26..9.76, dinding tinggi 1.05 m
  - Marker ArUco     : di dinding utara ujung lorong 1 (x -20.67, z 0.63)
  - Lorong 2 (timur) : y 8.26..9.76 (lebar 1.5 m), x -19.7..-14.9
  - Lingkaran hijau  : di ujung lorong 2 (x -13.2, y 9.04), disusul pelat oranye di x -12.1

Urutan tahap:
  1. Geser ke kiri sejauh JARAK_GESER_AWAL (TIDAK DIUBAH)
  2. Maju sampai tanjakan terdeteksi lidar depan
  3. Hindari tanjakan lewat sisi kanan, lalu kembali ke sumbu lorong
  4. Maju sampai dinding lorong 1 terdeteksi lidar kiri & kanan
  5. Lorong 1: maju di tengah (lidar kiri-kanan) sambil scan ArUco, berhenti dekat dinding ujung
  6. Putar yaw 90 derajat ke kanan (belok ke lorong 2)
  7. Lorong 2: maju di tengah (lidar kiri-kanan)
  8. Pusatkan drone di lingkaran hijau (HSV kamera bawah), lalu mendarat

Deteksi bentuk berwarna (HSV OpenCV) berjalan terus di kamera depan & bawah sejak awal.
"""

import math
import threading
import time
import traceback
import cv2
import numpy as np
from pymavlink import mavutil

# --- IMPORT ROS 2 ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan

# ===================================================================
# ====================== PENGATURAN UTAMA ===========================
# ===================================================================
KONEKSI = 'udpin:127.0.0.1:14560'
TINGGI = 0.4
JARAK_GESER_AWAL = 20.6  # Sesuaikan meter pergeseran awal ke kanan

# Pengaturan ArUco (ubah ke DICT_4X4_250 / DICT_APRILTAG_16h5 jika ID belum terbaca)
JENIS_ARUCO = cv2.aruco.DICT_4X4_50
TAMPILKAN_KAMERA = False  # Set False jika ingin running headless tanpa jendela OpenCV
MODE_KALIBRASI_HSV = False  # True: tidak terbang, hanya cetak nilai HSV piksel tengah kamera

# ----- Parameter navigasi jalur kiri -----
JARAK_STOP_TANJAKAN = 1.5   # m, berhenti kalau lidar depan < ini (tanjakan oranye)
GESER_LEBIH = 0.6           # m, kelonggaran setelah ujung tanjakan lolos dari lidar depan
UJUNG_TANJAKAN = 3.5        # m, jarak dari titik lidar kena tanjakan sampai ujung utaranya + margin
OFFSET_SUMBU_LORONG = 0.155 # m, sumbu lorong 1 (x=-20.445) ada segini di KANAN titik akhir Tahap 1 (x=-20.6)
KP_GARIS = 0.8              # gain menahan garis lurus (EKF) saat maju
BATAS_DEKAT_LORONG = 1.5    # m, lidar kiri & kanan sama-sama di bawah ini = sudah masuk lorong (kiri+kanan ~1.3 m)
DEPAN_UJUNG_LORONG1 = 0.70  # m, jarak berhenti dari dinding ujung lorong 1 (pas di tengah lorong 2)
JARAK_LORONG2 = 7.0         # m, panjang tempuh lorong 2 (berhenti di atas lingkaran hijau)
DEPAN_LORONG2 = 1.2         # m, berhenti lebih awal kalau ada dinding di depan
BATAS_LORONG = 1.2          # m, lidar samping di bawah ini dianggap dinding lorong
KP_TENGAH = 0.8             # gain menjaga posisi tengah lorong
BATAS_VY_TENGAH = 0.25      # m/s, koreksi samping maksimum

# ----- Pengaturan deteksi bentuk berwarna (HSV OpenCV: H 0-179, S 0-255, V 0-255) -----
# Rentang diturunkan dari warna material di obstacles.dae. Kalau warna di Gazebo terlihat
# bergeser karena pencahayaan, geser H/S/V di sini (pakai MODE_KALIBRASI_HSV).
HSV_WARNA = {
    'merah': [((0, 150, 60), (6, 255, 255)), ((172, 150, 60), (179, 255, 255))],
    'oranye': [((7, 170, 60), (20, 255, 255))],
    'kuning': [((22, 180, 60), (34, 255, 255))],
    'hijau': [((45, 150, 50), (75, 255, 255))],
    'cyan': [((85, 150, 50), (102, 255, 255))],
    'biru': [((108, 150, 40), (130, 255, 255))],
    'magenta': [((140, 150, 50), (168, 255, 255))],
}
WARNA_DIABAIKAN = ('biru',)   # biru = dinding lorong, bukan bentuk target
MIN_LUAS_BENTUK = 400         # piksel; bentuk lebih kecil dianggap noise
MIN_LUAS_LUBANG = 150
KERNEL_MORF = np.ones((5, 5), np.uint8)

# Variabel Global Sensor LiDAR
jarak_depan = 99.0
jarak_kiri = 99.0
jarak_kanan = 99.0
jarak_bawah = 99.0
jarak_atas = 99.0
jarak_belakang = 99.0

# Variabel Global Kamera & ArUco
frame_kamera_depan = None
frame_kamera_bawah = None

aruco_front_id = None
aruco_front_pos = None  # (cx, cy)
aruco_down_id = None
aruco_down_pos = None  # (cx, cy)
riwayat_aruco = set()  # Menyimpan semua ID unik yang pernah terbaca selama misi

# Variabel Global Bentuk Berwarna (hasil deteksi HSV terbaru + riwayat)
bentuk_depan = []
bentuk_bawah = []
riwayat_bentuk = set()
_hitung_frame = {'DEPAN': 0, 'BAWAH': 0}

# ===================================================================
# ====================== INISIALISASI ARUCO =========================
# ===================================================================
aruco_dict = cv2.aruco.getPredefinedDictionary(JENIS_ARUCO)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def proses_deteksi_aruco(frame, sumber_kamera):
  if frame is None:
    return None, None

  corners, ids, _ = aruco_detector.detectMarkers(frame)
  if ids is not None:
    ids_flat = np.asarray(ids).flatten()
    if len(ids_flat) > 0:
      marker_id = int(ids_flat[0])
      c = np.asarray(corners[0]).reshape(-1, 2)
      cx = int(c[:, 0].mean())
      cy = int(c[:, 1].mean())

      if (sumber_kamera, marker_id) not in riwayat_aruco:
        riwayat_aruco.add((sumber_kamera, marker_id))
        print(
            f'[ARUCO {sumber_kamera}] Terdeteksi ID: {marker_id} pada piksel'
            f' ({cx}, {cy}) [Lanjut misi...]'
        )
      return marker_id, (cx, cy)

  return None, None


# ===================================================================
# ====================== DETEKSI BENTUK BERWARNA (HSV) ==============
# ===================================================================
def klasifikasi_bentuk(kontur):
  """Nama bangun datar dari sebuah kontur: segitiga, persegi, lingkaran, dst."""
  peri = cv2.arcLength(kontur, True)
  luas = cv2.contourArea(kontur)
  if peri == 0 or luas == 0:
    return 'tak dikenal'

  approx = cv2.approxPolyDP(kontur, 0.03 * peri, True)
  n_sisi = len(approx)
  bulat = 4.0 * math.pi * luas / (peri * peri)  # 1.0 = lingkaran sempurna

  if n_sisi == 3:
    return 'segitiga'
  if n_sisi == 4 and cv2.isContourConvex(approx):
    (_, _), (w, h), _ = cv2.minAreaRect(approx)
    if min(w, h) == 0:
      return 'tak dikenal'
    rasio = max(w, h) / min(w, h)
    return 'persegi' if rasio < 1.2 else 'persegi panjang'
  if n_sisi == 5:
    return 'segilima'
  if n_sisi == 6:
    return 'segienam'
  if bulat > 0.80:
    if len(kontur) >= 5:
      (_, _), (a, b), _ = cv2.fitEllipse(kontur)
      if min(a, b) > 0 and max(a, b) / min(a, b) > 1.25:
        return 'elips'
    return 'lingkaran'
  return 'tak beraturan'


def _pusat_kontur(kontur):
  m = cv2.moments(kontur)
  if m['m00'] == 0:
    x, y, w, h = cv2.boundingRect(kontur)
    return (x + w // 2, y + h // 2)
  return (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))


def deteksi_bentuk_hsv(frame):
  """
  Cari bangun datar berwarna di frame BGR memakai HSV.
  Return list dict (terbesar dulu):
    warna, bentuk, pusat (px, py), luas, lubang (None atau {'bentuk', 'pusat'})
  'lubang' diisi kalau bentuknya berbingkai (ada lubang di tengah, mis. gerbang).
  """
  if frame is None:
    return []

  hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
  hasil = []
  for warna, rentang in HSV_WARNA.items():
    if warna in WARNA_DIABAIKAN:
      continue
    mask = None
    for bawah, atas in rentang:
      m = cv2.inRange(hsv, np.array(bawah), np.array(atas))
      mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_MORF)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_MORF)

    kontur, hirarki = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hirarki is None:
      continue
    for i, c in enumerate(kontur):
      if hirarki[0][i][3] != -1:  # kontur lubang, bukan bentuk luar
        continue
      luas = cv2.contourArea(c)
      if luas < MIN_LUAS_BENTUK:
        continue

      lubang = None
      anak = hirarki[0][i][2]
      terbesar = None
      while anak != -1:
        luas_anak = cv2.contourArea(kontur[anak])
        if luas_anak >= MIN_LUAS_LUBANG and (
            terbesar is None or luas_anak > terbesar[0]
        ):
          terbesar = (luas_anak, anak)
        anak = hirarki[0][anak][0]
      if terbesar is not None:
        lubang = {
            'bentuk': klasifikasi_bentuk(kontur[terbesar[1]]),
            'pusat': _pusat_kontur(kontur[terbesar[1]]),
        }

      hasil.append({
          'warna': warna,
          'bentuk': klasifikasi_bentuk(c),
          'pusat': _pusat_kontur(c),
          'luas': luas,
          'lubang': lubang,
      })

  hasil.sort(key=lambda d: -d['luas'])
  return hasil


def proses_deteksi_bentuk(frame, sumber_kamera):
  """Jalankan deteksi HSV (tiap 2 frame agar CPU ringan) & catat bentuk baru ke log."""
  _hitung_frame[sumber_kamera] += 1
  if _hitung_frame[sumber_kamera] % 2 != 0:
    return None

  daftar = deteksi_bentuk_hsv(frame)
  for d in daftar:
    kunci = (
        sumber_kamera,
        d['warna'],
        d['bentuk'],
        d['lubang']['bentuk'] if d['lubang'] else None,
    )
    if kunci not in riwayat_bentuk:
      riwayat_bentuk.add(kunci)
      ket_lubang = f" | lubang: {d['lubang']['bentuk']}" if d['lubang'] else ''
      print(
          f"[HSV {sumber_kamera}] {d['warna']} - {d['bentuk']} di"
          f" {d['pusat']} (luas {int(d['luas'])}px){ket_lubang}"
      )
  return daftar


def bentuk_terbesar(daftar, warna=None, bentuk=None):
  """Ambil bentuk terbesar dari hasil deteksi, bisa difilter warna/bentuk."""
  for d in daftar or []:
    if (warna is None or d['warna'] == warna) and (
        bentuk is None or d['bentuk'] == bentuk
    ):
      return d
  return None


def gambar_deteksi(frame, daftar):
  for d in daftar or []:
    cx, cy = d['pusat']
    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
    teks = f"{d['warna']} {d['bentuk']}"
    if d['lubang']:
      teks += f" (lubang {d['lubang']['bentuk']})"
    cv2.putText(frame, teks, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)


def cetak_hsv_tengah(frame, sumber):
  """Untuk kalibrasi: cetak HSV piksel tengah gambar."""
  h, w = frame.shape[:2]
  patch = frame[h // 2 - 3:h // 2 + 4, w // 2 - 3:w // 2 + 4]
  hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(axis=0)
  print(f'[KALIBRASI {sumber}] HSV tengah = H:{hsv[0]:.0f} S:{hsv[1]:.0f} V:{hsv[2]:.0f}')


# ===================================================================
# ====================== CALLBACK SENSOR ROS 2 ======================
# ===================================================================
def _baca_lidar(msg):
  """
  +inf = tidak ada benda dalam 10 m -> 10.0 (bukan nilai lama!)
  -inf / < 0.12 m = terlalu dekat   -> 0.0
  """
  if not msg.ranges:
    return None
  baik = [r for r in msg.ranges if r == r]  # buang NaN
  if not baik:
    return None
  valid = [r for r in baik if 0.12 <= r < 10.0]
  if valid:
    return min(valid)
  if any(r < 0.12 for r in baik):
    return 0.0
  return 10.0


def lidar_depan_callback(msg):
  global jarak_depan
  r = _baca_lidar(msg)
  if r is not None:
    jarak_depan = r


def lidar_kanan_callback(msg):
  global jarak_kanan
  r = _baca_lidar(msg)
  if r is not None:
    jarak_kanan = r


def lidar_kiri_callback(msg):
  global jarak_kiri
  r = _baca_lidar(msg)
  if r is not None:
    jarak_kiri = r


def lidar_bawah_callback(msg):
  global jarak_bawah
  r = _baca_lidar(msg)
  if r is not None:
    jarak_bawah = r


def lidar_atas_callback(msg):
  global jarak_atas
  r = _baca_lidar(msg)
  if r is not None:
    jarak_atas = r


def lidar_belakang_callback(msg):
  global jarak_belakang
  r = _baca_lidar(msg)
  if r is not None:
    jarak_belakang = r


def imgmsg_to_cv2(img_msg: Image):
  if img_msg.encoding in ('bgr8', 'rgb8'):
    img_array = np.frombuffer(img_msg.data, dtype=np.uint8)
    img_array = img_array.reshape((img_msg.height, img_msg.width, 3))
    if img_msg.encoding == 'rgb8':
      img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
      img_array = img_array.copy()  # Copy agar writable oleh OpenCV
    return img_array
  return None


def camera_front_callback(msg):
  global frame_kamera_depan, aruco_front_id, aruco_front_pos, bentuk_depan
  frame = imgmsg_to_cv2(msg)
  if frame is not None:
    if MODE_KALIBRASI_HSV:
      _hitung_frame['DEPAN'] += 1
      if _hitung_frame['DEPAN'] % 30 == 0:
        cetak_hsv_tengah(frame, 'DEPAN')
      return
    aruco_front_id, aruco_front_pos = proses_deteksi_aruco(frame, 'DEPAN')
    hasil = proses_deteksi_bentuk(frame, 'DEPAN')
    if hasil is not None:
      bentuk_depan = hasil
    frame_kamera_depan = frame
    if TAMPILKAN_KAMERA:
      gambar_deteksi(frame_kamera_depan, bentuk_depan)
      cv2.imshow('Kamera Depan (ArUco + HSV)', frame_kamera_depan)
      cv2.waitKey(1)


def camera_down_callback(msg):
  global frame_kamera_bawah, aruco_down_id, aruco_down_pos, bentuk_bawah
  frame = imgmsg_to_cv2(msg)
  if frame is not None:
    if MODE_KALIBRASI_HSV:
      _hitung_frame['BAWAH'] += 1
      if _hitung_frame['BAWAH'] % 30 == 0:
        cetak_hsv_tengah(frame, 'BAWAH')
      return
    aruco_down_id, aruco_down_pos = proses_deteksi_aruco(frame, 'BAWAH')
    hasil = proses_deteksi_bentuk(frame, 'BAWAH')
    if hasil is not None:
      bentuk_bawah = hasil
    frame_kamera_bawah = frame
    if TAMPILKAN_KAMERA:
      gambar_deteksi(frame_kamera_bawah, bentuk_bawah)
      cv2.imshow('Kamera Bawah (ArUco + HSV)', frame_kamera_bawah)
      cv2.waitKey(1)


def jalankan_ros():
  rclpy.init()
  node = Node('sensor_listener')
  node.create_subscription(LaserScan, '/lidar_front', lidar_depan_callback, 10)
  node.create_subscription(LaserScan, '/lidar_left', lidar_kiri_callback, 10)
  node.create_subscription(LaserScan, '/lidar_right', lidar_kanan_callback, 10)
  node.create_subscription(LaserScan, '/lidar_down', lidar_bawah_callback, 10)
  node.create_subscription(
      LaserScan, '/lidar_back', lidar_belakang_callback, 10
  )
  node.create_subscription(LaserScan, '/lidar_up', lidar_atas_callback, 10)
  node.create_subscription(Image, '/camera_front', camera_front_callback, 10)
  node.create_subscription(Image, '/camera_down', camera_down_callback, 10)
  print('[ROS 2] Semua Sensor LiDAR 1D & Kamera ArUco + HSV siap.')
  rclpy.spin(node)


# ===================================================================
# ====================== LOGIKA MAVLINK =============================
# ===================================================================
mavlink = mavutil.mavlink


def _norm_sudut(a):
  """Bungkus sudut ke -pi..pi."""
  return math.atan2(math.sin(a), math.cos(a))


class _Drone:

  def __init__(self, koneksi):
    print(f'Menghubungkan ke {koneksi} ...')
    self.m = mavutil.mavlink_connection(koneksi)
    self.m.mav.heartbeat_send(0, 0, 0, 0, 0)
    print('Menunggu heartbeat...')
    self.m.wait_heartbeat(timeout=10)

    if self.m.target_system == 0:
      raise RuntimeError(
          'Gagal terhubung. Pastikan MAVROS mati dan SITL jalan.'
      )

    print(f'Terhubung ke drone (sysid {self.m.target_system})')
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
        self.m.target_system,
        self.m.target_component,
        mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id,
        int(1e6 / hz),
        0,
        0,
        0,
        0,
        0,
    )

  def _update(self, tunggu=0.05):
    if tunggu > 0:
      msg = self.m.recv_match(blocking=True, timeout=tunggu)
    else:
      msg = self.m.recv_match(blocking=False)

    while msg is not None:
      t = msg.get_type()
      if t == 'LOCAL_POSITION_NED':
        self.n = msg.x  # North (m)
        self.e = msg.y  # East (m)
        self.d = msg.z  # Down (m)
        self.dapat_posisi = True
      elif t == 'ATTITUDE':
        self.yaw = msg.yaw
      elif (
          t == 'HEARTBEAT'
          and msg.get_srcSystem() == self.m.target_system
          and msg.type != mavlink.MAV_TYPE_GCS
      ):
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
        0,
        self.m.target_system,
        self.m.target_component,
        mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0,
        0,
        0,
        vn,
        ve,
        vd,
        0,
        0,
        0,
        yaw_cmd,
        0,
    )

  def _kirim_kecepatan_body_terkunci_yaw(
      self, vx_maju, vy_kanan, vz_atas, target_yaw
  ):
    """Konversi kecepatan bodi (maju, kanan, atas) ke Global NED sambil mengunci yaw.

    Arah maju  = (cos yaw, sin yaw) di (utara, timur)
    Arah kanan = (-sin yaw, cos yaw)   -> vy_kanan > 0 benar-benar ke KANAN.
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
          self.m.target_system,
          self.m.target_component,
          mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
          0,
          1,
          21196,
          0,
          0,
          0,
          0,
          0,
      )

      if self._tunggu(lambda: self.armed, 1, 'arm'):
        print('Drone berhasil di-ARM')
        break
      print(f'Menunggu ARM... ({i+1}/30)')

    if not self.armed:
      raise RuntimeError('Gagal ARM drone setelah 30 kali percobaan.')

    print(f'Mengirim perintah Takeoff ke {tinggi} meter...')
    self.m.mav.command_long_send(
        self.m.target_system,
        self.m.target_component,
        mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        tinggi,
    )

    berhasil_naik = self._tunggu(
        lambda: -self.d >= tinggi * 0.9, 20, 'naik ke ketinggian'
    )
    if berhasil_naik:
      print(f'Sudah di ketinggian {-self.d:.2f} m')
    else:
      print(
          'Peringatan: Drone mungkin belum mencapai ketinggian yang diinginkan.'
      )

    self._update(0.1)
    self.yaw_referensi = self.yaw

  def gerak_relatif_presisi(
      self,
      maju_m=0.0,
      kanan_m=0.0,
      naik_m=0.0,
      kecepatan_maks=1.2,
      toleransi=0.08,
  ):
    """Menggerakkan drone ke titik setpoint koordinat EKF tertentu secara akurat."""
    self._update(0.1)
    yaw_kunci = self.yaw_referensi

    target_n = self.n + (
        maju_m * math.cos(yaw_kunci) - kanan_m * math.sin(yaw_kunci)
    )
    target_e = self.e + (
        maju_m * math.sin(yaw_kunci) + kanan_m * math.cos(yaw_kunci)
    )
    target_d = self.d - naik_m

    print(
        f'Setpoint EKF (Maju:{maju_m}m, Kanan:{kanan_m}m) -> Target'
        f' NED({target_n:.2f}, {target_e:.2f}, {target_d:.2f})'
    )

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
    print(
        f'Tiba di setpoint! Posisi aktual NED({self.n:.2f}, {self.e:.2f},'
        f' {self.d:.2f}) | Error: {jarak_sisa:.3f}m'
    )

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

      self._kirim_kecepatan_body_terkunci_yaw(
          vx_maju, vy_kanan, vz_atas, self.yaw_referensi
      )
      self._update(0.0)

      next_tick += dt
      sisa_tidur = next_tick - time.monotonic()
      if sisa_tidur > 0:
        time.sleep(min(sisa_tidur, max(0.0, t_akhir - time.monotonic())))
      else:
        next_tick = time.monotonic()

    if vx_maju == 0.0 and vy_kanan == 0.0 and vz_atas == 0.0:
      self.rem_aktif()

  def putar_yaw(self, derajat, timeout=20.0, toleransi_deg=3.0):
    """Putar di tempat. derajat > 0 = searah jarum jam (belok kanan). Yaw terkunci ke arah baru."""
    target = _norm_sudut(self.yaw_referensi + math.radians(derajat))
    print(f'Memutar yaw {derajat:+.0f} derajat (target {math.degrees(target):.0f})')
    t_mulai = time.time()
    ok_sejak = None
    while time.time() - t_mulai < timeout:
      self._update(0.0)
      self._kirim_kecepatan_lokal_ned(0.0, 0.0, 0.0, target_yaw=target)
      galat = abs(_norm_sudut(target - self.yaw))
      if galat < math.radians(toleransi_deg):
        if ok_sejak is None:
          ok_sejak = time.time()
        elif time.time() - ok_sejak > 0.5:
          break
      else:
        ok_sejak = None
      time.sleep(1.0 / 30.0)
    else:
      print('Peringatan: yaw belum mencapai target, lanjut.')
    self.yaw_referensi = target
    self.rem_aktif()

  def scan_aruco_sambil_maju(self, vx=0.4, max_detik=10.0, batas_depan=1.2):
    """Maju sambil mencatat marker tanpa berhenti saat marker terlihat."""
    terdeteksi = []
    t_mulai = time.time()
    while time.time() - t_mulai < max_detik:
      if aruco_front_id is not None and aruco_front_id not in terdeteksi:
        terdeteksi.append(aruco_front_id)
        print(f'[SCAN] Marker ID {aruco_front_id} terbaca, lanjut gerak...')

      if jarak_depan < batas_depan:
        print(f'[SCAN] Berhenti karena jarak depan {jarak_depan:.2f}m.')
        self.rem_aktif()
        break

      self.move_at_vel(vx, 0, 0, 0.1)
    return terdeteksi

  def rem_aktif(self):
    """Menghentikan laju drone seketika di tempat."""
    for _ in range(5):
      self._kirim_kecepatan_lokal_ned(
          0.0, 0.0, 0.0, target_yaw=self.yaw_referensi
      )
      time.sleep(0.015)

  def mendarat(self):
    print('Mendarat...')
    self._ganti_mode('LAND')
    self._tunggu(lambda: (not self.armed) or -self.d < 0.15, 60, 'mendarat')


_drone = None


def move_at_vel(vx, vy, vz, detik):
  _drone.move_at_vel(vx, vy, vz, detik)


# ===================================================================
# ====================== FUNGSI BANTU NAVIGASI ======================
# ===================================================================
def _batasi(v, vmax):
  return max(-vmax, min(vmax, v))


def koreksi_tengah():
  """
  Kecepatan samping (kanan > 0) supaya drone ada di TENGAH dua dinding.
  Lidar kiri & kanan simetris, jadi tengah lorong = jarak kiri sama dengan jarak kanan.
  Kalau salah satu sisi tidak ada dinding (mis. di tikungan), tidak ada koreksi.
  """
  if jarak_kiri < BATAS_LORONG and jarak_kanan < BATAS_LORONG:
    galat = (jarak_kanan - jarak_kiri) / 2.0  # > 0: sisi kanan lebih lega
    if abs(galat) < 0.03:
      return 0.0
    return _batasi(KP_TENGAH * galat, BATAS_VY_TENGAH)
  return 0.0


def koreksi_garis(e_ref):
  """Kecepatan samping (kanan > 0) supaya drone (yaw utara) tetap di garis timur e_ref (EKF)."""
  galat = e_ref - _drone.e  # > 0: drone di kiri garis -> geser kanan
  if abs(galat) < 0.02:
    return 0.0
  return _batasi(KP_GARIS * galat, 0.3)


def maju_di_tengah_lorong(v_maks, berhenti_depan, jarak_maks, timeout):
  """
  Maju lurus sambil menjaga posisi tengah lorong. Berhenti kalau:
    - lidar depan <= berhenti_depan (melambat otomatis mendekati dinding), atau
    - jarak tempuh >= jarak_maks (dari EKF), atau
    - timeout.
  Return alasan berhenti.
  """
  _drone._update(0.1)
  n0, e0 = _drone.n, _drone.e
  yaw = _drone.yaw_referensi
  t_mulai = time.time()
  alasan = 'timeout'
  while time.time() - t_mulai < timeout:
    tempuh = (_drone.n - n0) * math.cos(yaw) + (_drone.e - e0) * math.sin(yaw)
    if jarak_depan <= berhenti_depan:
      alasan = f'dinding depan {jarak_depan:.2f}m'
      break
    if tempuh >= jarak_maks:
      alasan = f'jarak tempuh {tempuh:.2f}m'
      break
    vx = min(v_maks, max(0.10, 0.8 * (jarak_depan - berhenti_depan)))
    move_at_vel(vx, koreksi_tengah(), 0, 0.1)
  move_at_vel(0, 0, 0, 1.0)
  return alasan


def pusatkan_di_lingkaran_hijau(timeout=20.0):
  """
  Posisikan drone tepat di atas lingkaran hijau memakai kamera bawah + HSV.
  Gambar kamera bawah: atas = depan drone, kanan = kanan drone.
  """
  t_mulai = time.time()
  stabil = 0
  while time.time() - t_mulai < timeout:
    frame = frame_kamera_bawah
    b = bentuk_terbesar(bentuk_bawah, warna='hijau')
    if frame is None or b is None:
      move_at_vel(0, 0, 0, 0.1)
      stabil = 0
      continue

    tinggi_img, lebar_img = frame.shape[:2]
    cx, cy = b['pusat']
    err_x = (cx - lebar_img / 2.0) / (lebar_img / 2.0)   # + : lingkaran di kanan
    err_y = (tinggi_img / 2.0 - cy) / (tinggi_img / 2.0)  # + : lingkaran di depan

    if abs(err_x) < 0.12 and abs(err_y) < 0.12:
      stabil += 1
      move_at_vel(0, 0, 0, 0.1)
      if stabil >= 10:
        print('[HSV] Drone sudah di tengah lingkaran hijau.')
        return True
    else:
      stabil = 0
      move_at_vel(_batasi(0.35 * err_y, 0.2), _batasi(0.35 * err_x, 0.2), 0, 0.1)
  print('[HSV] Lingkaran hijau tidak terpusatkan (tidak terlihat / timeout).')
  return False


# ===================================================================
# ====================== MISI UTAMA =================================
# ===================================================================
def misi():
  global jarak_depan, jarak_kiri, jarak_kanan, jarak_atas, jarak_bawah

  print(
      '>>> MEMULAI MISI JALUR KIRI (SETPOINT EKF + LIDAR 1D + ARUCO + HSV)...'
  )
  time.sleep(2.0)  # Tunggu stabil setelah takeoff
  _drone._update(0.1)
  # Garis lurus lorong 1 di koordinat EKF (timur): titik akhir Tahap 1 + OFFSET_SUMBU_LORONG
  e_sumbu = _drone.e - JARAK_GESER_AWAL + OFFSET_SUMBU_LORONG

  # ==========================================
  # TAHAP 1: Geser ke KANAN Presisi (Setpoint EKF)
  # ==========================================
  print(
      f'\n[Tahap 1] Bergerak ke kiri tepat sejauh {JARAK_GESER_AWAL}m (EKF'
      ' Setpoint)...'
  )
  _drone.gerak_relatif_presisi(
      maju_m=0.0, kanan_m=-JARAK_GESER_AWAL, naik_m=0.0, kecepatan_maks=1.2
  )
  move_at_vel(0, 0, 0, 1.0)  # Hover stabil 1 detik sebelum masuk lorong

  # ==========================================
  # TAHAP 2: Maju sampai tanjakan oranye terdeteksi lidar depan
  # ==========================================
  print(
      '\n[Tahap 2] Maju mencari rintangan pertama (tanjakan oranye),'
      f' jarak_depan awal: {jarak_depan:.2f}m...'
  )
  _drone._update(0.1)
  n_awal = _drone.n
  ketemu = False
  while _drone.n - n_awal < 12.0:  # tanjakan seharusnya ada < 9 m di depan
    if jarak_depan < JARAK_STOP_TANJAKAN:
      ketemu = True
      break
    vx = 0.4 if jarak_depan < 3.0 else 1.0
    move_at_vel(vx, koreksi_garis(e_sumbu), 0, 0.1)
  move_at_vel(0, 0, 0, 1.0)
  if not ketemu:
    raise RuntimeError('Tanjakan tidak terdeteksi di depan. Cek lidar_front / JARAK_GESER_AWAL.')
  jarak_ke_tanjakan = jarak_depan
  print(f'Tanjakan terdeteksi di depan ({jarak_ke_tanjakan:.2f}m). Berhenti.')

  # ==========================================
  # TAHAP 3: Hindari tanjakan lewat sisi KANAN, lalu kembali ke sumbu lorong
  # ==========================================
  print('\n[Tahap 3] Menghindari tanjakan lewat sisi kanan...')
  _drone._update(0.1)
  e_sebelum_hindar = _drone.e
  while jarak_depan < 3.0 and _drone.e - e_sebelum_hindar < 3.0:  # geser sampai lidar depan tidak kena tanjakan
    _drone.gerak_relatif_presisi(kanan_m=0.25, kecepatan_maks=0.4, toleransi=0.05)
    move_at_vel(0, 0, 0, 0.3)
  if jarak_depan < 3.0:
    raise RuntimeError('Sisi kanan tanjakan tidak lolos dalam 3 m geser. Berhenti.')
  _drone.gerak_relatif_presisi(kanan_m=GESER_LEBIH, kecepatan_maks=0.4, toleransi=0.05)
  _drone._update(0.1)
  geser_total = _drone.e - e_sebelum_hindar
  print(f'Sudah di sisi kanan tanjakan (geser total {geser_total:.2f}m). Maju melewatinya...')

  _drone.gerak_relatif_presisi(
      maju_m=jarak_ke_tanjakan + UJUNG_TANJAKAN, kecepatan_maks=0.7
  )
  move_at_vel(0, 0, 0, 0.5)
  print('Tanjakan sudah terlewati. Kembali ke sumbu lorong...')
  _drone._update(0.1)
  _drone.gerak_relatif_presisi(kanan_m=e_sumbu - _drone.e, kecepatan_maks=0.4, toleransi=0.05)
  move_at_vel(0, 0, 0, 1.0)

  # ==========================================
  # TAHAP 4: Maju sampai dinding lorong 1 terdeteksi di kiri dan kanan
  # ==========================================
  print('\n[Tahap 4] Maju menuju mulut lorong 1...')
  _drone._update(0.1)
  n_awal = _drone.n
  masuk = False
  while _drone.n - n_awal < 12.0:
    if jarak_kiri < BATAS_DEKAT_LORONG and jarak_kanan < BATAS_DEKAT_LORONG:
      masuk = True
      break
    vx = 0.4 if jarak_depan < 2.5 else 0.8
    move_at_vel(vx, koreksi_garis(e_sumbu), 0, 0.1)
  if not masuk:
    raise RuntimeError('Mulut lorong 1 tidak terdeteksi (lidar kiri/kanan tidak melihat dinding).')
  print(f'Masuk lorong 1 (kiri {jarak_kiri:.2f}m, kanan {jarak_kanan:.2f}m).')
  move_at_vel(0.4, koreksi_tengah(), 0, 0.5)

  # ==========================================
  # TAHAP 5: Lorong 1 - maju di tengah, scan ArUco, berhenti dekat dinding ujung
  # ==========================================
  print('\n[Tahap 5] Menyusuri lorong 1 (tengah lorong + scan ArUco)...')
  marker_terbaca = []
  _drone._update(0.1)
  n_awal, e_awal, yaw = _drone.n, _drone.e, _drone.yaw_referensi
  t_mulai = time.time()
  while time.time() - t_mulai < 40.0:
    tempuh = (_drone.n - n_awal) * math.cos(yaw) + (_drone.e - e_awal) * math.sin(yaw)
    if aruco_front_id is not None and aruco_front_id not in marker_terbaca:
      marker_terbaca.append(aruco_front_id)
      print(f'[LORONG 1] Marker ID {aruco_front_id} terbaca (jarak depan {jarak_depan:.2f}m)')
    if jarak_depan <= DEPAN_UJUNG_LORONG1:
      print(f'Ujung lorong 1 ({jarak_depan:.2f}m). Berhenti.')
      break
    if tempuh > 9.0:
      print('Peringatan: sudah 9 m di lorong 1, berhenti.')
      break
    vx = min(0.5, max(0.10, 0.8 * (jarak_depan - DEPAN_UJUNG_LORONG1)))
    move_at_vel(vx, koreksi_tengah(), 0, 0.1)
  move_at_vel(0, 0, 0, 1.5)
  if aruco_front_id is not None and aruco_front_id not in marker_terbaca:
    marker_terbaca.append(aruco_front_id)
  print(f'Marker ArUco di lorong 1: {marker_terbaca if marker_terbaca else "belum terbaca"}')

  # ==========================================
  # TAHAP 6: Belok kanan 90 derajat (masuk lorong 2 ke arah timur)
  # ==========================================
  print('\n[Tahap 6] Belok kanan ke lorong 2...')
  _drone.putar_yaw(90.0)
  move_at_vel(0, 0, 0, 1.0)
  print(f'Setelah belok: depan {jarak_depan:.2f}m, kiri {jarak_kiri:.2f}m, kanan {jarak_kanan:.2f}m')

  # ==========================================
  # TAHAP 7: Lorong 2 - maju di tengah lorong
  # ==========================================
  print('\n[Tahap 7] Menyusuri lorong 2...')
  alasan = maju_di_tengah_lorong(
      v_maks=0.5,
      berhenti_depan=DEPAN_LORONG2,
      jarak_maks=JARAK_LORONG2,
      timeout=40.0,
  )
  print(f'Lorong 2 selesai ({alasan}).')

  # ==========================================
  # TAHAP 8: Pusatkan di lingkaran hijau (HSV kamera bawah), lalu mendarat
  # ==========================================
  print('\n[Tahap 8] Memusatkan drone di lingkaran hijau (HSV)...')
  pusatkan_di_lingkaran_hijau()
  move_at_vel(0, 0, 0, 1.0)

  print('\nRingkasan bentuk berwarna yang terdeteksi (HSV):')
  for sumber, warna, bentuk, lubang in sorted(riwayat_bentuk, key=str):
    print(f'  [{sumber}] {warna} - {bentuk}' + (f' (lubang {lubang})' if lubang else ''))
  print('>>> MISI JALUR KIRI SELESAI. Mendarat.')


# ===================================================================
# ====================== EKSEKUSI UTAMA =============================
# ===================================================================
if __name__ == '__main__':
  ros_thread = threading.Thread(target=jalankan_ros, daemon=True)
  ros_thread.start()
  time.sleep(2.0)

  if MODE_KALIBRASI_HSV:
    print('MODE KALIBRASI HSV: arahkan drone ke objek berwarna di Gazebo. Ctrl+C untuk keluar.')
    try:
      while True:
        time.sleep(1.0)
    except KeyboardInterrupt:
      pass
    raise SystemExit(0)

  try:
    _drone = _Drone(KONEKSI)
    _drone.takeoff(TINGGI)
    misi()
  except BaseException:
    print('\nMISI BERHENTI:')
    traceback.print_exc()
  finally:
    if TAMPILKAN_KAMERA:
      try:
        cv2.destroyAllWindows()
      except cv2.error:
        pass
    if _drone:
      _drone.mendarat()
