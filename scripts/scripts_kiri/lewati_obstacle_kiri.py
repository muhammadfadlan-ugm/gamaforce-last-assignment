#!/usr/bin/env python3
"""
================================================================================
JALUR KIRI - FULL PIPELINE (LEWATI U TERBALIK → BENTUK ABSTRAK → LORONG →
SCAN QR → BELOK → WINDOW OBSTACLES → HOVER)
================================================================================
Pipeline navigasi otonom jalur kiri Gamaforce:
  1. Takeoff & geser ke kiri menuju obstacle kuning
  2. Deteksi + lompati obstacle kuning (U terbalik)
  3. Melewati obstacle merah (bentuk abstrak Bézier)
  4. Masuk lorong lurus + centering LiDAR
  5. Scan QR / ArUco marker di ujung lorong
  6. Belok 90° kanan (hadap Timur)
  7. Tembus rintangan berjendela (orange, biru, hijau, pink)
  8. Hover setelah rintangan terakhir, lanjut pad/ring dan verifikasi visual

Konstanta di bagian atas → mudah disesuaikan.
================================================================================
"""

import sys
import os
import time
import math
import re
from functools import partial
from typing import Optional, Tuple

# Auto venv fallback
try:
    import pymavlink
except ModuleNotFoundError:
    venv_py = os.path.expanduser('~/venv-ardupilot/bin/python3')
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise

# Path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)
for p in (SCRIPTS_DIR, CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np

from core import get_drone, get_sensor
from deteksi.belahketupat import COLOR_PRESETS
from deteksi import (
    QRCodeReader,
    ObstacleKuningDetector,
    ObstacleMerahDetector,
    KamikazeDetector,
)

# ============================================================================
# KONSTANTA – ubah sesuai kebutuhan arena
# ============================================================================

# -- Takeoff & Posisi Awal --
TINGGI_TAKEOFF       = 1.50   # m
MAJU_AWAL_M          = 4.0    # m, maju menuju mulut jalur kiri
GESER_KIRI_M         = 20.4  # m, geser ke kiri (barat) menuju track kiri

# -- Obstacle Kuning (U Terbalik) -- lewati BAWAH, bukan atas
JARAK_STOP_KUNING    = 2.50   # m, berhenti kalau LiDAR depan < ini
KECEPATAN_DEKAT_MS   = 0.45   # m/s, maju pelan mendekat
WAKTU_DEKAT_S        = 15.0   # detik maksimal mendeteksi
TINGGI_BAWAH_KUNING  = 0.60   # m, rendah untuk lewat BAWAH U terbalik
JARAK_LINTAS_KUNING  = 5.50   # m, maju melintasi bawah obstacle
TINGGI_NAIK_SETELAH  = 2.28  # m, naik lagi setelah berhasil lewat bawah

# -- Obstacle Merah (Bentuk Abstrak Bézier) -- lewati DALAM, bukan bawah
JARAK_STOP_MERAH     = 2.80   # m, berhenti kalau LiDAR depan < ini
TINGGI_DALAM_MERAH   = 2.55   # m, terbang di DALAM/MELALUI kurva merah (Z 1.17-4.19)
JARAK_LINTAS_MERAH   = 4.00   # m, maju menembus dalam obstacle merah

# -- Lorong exit & navigasi setelah --
TINGGI_EXIT_LORONG   = 1.00   # m, tinggi pas keluar lorong
JARAK_EXIT_DINDING   = 2.5    # m, threshold dinding lorong setelah belok
KECEPATAN_EXIT_MS    = 1.00   # m/s, kecepatan exit lorong

# -- Lorong --
TINGGI_LORONG        = 0.65   # m, tinggi masuk lorong
KECEPATAN_LORONG     = 0.80   # m/s, kecepatan maju di lorong
KECEPATAN_LORONG_DEKAT = 0.40 # m/s, kecepatan saat dekat ujung
JARAK_REM_DEPAN_LORONG = 0.70 # m, batas pengereman dari sensor depan
LIDAR_MIN_LORONG      = 0.12  # m, batas jangkauan sensor pada model drone
UMUR_MAKS_LIDAR_S    = 0.35  # detik sejak pesan LiDAR depan diterima
WAKTU_LORONG_S       = 90.0   # detik maksimal di lorong, termasuk menunggu data pulih
WAKTU_KONFIRMASI_BUKAAN_S = 5.0  # detik untuk konfirmasi bukaan saat hover

# -- Batas waktu tahap setelah lorong (detik) --
WAKTU_BELOK_S = 12.0
WAKTU_KOREKSI_YAW_S = 5.0
WAKTU_KELUAR_LORONG_S = 45.0
WAKTU_CENTERING_CELAH_S = 16.0
CADANGAN_LINTAS_CELAH_S = 10.0
LAJU_PERKIRAAN_LINTAS_MS = 0.12  # perkiraan gerak aktual untuk batas waktu, bukan perintah motor
WAKTU_TANPA_PROGRES_CELAH_S = 12.0
PROGRES_MIN_CELAH_M = 0.15
WAKTU_KONEKSI_MAVROS_S = 30.0
WAKTU_PAD_DAN_RING_S = 180.0
WAKTU_VERIFIKASI_PER_ARAH_S = 5.0
KP_CENTER           = 0.45   # gain centering LiDAR
VY_MAX              = 0.25   # m/s, koreksi samping maksimal

# -- Belok --
KECEPATAN_PUTAR_RAD  = math.degrees(45)  # rad/s
DURASI_PUTAR_S       = 2.0               # detik (45 deg/s * 2s = 90°)

# Geometri mesh obstacles.dae setelah translasi Track Vision (-27, -5, 0).
# Format: (nama, x muka depan, x tujuan setelah bingkai, y celah, z celah, warna).
WINDOW_OBSTACLES = [
    ("Orange", -12.09, -11.14, 9.175, 3.144, None),
    ("Cyan", -9.36, -7.90, 8.685, 1.220, "biru"),
    ("Hijau", -7.33, -5.88, 9.787, 4.071, "hijau"),
    ("Pink", -3.91, -2.30, 8.052, 2.451, "pink"),
]
WINDOW_SPAWN_Y = -15.0
WINDOW_SPAWN_Z = 0.195
WAKTU_SETPOINT_ORANGE_S = 50.0
WAKTU_SETPOINT_CELAH_S = 45.0
WAKTU_CEK_ULANG_MUNDUR_S = 8.0
KECEPATAN_LINTAS_CELAH_MS = 0.45

# -- Kecepatan umum navigasi (gerak_relatif_presisi) --
KECEPATAN_NAVIGASI   = 1.5    # m/s, kecepatan maks gerak setpoint EKF
KECEPATAN_PRESISI    = 1.0    # m/s, kecepatan untuk gerakan presisi (naik/turun/menembus)

# ============================================================================
# TAHAP 1: TAKEOFF & GESER KE KIRI
# ============================================================================
def tahap_takeoff_ke_kiri(drone) -> Tuple[float, float, float]:
    """Takeoff, lalu geser ke kiri menuju mulut jalur kiri."""
    print("\n" + "=" * 60)
    print(">>> [TAHAP 1] TAKEOFF & GESER KE JALUR KIRI")
    print("=" * 60)

    drone.takeoff_if_needed(TINGGI_TAKEOFF)
    drone.brake(1.0)
    print(f"[*] Drone di udara. Alt: {drone.altitude:.2f} m")

    # Ambil referensi setelah posisi EKF sudah stabil, sebelum translasi pertama.
    # Saat NED berimpit dengan world, spawn berada di utara -15 m dan Z 0,195 m.
    drone.update()
    alt_spawn_ned = WINDOW_SPAWN_Z if abs(drone.pos_x - WINDOW_SPAWN_Y) < 2.0 else 0.0
    asal_ned = (drone.pos_x, drone.pos_y, alt_spawn_ned)
    print(f"[*] Referensi window NED: utara {asal_ned[0]:.2f}, "
          f"timur {asal_ned[1]:.2f}, tinggi spawn {asal_ned[2]:.2f} m.")

    print(f"[*] Geser kiri {GESER_KIRI_M} m, maju {MAJU_AWAL_M} m...")
    drone.gerak_relatif_presisi(
        maju_m=MAJU_AWAL_M,
        kanan_m=-GESER_KIRI_M,  # negatif = kiri
        kecepatan_maks=KECEPATAN_NAVIGASI,
    )
    drone.brake(0.8)
    print(f"[OK] Sampai di mulut jalur kiri. Alt: {drone.altitude:.2f} m")
    return asal_ned

# ============================================================================
# TAHAP 2: LEWATI OBSTACLE KUNING (U TERBALIK)
# ============================================================================
def tahap_lewati_obstacle_kuning(drone, sensor):
    """Dekati obstacle kuning, lalu lewati BAWAH U terbalik, lalu naik lagi."""
    print("\n" + "=" * 60)
    print(">>> [TAHAP 2] LEWATI BAWAH U TERBALIK (KUNING)")
    print("=" * 60)

    # -- Turun ke ketinggian rendah --
    print(f"[*] Turun ke {TINGGI_BAWAH_KUNING} m untuk lewat bawah...")
    delta_turun = drone.altitude - TINGGI_BAWAH_KUNING
    if delta_turun > 0:
        drone.gerak_relatif_presisi(naik_m=-delta_turun, kecepatan_maks=0.6)
        drone.brake(0.8)

    # -- Deteksi & dekati obstacle --
    detector = ObstacleKuningDetector()
    t_start = time.time()

    while time.time() - t_start < WAKTU_DEKAT_S:
        drone.update()
        d_depan = sensor.get_lidar_front()
        frame = sensor.get_frame_front()

        if frame is not None:
            res = detector.detect(frame, draw=False)
            if res.detected:
                print(f"[OK] Kuning terdeteksi kamera! area={res.area:.0f}")

        if d_depan < JARAK_STOP_KUNING:
            print(f"[OK] Kuning dalam jarak {d_depan:.2f} m. Berhenti.")
            break

        drone.move_timed(KECEPATAN_DEKAT_MS, 0.0, 0.0, 0.1)

    drone.brake(1.0)

    # -- Lewat BAWAH obstacle --
    print(f"[*] Maju {JARAK_LINTAS_KUNING} m lewat BAWAH U terbalik (alt {drone.altitude:.2f} m)...")
    drone.gerak_relatif_presisi(maju_m=JARAK_LINTAS_KUNING, kecepatan_maks=0.8)
    drone.brake(0.8)
    print(f"[OK] Berhasil lewat bawah U terbalik!")

    # -- Naik lagi setelah lewat --
    print(f"[*] Naik ke {TINGGI_NAIK_SETELAH} m...")
    delta_naik = TINGGI_NAIK_SETELAH - drone.altitude
    if delta_naik > 0:
        drone.gerak_relatif_presisi(naik_m=delta_naik, kecepatan_maks=0.8)
        drone.brake(0.8)
    print(f"[OK] Altitude sekarang: {drone.altitude:.2f} m")

# ============================================================================
# TAHAP 3: LEWATI OBSTACLE MERAH (BENTUK ABSTRAK / BÉZIER)
# ============================================================================
def tahap_lewati_obstacle_merah(drone, sensor):
    """Deteksi dan lewati DALAM/MELALUI obstacle merah (Bézier Curve)."""
    print("\n" + "=" * 60)
    print(">>> [TAHAP 3] LEWATI DALAM OBSTACLE MERAH (BENTUK ABSTRAK)")
    print("=" * 60)

    # Set altitude untuk menembus DALAM bentuk abstrak (z 1.17-4.19, terbang ~2.8 m)
    print(f"[*] Set altitude {TINGGI_DALAM_MERAH} m (menembus dalam kurva merah)...")
    delta = TINGGI_DALAM_MERAH - drone.altitude
    drone.gerak_relatif_presisi(naik_m=delta, kecepatan_maks=0.7)
    drone.brake(0.8)

    detector = ObstacleMerahDetector()
    t_start = time.time()

    # Maju perlahan sampai dekat obstacle
    while time.time() - t_start < 10.0:
        drone.update()
        d_depan = sensor.get_lidar_front()
        frame = sensor.get_frame_front()

        if frame is not None:
            res = detector.detect(frame, draw=False)
            if res.detected:
                print(f"[OK] Merah terlihat! area={res.area:.0f}")

        if d_depan < JARAK_STOP_MERAH:
            print(f"[OK] Merah dalam jarak {d_depan:.2f} m.")
            break

        drone.move_timed(0.45, 0.0, 0.0, 0.1)

    drone.brake(0.8)

    # Tembus DALAM obstacle (bukan bawah, bukan atas)
    print(f"[*] Maju {JARAK_LINTAS_MERAH} m menembus DALAM merah (alt {drone.altitude:.2f} m)...")
    drone.gerak_relatif_presisi(maju_m=JARAK_LINTAS_MERAH, kecepatan_maks=0.8)
    drone.brake(0.8)
    print(f"[OK] Berhasil menembus dalam obstacle merah!")

    # Turun ke tinggi lorong
    delta_turun = drone.altitude - TINGGI_LORONG
    if delta_turun > 0:
        print(f"[*] Turun ke {TINGGI_LORONG} m...")
        drone.gerak_relatif_presisi(naik_m=-delta_turun, kecepatan_maks=0.7)
        drone.brake(1.0)
    print(f"[OK] Siap masuk lorong. Alt: {drone.altitude:.2f} m")

# ============================================================================
# TAHAP 4: LORONG + SCAN QR
# ============================================================================
def tahap_lorong_scan_qr(drone, sensor, baca_lidar_depan) -> Tuple[bool, Optional[str]]:
    """Maju lurus sambil memindai QR dan rem di depan dinding."""
    print("\n" + "=" * 60)
    print(">>> [TAHAP 4] LORONG LURUS + SCAN QR/ARUCO")
    print("=" * 60)

    qr_reader = QRCodeReader(tolerance_px=30)
    data_qr: Optional[str] = None
    deadline = time.monotonic() + WAKTU_LORONG_S
    scan_berikutnya = time.monotonic()
    lidar_terputus = False
    print(f"[*] Batas waktu susur lorong: {WAKTU_LORONG_S:.0f} detik.")

    def pindai_qr():
        frame = sensor.get_frame_front()
        if frame is not None:
            hasil = qr_reader.detect(frame, draw=False)
            if hasil.detected and hasil.data:
                return hasil.data
        return None

    try:
        while time.monotonic() < deadline:
            if data_qr is None and time.monotonic() >= scan_berikutnya:
                data_qr = pindai_qr()
                scan_berikutnya = time.monotonic() + 0.25

            drone.update()
            d_depan = baca_lidar_depan()

            if d_depan is None or not d_depan >= LIDAR_MIN_LORONG:
                if not lidar_terputus:
                    print("[!] LiDAR depan tidak tersedia. Rem sambil menunggu data pulih.")
                    lidar_terputus = True
                drone.brake(0.1)
                continue

            if lidar_terputus:
                print("[*] Data LiDAR depan pulih.")
                lidar_terputus = False

            if d_depan < JARAK_REM_DEPAN_LORONG:
                print(f"[*] Batas pengereman depan tercapai ({d_depan:.2f} m). Rem.")
                break

            if time.monotonic() >= deadline:
                continue

            # +inf berarti pesan diterima tanpa pantulan dalam jangkauan.
            vx = KECEPATAN_LORONG_DEKAT if d_depan < 2.5 else KECEPATAN_LORONG
            drone.send_velocity_body(vx=vx, vy=0.0, vz_up=0.0, yaw_rate=0.0)
            time.sleep(0.04)
        else:
            print("[!] Waktu susur lorong habis. Rem dan tahan posisi.")
            return False, data_qr
    finally:
        drone.brake(1.0)

    if data_qr is None:
        print("[*] Scan QR ulang saat hover...")
        for _ in range(15):
            drone.update()
            data_qr = pindai_qr()
            if data_qr is not None:
                break
            time.sleep(0.08)

    print(f"[OK] QR: '{data_qr}'" if data_qr else "[!] QR tidak terbaca")
    return True, data_qr

# ============================================================================
# TAHAP 6: BELOK KANAN MENGHADAP TIMUR
# ============================================================================
def tahap_belok_kanan(drone) -> bool:
    """Belok kanan dari arah lorong utara ke arah timur NED."""
    print("\n" + "=" * 60)
    print(">>> [TAHAP 6] BELOK KANAN MENGHADAP TIMUR")
    print("=" * 60)

    drone.update()
    yaw_awal = drone.yaw_deg
    target_yaw = 90.0
    selisih_utara = (yaw_awal + 180.0) % 360.0 - 180.0
    if not abs(selisih_utara) <= 15.0:
        print(f"[!] Heading awal {yaw_awal:.1f}° menyimpang dari arah lorong. Tahan posisi.")
        return False
    print(f"[*] Yaw awal: {yaw_awal:.1f}°, target: {target_yaw:.1f}°")

    from pymavlink import mavutil
    drone.m.mav.command_long_send(
        drone.m.target_system, drone.m.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        target_yaw, # param1: heading absolut timur
        45.0,      # param2: 45 deg/s
        1,         # param3: 1 = CW (kanan), -1 = CCW (kiri)
        0,         # param4: 0 = absolut
        0, 0, 0
    )

    t_start = time.monotonic()
    while time.monotonic() - t_start < WAKTU_BELOK_S:
        drone.update()
        selisih = (drone.yaw_deg - target_yaw + 180.0) % 360.0 - 180.0
        if abs(selisih) <= 5.0 and (time.monotonic() - t_start) > 2.0:
            break
        time.sleep(0.1)

    drone.brake(1.0)
    drone.update()
    selisih_akhir = (drone.yaw_deg - target_yaw + 180.0) % 360.0 - 180.0
    if abs(selisih_akhir) > 5.0:
        print(f"[*] Yaw meleset {selisih_akhir:.1f}°, koreksi tambahan...")
        drone.set_yaw(target_yaw, speed_deg_s=45.0, timeout=WAKTU_KOREKSI_YAW_S)
        drone.brake(0.5)
        drone.update()
    selisih_akhir = (drone.yaw_deg - target_yaw + 180.0) % 360.0 - 180.0
    if not abs(selisih_akhir) <= 5.0:
        print(f"[!] Yaw belum tercapai ({selisih_akhir:.1f}°). Tahan posisi.")
        return False
    print(f"[OK] Yaw akhir: {drone.yaw_deg:.1f}° (target {target_yaw:.1f}°)")
    return True

# ============================================================================
# TAHAP 7: WINDOW OBSTACLES (RINTANGAN BERJENDELA)
# ============================================================================
def deteksi_bukaan_window(frame, warna):
    """Cari kontur bagian dalam dari bingkai berwarna yang tertutup."""
    preset = COLOR_PRESETS[warna]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, preset["lower"], preset["upper"])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kontur, hierarki = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarki is None:
        return None

    tinggi, lebar = frame.shape[:2]
    kandidat = []
    for lubang, relasi in zip(kontur, hierarki[0]):
        induk = relasi[3]
        if induk < 0 or cv2.contourArea(lubang) < 200:
            continue
        bingkai = kontur[induk]
        x, y, w, h = cv2.boundingRect(bingkai)
        if x <= 1 or y <= 1 or x + w >= lebar - 1 or y + h >= tinggi - 1:
            continue
        rasio = cv2.contourArea(lubang) / cv2.contourArea(bingkai)
        if not 0.25 <= rasio <= 0.85:
            continue
        momen = cv2.moments(lubang)
        cx = momen["m10"] / momen["m00"]
        cy = momen["m01"] / momen["m00"]
        jarak_pusat = math.hypot(cx - lebar / 2, cy - tinggi / 2)
        kandidat.append((jarak_pusat, (x, y, w, h), lubang, cx, cy))
    if kandidat:
        _, bbox, lubang, cx, cy = min(kandidat, key=lambda item: item[0])
        return bbox, lubang, cx, cy

    # Pada jarak dekat, ujung atas/bawah berlian bisa terpotong kamera.
    # Cari celah di antara dua sisi berlian yang terlihat pada beberapa baris.
    if warna not in ("biru", "pink"):
        return None
    tengah = lebar // 2
    celah = {}
    for y in range(tinggi // 5, 4 * tinggi // 5, 3):
        kiri = np.flatnonzero(mask[y, :tengah])
        kanan = np.flatnonzero(mask[y, tengah:]) + tengah
        if len(kiri) < 6 or len(kanan) < 6:
            continue
        batas_kiri, batas_kanan = int(kiri[-1]), int(kanan[0])
        if batas_kanan - batas_kiri >= lebar // 5:
            celah[y] = (batas_kiri, batas_kanan)
    if not celah:
        return None

    y = max(celah, key=lambda baris: (celah[baris][1] - celah[baris][0],
                                      -abs(baris - tinggi / 2)))
    sekitar = [celah.get(y + dy) for dy in range(-15, 16, 3)]
    if any(batas is None for batas in sekitar):
        return None
    kiri = max(batas[0] for batas in sekitar)
    kanan = min(batas[1] for batas in sekitar)
    if kanan - kiri < lebar // 5:
        return None
    lubang = np.array([[kiri, y - 15], [kanan, y - 15],
                       [kanan, y + 15], [kiri, y + 15]], dtype=np.int32)
    return cv2.boundingRect(mask), lubang, (kiri + kanan) / 2, y


def siapkan_pandangan_hijau(drone, baca_lidar, baca_frame, muka_e):
    """Utamakan observasi di tempat, lalu mundur hanya jika jalur belakang bebas."""
    deadline = time.monotonic() + WAKTU_SETPOINT_CELAH_S
    tertahan_sejak = None
    log_berikutnya = 0.0
    try:
        while time.monotonic() < deadline:
            drone.update()
            sekarang = time.monotonic()
            jarak_muka = muka_e - drone.pos_y
            if jarak_muka >= 2.15:
                print(f"[OK] Jarak pengamatan Hijau {jarak_muka:.2f} m.")
                return

            pesan = baca_frame()
            if pesan is not None and deteksi_bukaan_window(pesan[0], "hijau") is not None:
                print("[OK] Bukaan Hijau sudah terlihat. Lanjut centering dari sini.")
                return

            belakang = baca_lidar("back")
            arah_aman = abs(drone.yaw_deg - 90.0) <= 5.0
            if belakang is None or not belakang >= 0.8 or not arah_aman:
                if tertahan_sejak is None:
                    tertahan_sejak = sekarang
                    log_berikutnya = sekarang
                if sekarang >= log_berikutnya:
                    print(f"[!] Cek mundur Hijau: LiDAR belakang {belakang!r} m "
                          f"(syarat ≥0,80 m), yaw {drone.yaw_deg:.1f}° "
                          f"(syarat 90° ±5°), jarak muka {jarak_muka:.2f} m. Tahan posisi.")
                    log_berikutnya = sekarang + 2.0
                if sekarang - tertahan_sejak >= WAKTU_CEK_ULANG_MUNDUR_S:
                    print("[*] Mundur tetap tertahan. Coba centering Hijau dari posisi sekarang.")
                    return
                drone.brake(0.12)
                continue

            if tertahan_sejak is not None:
                print("[*] LiDAR belakang dan yaw kembali aman. Lanjut menyesuaikan jarak.")
                tertahan_sejak = None
            drone.send_velocity_body(vx=-0.25, vy=0.0, vz_up=0.0, yaw_rate=0.0)
            time.sleep(0.10)
        print("[*] Batas waktu penyesuaian jarak Hijau habis. Coba centering dari posisi sekarang.")
    finally:
        drone.brake(0.4)


def tahap_window_obstacles(drone, baca_lidar, asal_ned, baca_frame, tampilkan_bbox) -> bool:
    """Keluar ke timur dan lewati setiap rintangan berjendela."""
    print("\n>>> [TAHAP 7] TEMBUS RINTANGAN BERJENDELA")

    asal_n, asal_e, asal_alt = asal_ned
    drone.update()
    utara_world = drone.pos_x - asal_n + WINDOW_SPAWN_Y
    if not 7.0 <= utara_world <= 11.0:
        print(f"[!] Referensi window tidak cocok dengan posisi lorong. "
              f"Utara world terhitung {utara_world:.2f} m dari NED {drone.pos_x:.2f} m.")
        return False

    deadline = time.monotonic() + WAKTU_KELUAR_LORONG_S
    terbuka_sejak = None
    drone.update()
    east_start = drone.pos_y
    while time.monotonic() < deadline:
        drone.update()
        depan = baca_lidar("front")
        kiri = baca_lidar("left")
        kanan = baca_lidar("right")
        if any(jarak is None or jarak < LIDAR_MIN_LORONG for jarak in (depan, kiri, kanan)):
            print("[!] Data LiDAR keluar lorong tidak valid.")
            return False
        if depan < JARAK_REM_DEPAN_LORONG:
            print(f"[!] Halangan di jalur keluar ({depan:.2f} m).")
            return False

        if kiri > 4.0 and kanan > 4.0:
            if terbuka_sejak is None:
                terbuka_sejak = time.monotonic()
            if time.monotonic() - terbuka_sejak >= 0.25 and drone.pos_y > east_start:
                print("[OK] Bukaan area rintangan terkonfirmasi.")
                break
        else:
            terbuka_sejak = None

        vy = 0.0
        if kiri < JARAK_EXIT_DINDING and kanan < JARAK_EXIT_DINDING:
            vy = max(-VY_MAX, min(VY_MAX, (kanan - kiri) * KP_CENTER))
        vx = 0.35 if depan < 1.5 else KECEPATAN_EXIT_MS
        drone.send_velocity_body(vx=vx, vy=vy, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)
    else:
        print("[!] Waktu keluar lorong habis.")
        return False
    drone.brake(0.8)

    for nama, depan_x, keluar_x, celah_y, celah_z, warna in WINDOW_OBSTACLES:
        print(f"\n--- Window {nama} ---")
        target_n = asal_n + celah_y - WINDOW_SPAWN_Y
        target_alt = asal_alt + celah_z - WINDOW_SPAWN_Z
        muka_e = asal_e + depan_x
        tujuan_e = asal_e + keluar_x

        batas_waktu = WAKTU_SETPOINT_ORANGE_S if warna is None else WAKTU_SETPOINT_CELAH_S
        deadline = time.monotonic() + batas_waktu
        log_berikutnya = time.monotonic()
        koreksi_lateral = False
        toleransi = 0.15 if warna is None else 0.12
        while time.monotonic() < deadline:
            drone.update()
            if drone.pos_y >= muka_e - 0.20 or not abs(drone.yaw_deg - 90.0) <= 5.0:
                print(f"[!] Posisi atau yaw tidak aman untuk menyiapkan {nama}.")
                return False
            err_alt = target_alt - drone.altitude
            err_n = target_n - drone.pos_x
            if not math.isfinite(err_alt) or not math.isfinite(err_n):
                print(f"[!] Telemetri posisi {nama} tidak valid.")
                return False
            if abs(err_alt) <= toleransi and abs(err_n) <= toleransi:
                break
            if time.monotonic() >= log_berikutnya:
                print(f"[*] {nama}: tinggi {drone.altitude:.2f}/{target_alt:.2f} m; "
                      f"utara NED {drone.pos_x:.2f}/{target_n:.2f} m.")
                log_berikutnya = time.monotonic() + 5.0

            # Jangan putus koreksi lateral ketika tinggi berfluktuasi sedikit.
            if abs(err_alt) <= 0.22:
                koreksi_lateral = True
            elif abs(err_alt) > 0.35:
                koreksi_lateral = False
            vy = max(-0.30, min(0.30, -0.9 * err_n)) if koreksi_lateral else 0.0
            vz = max(-0.55, min(0.55, 1.2 * err_alt))
            drone.send_velocity_body(vx=0.0, vy=vy, vz_up=vz, yaw_rate=0.0)
            time.sleep(0.04)
        else:
            drone.update()
            print(f"[!] Setpoint {nama} belum tercapai setelah {batas_waktu:.0f} detik. "
                  f"Sisa tinggi {target_alt - drone.altitude:+.2f} m, "
                  f"sisa utara NED {target_n - drone.pos_x:+.2f} m.")
            return False
        drone.brake(0.4)

        if warna == "hijau":
            siapkan_pandangan_hijau(drone, baca_lidar, baca_frame, muka_e)

        if warna is None:
            drone.update()
            depan = baca_lidar("front")
            # Pastikan sinar depan melihat ruang sampai titik setelah bingkai.
            jarak_lintas = tujuan_e - drone.pos_y
            if depan is None or not depan >= max(0.8, jarak_lintas):
                print(f"[!] Bukaan atas Orange belum bebas. "
                      f"LiDAR depan: {depan!r} m; butuh {jarak_lintas:.2f} m.")
                return False
            print("[OK] Jalur melalui bukaan atas Orange bebas menurut LiDAR depan.")
        else:
            deadline = time.monotonic() + WAKTU_CENTERING_CELAH_S
            stabil = 0
            header_terakhir = None
            log_berikutnya = time.monotonic()
            while time.monotonic() < deadline:
                drone.update()
                if not abs(drone.yaw_deg - 90.0) <= 5.0:
                    if time.monotonic() >= log_berikutnya:
                        print(f"[!] Yaw {drone.yaw_deg:.1f}° saat centering {nama}. "
                              "Tahan posisi sambil cek ulang.")
                        log_berikutnya = time.monotonic() + 2.0
                    stabil = 0
                    drone.brake(0.12)
                    continue
                pesan_kamera = baca_frame()
                vy = vz = 0.0
                if pesan_kamera is None:
                    stabil = 0
                else:
                    frame, header = pesan_kamera
                    if header is header_terakhir:
                        time.sleep(0.02)
                        continue
                    header_terakhir = header
                    bukaan = deteksi_bukaan_window(frame, warna)
                    if bukaan is None:
                        stabil = 0
                    else:
                        bbox, lubang, cx, cy = bukaan
                        tampilkan_bbox(frame, header, nama, bbox, lubang)
                        h, w = frame.shape[:2]
                        dx, dy = cx - w / 2, cy - h / 2
                        margin_px = cv2.pointPolygonTest(lubang, (w / 2, h / 2), True)
                        if abs(dx) <= 12 and abs(dy) <= 12 and margin_px >= 12:
                            stabil += 1
                            if stabil >= 4:
                                break
                        else:
                            stabil = 0
                            vy = max(-0.20, min(0.20, 0.6 * dx / (w / 2)))
                            vz = max(-0.20, min(0.20, -0.6 * dy / (h / 2)))
                drone.send_velocity_body(vx=0.0, vy=vy, vz_up=vz, yaw_rate=0.0)
                time.sleep(0.04)
            else:
                print(f"[!] Pusat bukaan {nama} belum terkonfirmasi.")
                return False
            drone.brake(0.3)

        drone.update()
        tahan_n, tahan_alt = drone.pos_x, drone.altitude
        mulai_lintas = time.monotonic()
        awal_e = drone.pos_y
        jarak_lintas = max(0.0, tujuan_e - awal_e)
        deadline = mulai_lintas + jarak_lintas / LAJU_PERKIRAAN_LINTAS_MS + CADANGAN_LINTAS_CELAH_S
        posisi_progres = awal_e
        waktu_progres = log_berikutnya = mulai_lintas
        while time.monotonic() < deadline:
            drone.update()
            sekarang = time.monotonic()
            if not math.isfinite(drone.pos_y):
                print(f"[!] Posisi timur tidak valid saat melintasi {nama}.")
                return False
            if drone.pos_y >= tujuan_e:
                break
            depan = baca_lidar("front")
            if depan is None or not depan >= 0.8 or not abs(drone.yaw_deg - 90.0) <= 5.0:
                print(f"[!] Lintasan {nama} dihentikan. LiDAR depan {depan!r} m, yaw {drone.yaw_deg:.1f}°.")
                return False
            if drone.pos_y - posisi_progres >= PROGRES_MIN_CELAH_M:
                posisi_progres = drone.pos_y
                waktu_progres = sekarang
            if sekarang - waktu_progres >= WAKTU_TANPA_PROGRES_CELAH_S:
                print(f"[!] {nama} tidak maju setidaknya {PROGRES_MIN_CELAH_M:.2f} m "
                      f"dalam {WAKTU_TANPA_PROGRES_CELAH_S:.0f} detik. "
                      f"Timur NED {drone.pos_y:.2f}/{tujuan_e:.2f} m.")
                return False
            if sekarang >= log_berikutnya:
                print(f"[*] Lintas {nama}: {drone.pos_y - awal_e:.2f}/{jarak_lintas:.2f} m "
                      f"ke timur, LiDAR depan {depan:.2f} m.")
                log_berikutnya = sekarang + 5.0
            vy = max(-0.15, min(0.15, 0.8 * (drone.pos_x - tahan_n)))
            vz = max(-0.15, min(0.15, 0.8 * (tahan_alt - drone.altitude)))
            drone.send_velocity_body(vx=KECEPATAN_LINTAS_CELAH_MS, vy=vy, vz_up=vz, yaw_rate=0.0)
            time.sleep(0.04)
        else:
            print(f"[!] Waktu lintas {nama} habis. Timur NED {drone.pos_y:.2f}/{tujuan_e:.2f} m.")
            return False
        drone.brake(0.5)
        print(f"[OK] Window {nama} terlewati.")

    return True

# ============================================================================
# TAHAP 9: PAD HIJAU DAN RING, SKENARIO KIRI
# ============================================================================
def tahap_pad_dan_ring() -> bool:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from scripts_kanan.misi2 import GateNode

    print("\n>>> [TAHAP 9] PAD HIJAU DAN RING (JALUR KIRI)")
    node = GateNode()
    node.scenario = "left"
    node.check_keepout = True
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        deadline = time.monotonic() + WAKTU_KONEKSI_MAVROS_S
        while rclpy.ok() and time.monotonic() < deadline:
            if node.R is not None and node.state is not None and node.state.connected:
                break
            executor.spin_once(timeout_sec=0.1)
        else:
            print("[!] Pose atau koneksi MAVROS belum tersedia.")
            return False

        if not node.state.armed:
            print("[!] Drone tidak siap untuk serah terima dalam penerbangan.")
            return False
        siap, pesan = node.prepare()
        if not siap:
            print(f"[!] Serah terima pad/ring ditolak: {pesan}")
            return False

        deadline = time.monotonic() + WAKTU_PAD_DAN_RING_S
        while rclpy.ok() and time.monotonic() < deadline:
            if node.phase == "DONE":
                return True
            if node.phase == "ABORT":
                return False
            executor.spin_once(timeout_sec=0.05)
        print("[!] Pad dan ring belum selesai. Tahan posisi.")
        return False
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()

# ============================================================================
# TAHAP 10: VERIFIKASI VISUAL BULLSEYE SAAT HOVER
# ============================================================================
def tahap_verifikasi_bullseye(drone, sensor, qr_data, baca_lidar_depan) -> bool:
    print("\n>>> [TAHAP 10] VERIFIKASI VISUAL BULLSEYE")
    marker = re.fullmatch(r"(?:ARUCO_ID_)?([123])", (qr_data or "").strip(), re.IGNORECASE)
    id_qr = int(marker.group(1)) if marker is not None else None
    if id_qr is None:
        print("[!] QR/ArUco tidak memberikan ID bullseye yang valid.")

    detector = KamikazeDetector(target_digit=None)
    semua_id = set()
    for label, heading in (("kanan", 45.0), ("tengah", 0.0), ("kiri", 315.0)):
        drone.set_yaw(heading, speed_deg_s=30.0, timeout=WAKTU_KOREKSI_YAW_S)
        drone.update()
        if not abs((drone.yaw_deg - heading + 180.0) % 360.0 - 180.0) <= 5.0:
            print(f"[!] Arah {label} belum tercapai. Yaw aktual {drone.yaw_deg:.1f}°.")
            continue

        id_per_arah = set()
        deadline = time.monotonic() + WAKTU_VERIFIKASI_PER_ARAH_S
        while time.monotonic() < deadline:
            jarak = baca_lidar_depan()
            if jarak is not None and jarak >= 2.0:
                frame = sensor.get_frame_front()
                if frame is not None:
                    hasil = detector.detect(frame, draw=False)
                    id_per_arah.update(objek.digit for objek in hasil.all_targets
                                       if objek.digit in (1, 2, 3))
            time.sleep(0.1)
        semua_id.update(id_per_arah)
        print(f"[*] Hasil scan {label}: {sorted(id_per_arah)}.")

    print(f"[*] ID bullseye yang terlihat: {sorted(semua_id)}.")
    if id_qr is None:
        return False
    if id_qr in semua_id:
        print(f"[OK] Bullseye #{id_qr} cocok dengan QR. Tahan posisi.")
        return True
    print(f"[!] Bullseye #{id_qr} belum terverifikasi dari posisi hover.")
    return False


# ============================================================================
# MAIN
# ============================================================================
def main():
    from core import sensor as sensor_runtime
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, LaserScan

    print("\n" + "=" * 70)
    print("[GAMAFORCE] JALUR KIRI - HOVER SETELAH RINTANGAN")
    print("=" * 70)
    print("Rute: U terbalik → Bentuk abstrak → Lorong → Belok → Rintangan → Pad/Ring → Verifikasi saat hover")
    print("=" * 70)

    drone = get_drone()
    sensor = get_sensor()

    lidar = {}

    def simpan_lidar(arah, scan):
        jarak = min((r for r in scan.ranges if not math.isnan(r)), default=None)
        lidar[arah] = (jarak, time.monotonic())

    def baca_lidar(arah):
        jarak, diterima = lidar.get(arah, (None, -math.inf))
        if time.monotonic() - diterima <= UMUR_MAKS_LIDAR_S:
            return jarak
        return None

    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    for arah in ("front", "left", "right", "back"):
        sensor_runtime._global_node.create_subscription(
            LaserScan, f"/lidar_{arah}",
            partial(simpan_lidar, arah), qos,
        )

    frame_window = None

    def simpan_frame_window(msg):
        nonlocal frame_window
        frame = sensor._imgmsg_to_bgr(msg)
        frame_window = (frame, msg.header, time.monotonic()) if frame is not None else None

    def baca_frame_window():
        snapshot = frame_window
        if snapshot is not None and time.monotonic() - snapshot[2] <= UMUR_MAKS_LIDAR_S:
            return snapshot[0], snapshot[1]
        return None

    def tampilkan_bbox(frame, header, nama, bbox, lubang=None):
        anotasi = frame.copy()
        x, y, w, h = bbox
        cv2.rectangle(anotasi, (x, y), (x + w, y + h), (0, 220, 255), 2)
        if lubang is not None:
            cv2.drawContours(anotasi, [lubang], -1, (0, 255, 0), 2)
        cv2.putText(anotasi, nama, (x, max(y - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 220, 255), 2)
        bbox_pub.publish(sensor._bgr_to_imgmsg(anotasi, header))

    asal_ned = tahap_takeoff_ke_kiri(drone)
    tahap_lewati_obstacle_kuning(drone, sensor)
    tahap_lewati_obstacle_merah(drone, sensor)
    sampai_ujung, qr_data = tahap_lorong_scan_qr(drone, sensor, lambda: baca_lidar("front"))
    if qr_data:
        print(f"QR terbaca: {qr_data}")
    if not sampai_ujung:
        return

    try:
        # Tunggu bukaan stabil selama hover, tanpa maju ke dinding depan.
        deadline = time.monotonic() + WAKTU_KONFIRMASI_BUKAAN_S
        terbuka_sejak = None
        while time.monotonic() < deadline:
            sekarang = time.monotonic()
            kanan = baca_lidar("right")
            if kanan is not None and kanan > 2.5:
                if terbuka_sejak is None:
                    terbuka_sejak = sekarang
                if sekarang - terbuka_sejak >= 0.24:
                    break
            else:
                terbuka_sejak = None
            time.sleep(0.08)
        else:
            print("[!] Bukaan kanan belum terkonfirmasi. Tahan posisi.")
            return

        if not tahap_belok_kanan(drone):
            return
        node = sensor_runtime._global_node
        bbox_pub = node.create_publisher(Image, "/camera_front/windows_bbox", qos)
        sub_window = node.create_subscription(Image, "/camera_front", simpan_frame_window, qos)
        try:
            if not tahap_window_obstacles(drone, baca_lidar, asal_ned,
                                          baca_frame_window, tampilkan_bbox):
                return
        finally:
            node.destroy_subscription(sub_window)
            node.destroy_publisher(bbox_pub)
    finally:
        drone.brake(1.0)

    print(f">>> [TAHAP 8] HOVER SERAH TERIMA pada altitude {drone.altitude:.2f} m.")
    try:
        if tahap_pad_dan_ring():
            tahap_verifikasi_bullseye(drone, sensor, qr_data, lambda: baca_lidar("front"))
    finally:
        drone.brake(1.0)


if __name__ == '__main__':
    from core import sensor as sensor_runtime

    try:
        main()
    finally:
        # Proses CLI ini memiliki singleton sensor dan thread ROS 2 miliknya.
        if sensor_runtime._global_executor is not None:
            sensor_runtime._global_executor.shutdown()
        if sensor_runtime._global_thread is not None:
            sensor_runtime._global_thread.join()
        if sensor_runtime._global_node is not None:
            sensor_runtime._global_node.destroy_node()
        sensor_runtime.rclpy.try_shutdown()
