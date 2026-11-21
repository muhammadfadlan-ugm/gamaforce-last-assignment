#!/usr/bin/env python3
"""
================================================================================
JALUR KIRI - FULL PIPELINE (LEWATI U TERBALIK → BENTUK ABSTRAK → LORONG →
SCAN QR → BELOK → WINDOW OBSTACLES → BULLSEYE)
================================================================================
Pipeline navigasi otonom jalur kiri Gamaforce:
  1. Takeoff & geser ke kiri menuju obstacle kuning
  2. Deteksi + lompati obstacle kuning (U terbalik)
  3. Melewati obstacle merah (bentuk abstrak Bézier)
  4. Masuk lorong lurus + centering LiDAR
  5. Scan QR / ArUco marker di ujung lorong
  6. Belok 90° kanan (hadap Timur)
  7. Tembus rintangan berjendela (orange, biru, hijau, pink)
  8. Navigasi ke area bullseye + serangan kamikaze

Konstanta di bagian atas → mudah disesuaikan.
================================================================================
"""

import sys
import os
import time
import math
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

from core import get_drone, get_sensor
from deteksi import (
    QRCodeReader,
    ObstacleKuningDetector,
    ObstacleMerahDetector,
    KamikazeDetector,
    BelahKetupatBiruDetector,
    BelahKetupatHijauDetector,
    BelahKetupatPinkDetector,
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
TINGGI_DALAM_MERAH   = 2.730   # m, terbang di DALAM/MELALUI kurva merah (Z 1.17-4.19)
JARAK_LINTAS_MERAH   = 4.00   # m, maju menembus dalam obstacle merah

# -- Lorong exit & navigasi setelah --
TINGGI_EXIT_LORONG   = 1.00   # m, tinggi pas keluar lorong
JARAK_EXIT_DINDING   = 2.5    # m, threshold dinding lorong setelah belok
KECEPATAN_EXIT_MS    = 1.00   # m/s, kecepatan exit lorong

# -- Lorong --
TINGGI_LORONG        = 0.65   # m, tinggi masuk lorong
JARAK_DINDING_LORONG = 1.3   # m, threshold deteksi dinding
KECEPATAN_LORONG     = 0.80   # m/s, kecepatan maju di lorong
KECEPATAN_LORONG_DEKAT = 0.40 # m/s, kecepatan saat dekat ujung
BATAS_UJUNG_LORONG   = 0.15  # m, baru belok kalau LiDAR depan udah < 0.5m (dekat ujung lorong)
BATAS_EXIT_KANAN     = 2.5    # m, lidar kanan > ini = tembok kanan udah habis (bisa belok)
WAKTU_LORONG_S       = 40.0   # detik maksimal di lorong
KP_CENTER           = 0.45   # gain centering LiDAR
VY_MAX              = 0.25   # m/s, koreksi samping maksimal

# -- Belok --
KECEPATAN_PUTAR_RAD  = math.degrees(45)  # rad/s
DURASI_PUTAR_S       = 2.0               # detik (45 deg/s * 2s = 90°)

# -- Window Obstacles (berurutan) --
# Format: (maju_m, naik_m, nama)
WINDOW_OBSTACLES = [
    (3.5,  3.16, "Orange Barrier (celah atas)"),
    (2.5,  1.21, "Belah Ketupat Biru (celah bawah)"),
    (2.8,  4.06, "Belah Ketupat Hijau (celah tinggi)"),
    (3.5,  2.45, "Belah Ketupat Pink (celah tengah)"),
]

# -- Kecepatan umum navigasi (gerak_relatif_presisi) --
KECEPATAN_NAVIGASI   = 1.5    # m/s, kecepatan maks gerak setpoint EKF
KECEPATAN_PRESISI    = 1.0    # m/s, kecepatan untuk gerakan presisi (naik/turun/menembus)

# -- Bullseye / Kamikaze --
MAJU_KE_BULLSEYE_M  = 12.0   # m, maju dari area window menuju bullseye
GESER_KE_BULLSEYE_M  = 3.0    # m, geser ke arah bullseye
TINGGI_BULLSEYE_APP  = 2.5    # m, altitude approach bullseye
KECEPATAN_KAMIKAZE   = 7.0    # m/s, sprint kamikaze
JARAK_IMPACT         = 0.45   # m, LiDAR depan deteksi impact

# ============================================================================
# TAHAP 1: TAKEOFF & GESER KE KIRI
# ============================================================================
def tahap_takeoff_ke_kiri(drone):
    """Takeoff, lalu geser ke kiri menuju mulut jalur kiri."""
    print("\n" + "=" * 60)
    print(">>> [TAHAP 1] TAKEOFF & GESER KE JALUR KIRI")
    print("=" * 60)

    drone.takeoff_if_needed(TINGGI_TAKEOFF)
    drone.brake(1.0)
    print(f"[*] Drone di udara. Alt: {drone.altitude:.2f} m")

    print(f"[*] Geser kiri {GESER_KIRI_M} m, maju {MAJU_AWAL_M} m...")
    drone.gerak_relatif_presisi(
        maju_m=MAJU_AWAL_M,
        kanan_m=-GESER_KIRI_M,  # negatif = kiri
        kecepatan_maks=KECEPATAN_NAVIGASI,
    )
    drone.brake(0.8)
    print(f"[OK] Sampai di mulut jalur kiri. Alt: {drone.altitude:.2f} m")

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
def tahap_lorong_scan_qr(drone, sensor) -> Optional[str]:
    """Susuri lorong dengan centering LiDAR, scan QR/ArUco di ujung."""
    print("\n" + "=" * 60)
    print(">>> [TAHAP 4] LORONG LURUS + SCAN QR/ARUCO")
    print("=" * 60)

    qr_reader = QRCodeReader(tolerance_px=30)
    data_qr: Optional[str] = None
    t_start = time.time()

    sudah_di_lorong = False
    jarak_tempuh_min = 2.7
    pos_awal_n = None; pos_awal_e = None

    while time.time() - t_start < WAKTU_LORONG_S:
        drone.update()
        d_depan = sensor.get_lidar_front()
        d_kiri = sensor.get_lidar_left()
        d_kanan = sensor.get_lidar_right()
        frame = sensor.get_frame_front()

        if frame is not None:
            res = qr_reader.detect(frame, draw=False)
            if res.detected and res.data:
                data_qr = res.data
                print(f"\n[🏆 QR SCAN] '{data_qr}' ({res.code_type}) dx:{res.dx} dy:{res.dy}")

        if not sudah_di_lorong and d_kiri < JARAK_DINDING_LORONG and d_kanan < JARAK_DINDING_LORONG:
            sudah_di_lorong = True
            pos_awal_n = drone.pos_x
            pos_awal_e = drone.pos_y
            print(f"[OK] Masuk lorong! kiri={d_kiri:.2f}m kanan={d_kanan:.2f}m")

        if sudah_di_lorong and pos_awal_n is not None:
            jarak_masuk = math.hypot(drone.pos_x - pos_awal_n, drone.pos_y - pos_awal_e)
            if d_depan <= BATAS_UJUNG_LORONG and d_kanan > BATAS_EXIT_KANAN and jarak_masuk >= jarak_tempuh_min:
                print(f"[OK] Ujung lorong (depan {d_depan:.2f}m, kanan {d_kanan:.2f}m). Rem.")
                break

        vy = 0.0
        if d_kiri > 0.03 and d_kanan > 0.03 and d_kiri < JARAK_DINDING_LORONG and d_kanan < JARAK_DINDING_LORONG:
            err = d_kiri - d_kanan
            vy = float(math.copysign(min(VY_MAX, abs(err * KP_CENTER)), -err))

        vx = KECEPATAN_LORONG_DEKAT if d_depan < 2.5 else KECEPATAN_LORONG
        drone.send_velocity_body(vx=vx, vy=vy, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)

    drone.brake(1.0)

    # Final scan saat hover kalau belum kebaca
    if data_qr is None:
        print("[*] Scan ulang QR saat hover...")
        for _ in range(15):
            drone.update()
            f = sensor.get_frame_front()
            if f is not None:
                r = qr_reader.detect(f, draw=False)
                if r.detected and r.data:
                    data_qr = r.data
                    print(f"[🏆 QR SCAN (hover)] '{data_qr}'")
                    break
            time.sleep(0.08)

    print(f"[OK] QR: '{data_qr}'" if data_qr else "[!] QR tidak terbaca")
    return data_qr

# ============================================================================
# TAHAP 5: BELOK 90° KANAN
# ============================================================================
def tahap_belok_kanan(drone):
    """Putar drone 90° ke kanan relatif dari heading saat ini."""
    print("\n" + "=" * 60)
    print(">>> [TAHAP 5] BELOK 90° KANAN (RELATIF)")
    print("=" * 60)

    drone.update()
    yaw_awal = drone.yaw_deg
    target_yaw = (yaw_awal + 90.0) % 360.0
    print(f"[*] Yaw awal: {yaw_awal:.1f}°, target: {target_yaw:.1f}°")

    from pymavlink import mavutil
    drone.m.mav.command_long_send(
        drone.m.target_system, drone.m.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        90.0,      # param1: 90° belok
        45.0,      # param2: 45 deg/s
        1,         # param3: 1 = CW (kanan), -1 = CCW (kiri)
        1,         # param4: 1 = relatif
        0, 0, 0
    )

    t_start = time.time()
    while time.time() - t_start < 8.0:
        drone.update()
        selisih = (drone.yaw_deg - target_yaw + 180.0) % 360.0 - 180.0
        if abs(selisih) <= 5.0 and (time.time() - t_start) > 2.0:
            break
        time.sleep(0.1)

    drone.brake(1.0)
    drone.update()
    selisih_akhir = (drone.yaw_deg - target_yaw + 180.0) % 360.0 - 180.0
    if abs(selisih_akhir) > 5.0:
        print(f"[*] Yaw meleset {selisih_akhir:.1f}°, koreksi tambahan...")
        drone.set_yaw(target_yaw, speed_deg_s=45.0, timeout=3.0)
        drone.brake(0.5)
        drone.update()
    print(f"[OK] Yaw akhir: {drone.yaw_deg:.1f}° (target {target_yaw:.1f}°)")

# ============================================================================
# TAHAP 6: WINDOW OBSTACLES (RINTANGAN BERJENDELA)
# ============================================================================
def tahap_window_obstacles(drone, sensor):
    """Tembus rintangan berjendela berurutan dengan exit lorong yg lebih akurat."""
    print("\n" + "=" * 60)
    print(">>> [TAHAP 6] TEMBUS RINTANGAN BERJENDELA")
    print("=" * 60)

    # -- Keluar lorong (maju arah Timur/X+) --
    print("[*] Keluar lorong menuju area rintangan...")
    berhasil_keluar = False
    for _ in range(3):  # maksimal 3x percobaan
        t_start = time.time()
        while time.time() - t_start < 15.0:
            drone.update()
            d_depan = sensor.get_lidar_front()
            d_kiri = sensor.get_lidar_left()
            d_kanan = sensor.get_lidar_right()

            # Guard LiDAR depan: kurangi vx kalau dekat
            vx = KECEPATAN_EXIT_MS
            if d_depan < 1.5:
                vx = 0.35
                print(f"    [!] Depan {d_depan:.2f}m, pelan...")

            # Centering
            vy = 0.0
            if d_kiri < JARAK_EXIT_DINDING and d_kanan < JARAK_EXIT_DINDING:
                err = d_kiri - d_kanan
                vy = float(math.copysign(min(VY_MAX, abs(err * KP_CENTER)), -err))
            else:
                if d_kiri < JARAK_EXIT_DINDING:
                    vy = min(VY_MAX, (JARAK_EXIT_DINDING - d_kiri) * 0.3)
                elif d_kanan < JARAK_EXIT_DINDING:
                    vy = -min(VY_MAX, (JARAK_EXIT_DINDING - d_kanan) * 0.3)

            drone.send_velocity_body(vx=vx, vy=vy, vz_up=0.0, yaw_rate=0.0)
            time.sleep(0.04)

            if d_kiri > 4.0 and d_kanan > 4.0:
                berhasil_keluar = True
                print(f"[OK] Keluar lorong (kiri {d_kiri:.1f}m, kanan {d_kanan:.1f}m)")
                break

        if berhasil_keluar:
            break
        print("    [!] Exit timeout, coba ulang...")
        drone.brake(0.5)

    if not berhasil_keluar:
        print("    [!] Gagal keluar lorong setelah 3x, lanjut dengan resiko")
    drone.brake(0.8)

    # -- Tembus satu per satu --
    detektor_warna = {
        "Orange Barrier": (ObstacleKuningDetector, "kuning"),
        "Belah Ketupat Biru": (BelahKetupatBiruDetector, "biru"),
        "Belah Ketupat Hijau": (BelahKetupatHijauDetector, "hijau"),
        "Belah Ketupat Pink": (BelahKetupatPinkDetector, "pink"),
    }

    for maju, naik, nama in WINDOW_OBSTACLES:
        print(f"\n--- {nama} ---")

        # LANGKAH C1: Set altitude kasar
        delta_z = naik - drone.altitude
        drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=0.0, naik_m=delta_z,
                                     kecepatan_maks=KECEPATAN_PRESISI)
        drone.brake(0.5)
        print(f"    Altitude kasar: {drone.altitude:.2f} m")

        # LANGKAH C2: Bounding box detection + centering (vx=0)
        det_cls = detektor_warna.get(nama, (None, None))[0]
        centered = False
        if det_cls is not None:
            det = det_cls()
            t_center = time.time()
            while time.time() - t_center < 8.0:
                drone.update()
                frame = sensor.get_frame_front()
                if frame is not None:
                    res = det.detect(frame, draw=False)
                    if res.detected:
                        vy = float(math.copysign(min(0.20, abs(res.norm_dx * 0.3)), res.norm_dx))
                        vz = float(math.copysign(min(0.20, abs(res.norm_dy * 0.3)), -res.norm_dy))
                        drone.send_velocity_body(vx=0.0, vy=vy, vz_up=vz, yaw_rate=0.0)
                        print(f"    [centering] dx:{res.dx} dy:{res.dy} norm_dx:{res.norm_dx:.2f} norm_dy:{res.norm_dy:.2f}")
                        if res.is_centered:
                            print(f"    [OK] Bounding box center! dx:{res.dx} dy:{res.dy}")
                            centered = True
                            drone.brake(0.3)
                            break
                        time.sleep(0.04)
                        continue
                drone.send_velocity_body(vx=0.0, vy=0.0, vz_up=0.0, yaw_rate=0.0)
                time.sleep(0.04)
            if not centered:
                print("    [!] Center timeout 8s, lanjut pakai posisi altitude terakhir")
                drone.brake(0.3)

        # LANGKAH C3: MAJU MENEMBUS — berbasis jarak tempuh aktual
        drone.update()
        pos_n = drone.pos_x
        pos_e = drone.pos_y
        print(f"    Maju menembus target {maju} m...")
        t_maju = time.time()
        timeout_maju = (maju / 0.5) + 3.0
        jarak_tempuh = 0.0
        while time.time() - t_maju < timeout_maju:
            drone.update()
            d_depan = sensor.get_lidar_front()
            jarak_tempuh = math.hypot(drone.pos_x - pos_n, drone.pos_y - pos_e)

            if d_depan < 0.8:
                print(f"    [!] LiDAR depan {d_depan:.2f}m terlalu dekat, berhenti.")
                break
            if jarak_tempuh >= maju:
                print(f"    [OK] Jarak tempuh {jarak_tempuh:.2f}m >= target {maju}m.")
                break

            drone.send_velocity_body(vx=0.5, vy=0.0, vz_up=0.0, yaw_rate=0.0)
            time.sleep(0.04)
        drone.brake(0.8)
        print(f"    [OK] Alt akhir: {drone.altitude:.2f} m, jarak tempuh: {jarak_tempuh:.2f} m")

    print(f"[OK] Semua rintangan tertembus. Alt: {drone.altitude:.2f} m")

# ============================================================================
# TAHAP 7: NAVIGASI KE BULLSEYE
# ============================================================================
def tahap_ke_bullseye(drone, sensor, qr_data: Optional[str]):
    """Navigasi ke bullseye + kamikaze strike berdasarkan QR."""
    print("\n" + "=" * 60)
    print(">>> [TAHAP 7] NAVIGASI + KAMIKAZE KE BULLSEYE")
    print("=" * 60)

    # Parse target digit dari QR (misal: "ARUCO_ID_2" → digit 2)
    target_digit = 2  # default
    if qr_data:
        import re
        match = re.search(r'(\d+)', str(qr_data))
        if match:
            target_digit = int(match.group(1))
            print(f"[*] Target bullseye dari QR: angka #{target_digit}")
        else:
            print(f"[*] QR: '{qr_data}' (gak bisa parse digit, pakai default {target_digit})")
    else:
        print(f"[*] QR gak terbaca, target default: angka #{target_digit}")

    # -- Navigasi ke area bullseye --
    # Set tinggi approach
    delta_z = TINGGI_BULLSEYE_APP - drone.altitude
    drone.gerak_relatif_presisi(naik_m=delta_z, kecepatan_maks=KECEPATAN_PRESISI)
    drone.brake(0.5)

    print(f"[*] Terbang menuju area bullseye ({MAJU_KE_BULLSEYE_M}m maju)...")
    drone.gerak_relatif_presisi(
        maju_m=MAJU_KE_BULLSEYE_M,
        kanan_m=GESER_KE_BULLSEYE_M,
        naik_m=0.0,
        kecepatan_maks=KECEPATAN_NAVIGASI,
    )
    drone.brake(0.5)
    print(f"[OK] Drone di area bullseye. Alt: {drone.altitude:.2f} m")

    # -- Kamikaze strike --
    print(f"\n{'=' * 60}")
    print(f">>> [KAMIKAZE] BURU TARGET #{target_digit} + SPRINT!")
    print(f"{'=' * 60}")

    # Aktifkan kamikaze detector
    kamikaze = KamikazeDetector(target_digit=target_digit)

    # Maju pelan sambil cari target
    t_start = time.time()
    found_target = False
    while time.time() - t_start < 15.0 and not found_target:
        drone.update()
        d_depan = sensor.get_lidar_front()
        frame = sensor.get_frame_front()

        if frame is not None:
            res = kamikaze.detect(frame, draw=False)
            if res.locked_target and res.locked_target.digit == target_digit:
                print(f"[OK] Target #{target_digit} terkunci! cx={res.locked_target.cx} cy={res.locked_target.cy}")
                found_target = True
                break

        if d_depan < 2.0:
            print(f"[OK] LiDAR depan {d_depan:.2f}m — target dekat, gas kamikaze!")
            found_target = True
            break

        drone.send_velocity_body(vx=0.3, vy=0.0, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)

    # Sprint kamikaze
    print(f"🚀🚀 SPRINT KAMIKAZE {KECEPATAN_KAMIKAZE} m/s! 🚀🚀")
    t_start = time.time()
    while time.time() - t_start < 5.0:
        drone.update()
        d_depan = sensor.get_lidar_front()

        if d_depan <= JARAK_IMPACT:
            print(f"💥 IMPACT! LiDAR depan {d_depan:.2f}m 💥")
            break

        drone.send_velocity_body(vx=KECEPATAN_KAMIKAZE, vy=0.0, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.02)

    drone.brake(2.0)
    print(f"[OK] Kamikaze selesai. Alt: {drone.altitude:.2f} m")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("🚀 [GAMAFORCE] JALUR KIRI - FULL PIPELINE 🚀")
    print("=" * 70)
    print("Target: U terbalik → Bentuk abstrak → Lorong → QR → Belok")
    print("        → Window obstacles → Bullseye")
    print("=" * 70)

    drone = get_drone()
    sensor = get_sensor()

    # Eksekusi berurutan
    tahap_takeoff_ke_kiri(drone)
    tahap_lewati_obstacle_kuning(drone, sensor)
    tahap_lewati_obstacle_merah(drone, sensor)
    qr_data = tahap_lorong_scan_qr(drone, sensor)
    tahap_belok_kanan(drone)
    tahap_window_obstacles(drone, sensor)
    tahap_ke_bullseye(drone, sensor, qr_data)

    print("\n" + "=" * 70)
    print("🏆🏆🏆 [MISI SELESAI] JALUR KIRI SUKSES! 🏆🏆🏆")
    if qr_data:
        print(f"📌 QR terbaca: {qr_data}")
    print(f"📌 Altitude akhir: {drone.altitude:.2f} m")
    print("=" * 70)

    drone.brake(1.0)


if __name__ == '__main__':
    main()