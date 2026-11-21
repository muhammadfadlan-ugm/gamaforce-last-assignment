#!/usr/bin/env python3
"""
================================================================================
GAMAFORCE - NAVIGASI OTONOM JALUR KIRI (FULL MISSION PIPELINE)
================================================================================
Misi Navigasi Arena Kiri:
1. Takeoff & Bergerak ke Kiri:
   - Takeoff ke ketinggian 1.5 m.
   - Bergerak ke kiri (arah Barat / -X) menuju koridor rintangan kuning (X ≈ -20.6).
   - Mendeteksi obstacle kuning berbentuk U terbalik (Cube_015) via kamera depan & LiDAR.
2. Melewati di ATAS Rintangan Kuning:
   - Naik (climb) ke ketinggian aman Z ≈ 3.8 - 4.0 m untuk melompati rintangan kuning (Z_max ≈ 2.11 m).
   - Terbang maju melintasi rintangan kuning hingga mencapai Y ≈ -5.0 m.
   - Turun kembali ke ketinggian jelajah Z ≈ 2.0 m.
3. Melewati Obstacle Merah Sembarang (BézierCurve):
   - Bergerak maju menuju Y ≈ -3.0 m.
   - Mendeteksi obstacle merah (BézierCurve) menggunakan ObstacleMerahDetector.
   - Naik melintasi obstacle merah pada Z ≈ 4.4 m (Z_max obstacle = 4.19 m) hingga Y ≈ 0.5 m.
   - Turun ke ketinggian lorong Z ≈ 0.85 m.
4. Menyusuri Lorong Lurus & Membaca QR / ArUco:
   - Terbang menyusuri lorong lurus (Y = 1.5 hingga 9.0 m) pada Z ≈ 0.85 m.
   - Centering otomatis antara dinding kiri & kanan menggunakan LiDAR (lidar_left & lidar_right).
   - Kamera depan mendeteksi & mendekode ArUco/QR marker di ujung lorong menggunakan QRCodeReader.
   - Melakukan pengereman aktif di dekat ujung lorong (Y ≈ 8.8 m).
5. Menghadap ke Kanan (Putar Yaw ke Arah Timur / +X):
   - Memutar hadap drone 90° ke kanan menghadap Timur (Yaw 0° / +X).
   - Pengereman aktif untuk stabilisasi heading.
6. Bergerak Lurus & Menembus Setiap Rintangan Berjendela (Celah Kosong):
   - Bergerak lurus ke arah Timur (+X) melalui lorong keluar menuju ruang rintangan.
   - Secara berurutan menyelaraskan posisi & ketinggian untuk menembus celah kosong:
     * Rintangan 1 (Cube - Orange Barrier, X ≈ -11.84 m): Celah atas Z ≈ 3.16 m, Y ≈ 9.16 m.
     * Rintangan 2 (Cube_007 - Belah Ketupat Biru, X ≈ -8.86 m): Celah tengah Z ≈ 1.21 m, Y ≈ 8.69 m.
     * Rintangan 3 (Cube_008 - Belah Ketupat Hijau, X ≈ -6.83 m): Celah tinggi Z ≈ 4.06 m, Y ≈ 9.78 m.
     * Rintangan 4 (Cube_009 - Belah Ketupat Pink, X ≈ -3.41 m): Celah tengah Z ≈ 2.45 m, Y ≈ 8.04 m.
   - Keluar ke ruang terbuka (X ≈ -1.5 .. 0.0 m, Y ≈ 10.0 .. 12.0 m) sebelum Pad Hijau.
   - Opsi: Melanjutkan handoff otomatis ke Misi 2 (Pad Hijau & Ring scenario='left').
================================================================================
"""

import sys
import os
import time
import math
import argparse
from typing import Optional, Tuple

# Otomatis gunakan venv ArduPilot jika modul pymavlink belum terinstal di system
try:
    import pymavlink
except ModuleNotFoundError:
    venv_py = os.path.expanduser('~/venv-ardupilot/bin/python3')
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise

# Pastikan path modul scripts terdeteksi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)
for p in (SCRIPTS_DIR, CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from core import DroneController, SensorListener, get_drone, get_sensor
from deteksi import (
    QRCodeReader,
    BelahKetupatDetector,
    ObstacleMerahDetector,
    ObstacleKuningDetector,
)

# Impor Misi 2 untuk opsi handoff pipeline penuh
try:
    from scripts_kanan import misi2
except ImportError:
    misi2 = None


# ==============================================================================
# TAHAP 1: TAKEOFF & BERGERAK KE KIRI MENUJU OBSTACLE KUNING
# ==============================================================================
def tahap_1_takeoff_dan_menuju_kuning(
    drone: DroneController,
    sensor: SensorListener
):
    print("\n" + "=" * 65)
    print(">>> [TAHAP 1] TAKEOFF & BERGERAK KE KIRI MENUJU OBSTACLE KUNING")
    print("=" * 65)

    # 1. Takeoff ke ketinggian 1.5 m jika drone masih di permukaan tanah
    drone.takeoff_if_needed(1.50)
    drone.brake(1.0)
    print(f"[*] Drone stabil di udara. Altitude: {drone.altitude:.2f} m | Yaw: {drone.yaw_deg:.1f}°")

    # 2. Bergerak ke kiri (-20.6 m) dan sedikit maju (+4.0 m) menuju mulut lintasan kiri
    # Pada saat spawn: Drone menghadap Utara (+Y), kiri = Barat (-X).
    print("[*] Meluncur ke arah KIRI (-20.6 m) menuju track kiri...")
    drone.gerak_relatif_presisi(
        maju_m=4.0,
        kanan_m=-20.6,
        naik_m=0.0,
        kecepatan_maks=1.2,
        toleransi=0.20
    )
    drone.brake(0.8)
    print(f"[OK] Tiba di koordinat track kiri. Altitude: {drone.altitude:.2f} m")

    # 3. Deteksi Obstacle Kuning (U Terbalik) di depan
    print("[*] Memindai obstacle kuning di depan menggunakan Kamera Depan & LiDAR...")
    detector_kuning = ObstacleKuningDetector()
    t_start = time.time()
    kuning_ditemukan = False

    while time.time() - t_start < 12.0:
        drone.update()
        d_depan = sensor.get_lidar_front()
        frame_front = sensor.get_frame_front()

        if frame_front is not None:
            res = detector_kuning.detect(frame_front, draw=False)
            if res.detected:
                print(f"[OK] Obstacle Kuning terdeteksi kamera! Area={res.area:.0f} px, dx={res.dx}, dy={res.dy}")
                kuning_ditemukan = True

        if d_depan < 2.50 or (kuning_ditemukan and d_depan < 3.20):
            print(f"[OK] Obstacle Kuning dalam jarak aman ({d_depan:.2f} m). Berhenti & hover.")
            drone.brake(1.0)
            break

        # Maju perlahan mendekati obstacle kuning
        drone.move_timed(0.40, 0.0, 0.0, 0.1)

    drone.brake(1.0)
    print(f"[SELESAI TAHAP 1] Drone siap melompati obstacle kuning. Altitude: {drone.altitude:.2f} m")


# ==============================================================================
# TAHAP 2: LEWAT DI ATAS RINTANGAN KUNING (U TERBALIK)
# ==============================================================================
def tahap_2_lewati_atas_kuning(
    drone: DroneController,
    sensor: SensorListener
):
    print("\n" + "=" * 65)
    print(">>> [TAHAP 2] NAIK & TERBANG MELINTASI ATAS OBSTACLE KUNING")
    print("=" * 65)

    # Obstacle kuning (Cube_015) memiliki Z_max ≈ 2.11 m.
    # Target ketinggian aman: Z ≈ 3.80 m (clearance > 1.6 m).
    alt_sekarang = drone.altitude
    delta_z = max(0.0, 3.85 - alt_sekarang)
    print(f"[*] Naik ke ketinggian Z ≈ 3.85 m (delta naik = {delta_z:.2f} m)...")
    drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=0.0, naik_m=delta_z, kecepatan_maks=0.8, toleransi=0.15)
    drone.brake(0.8)
    print(f"[OK] Ketinggian aman tercapai: {drone.altitude:.2f} m.")

    # Terbang maju melintasi obstacle kuning sejauh 5.5 m
    print("[*] Terbang maju melintasi bagian atas obstacle kuning...")
    drone.gerak_relatif_presisi(maju_m=5.5, kanan_m=0.0, naik_m=0.0, kecepatan_maks=1.0, toleransi=0.20)
    drone.brake(0.8)
    print(f"[OK] Rintangan kuning terlewati! Altitude: {drone.altitude:.2f} m")

    # Turun kembali ke ketinggian jelajah (Z ≈ 2.0 m)
    turun_z = max(0.0, drone.altitude - 2.00)
    print(f"[*] Turun kembali ke ketinggian jelajah (turun {turun_z:.2f} m)...")
    drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=0.0, naik_m=-turun_z, kecepatan_maks=0.8, toleransi=0.15)
    drone.brake(0.8)
    print(f"[SELESAI TAHAP 2] Ketinggian jelajah: {drone.altitude:.2f} m")


# ==============================================================================
# TAHAP 3: MELEWATI OBSTACLE MERAH SEMBARANG (BEZIER CURVE)
# ==============================================================================
def tahap_3_lewati_obstacle_merah(
    drone: DroneController,
    sensor: SensorListener
):
    print("\n" + "=" * 65)
    print(">>> [TAHAP 3] DETEKSI & MELEWATI OBSTACLE MERAH (BEZIER CURVE)")
    print("=" * 65)

    # Maju perlahan menuju area obstacle merah (sekitar Y ≈ -3.0 m)
    print("[*] Maju mendekati area obstacle merah...")
    detector_merah = ObstacleMerahDetector()
    t_start = time.time()
    merah_terdeteksi = False

    while time.time() - t_start < 10.0:
        drone.update()
        frame_front = sensor.get_frame_front()
        d_depan = sensor.get_lidar_front()

        if frame_front is not None:
            res = detector_merah.detect(frame_front, draw=False)
            if res.detected:
                print(f"[OK] Obstacle Merah terdeteksi kamera! Area={res.area:.0f} px, dx={res.dx}, dy={res.dy}")
                merah_terdeteksi = True

        if d_depan < 2.80 or (merah_terdeteksi and d_depan < 3.20):
            print(f"[OK] Obstacle Merah teridentifikasi pada jarak {d_depan:.2f} m.")
            drone.brake(0.8)
            break

        drone.move_timed(0.45, 0.0, 0.0, 0.1)

    drone.brake(0.8)

    # Obstacle BézierCurve berada pada Y ≈ -1.39 m dengan Z_max ≈ 4.19 m.
    # Naik ke ketinggian aman Z ≈ 4.45 m untuk melintasinya secara bersih dan aman.
    alt_target = 4.45
    delta_naik = max(0.0, alt_target - drone.altitude)
    print(f"[*] Melompati obstacle merah: naik ke Z ≈ {alt_target:.2f} m (delta: {delta_naik:.2f} m)...")
    drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=0.0, naik_m=delta_naik, kecepatan_maks=0.8, toleransi=0.15)
    drone.brake(0.6)

    # Terbang maju melintasi obstacle merah sejauh 4.0 m (menuju Y ≈ 0.5 m, di depan lorong)
    print("[*] Maju melintasi bagian atas obstacle merah...")
    drone.gerak_relatif_presisi(maju_m=4.0, kanan_m=0.0, naik_m=0.0, kecepatan_maks=1.0, toleransi=0.20)
    drone.brake(0.8)
    print(f"[OK] Obstacle merah terlewati! Drone berada di depan mulut lorong.")

    # Turun ke ketinggian mulut lorong (Z ≈ 0.85 m agar sejajar dengan dinding lorong)
    alt_lorong = 0.85
    delta_turun = max(0.0, drone.altitude - alt_lorong)
    print(f"[*] Turun ke ketinggian masuk lorong: Z ≈ {alt_lorong:.2f} m (delta turun: {delta_turun:.2f} m)...")
    drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=0.0, naik_m=-delta_turun, kecepatan_maks=0.7, toleransi=0.12)
    drone.brake(1.0)
    print(f"[SELESAI TAHAP 3] Drone siap memasuki lorong lurus. Altitude: {drone.altitude:.2f} m")


# ==============================================================================
# TAHAP 4: MENYUSURI LORONG LURUS & MEMBACA QR / ARUCO MARKER
# ==============================================================================
def tahap_4_susuri_lorong_dan_baca_qr(
    drone: DroneController,
    sensor: SensorListener
) -> Optional[str]:
    print("\n" + "=" * 65)
    print(">>> [TAHAP 4] MENYUSURI LORONG LURUS & MEMBACA QR / ARUCO MARKER")
    print("=" * 65)

    qr_reader = QRCodeReader(tolerance_px=30)
    data_qr_terbaca: Optional[str] = None
    waktu_mulai = time.time()
    durasi_maksimal = 25.0

    print("[*] Maju menyusuri lorong dengan kendali centering LiDAR & scan QR kamera depan...")

    while time.time() - waktu_mulai < durasi_maksimal:
        drone.update()
        d_depan = sensor.get_lidar_front()
        d_kiri = sensor.get_lidar_left()
        d_kanan = sensor.get_lidar_right()
        frame_front = sensor.get_frame_front()

        # 1. Pindai QR / ArUco Marker pada frame kamera depan
        if frame_front is not None:
            res_qr = qr_reader.detect(frame_front, draw=False)
            if res_qr.detected and res_qr.data:
                data_qr_terbaca = res_qr.data
                print(f"\n[🏆 SUKSES QR] MARKER TERBACA: '{data_qr_terbaca}' ({res_qr.code_type}) | Offset dx:{res_qr.dx}, dy:{res_qr.dy}")

        # 2. Pengecekan ujung lorong (dinding penutup marker slot)
        if d_depan <= 1.15:
            print(f"\n[OK] Ujung lorong terdeteksi (LiDAR depan: {d_depan:.2f} m)! Rem aktif.")
            drone.brake(1.0)
            break

        # 3. Kendali Centering Lateral LiDAR dalam lorong
        # Dinding kiri & kanan terdeteksi jika jarak < 1.8 m
        vy_cmd = 0.0
        if d_kiri < 1.8 and d_kanan < 1.8:
            error_posisi = d_kiri - d_kanan  # Jika d_kiri > d_kanan, drone terlalu ke kanan -> geser kiri (vy < 0)
            # vy > 0 = kanan, vy < 0 = kiri
            vy_cmd = float(math.copysign(min(0.25, abs(error_posisi * 0.45)), -error_posisi))

        # Kecepatan maju proporsional terhadap sisa jarak depan
        vx_cmd = 0.35 if d_depan < 2.5 else 0.55

        # Kirim perintah kecepatan body
        drone.send_velocity_body(vx=vx_cmd, vy=vy_cmd, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)

    drone.brake(1.0)

    # Jika belum sempat terbaca saat terbang, lakukan pemindaian final saat hover
    if data_qr_terbaca is None:
        print("[*] Melakukan pemindaian QR statis saat hover di ujung lorong...")
        for _ in range(15):
            drone.update()
            f = sensor.get_frame_front()
            if f is not None:
                r = qr_reader.detect(f, draw=False)
                if r.detected and r.data:
                    data_qr_terbaca = r.data
                    print(f"[🏆 SUKSES QR] MARKER TERBACA (Static Hover): '{data_qr_terbaca}' ({r.code_type})")
                    break
            time.sleep(0.08)

    print(f"[SELESAI TAHAP 4] Hasil Scan: '{data_qr_terbaca}'. Siap melakukan putaran hadap.")
    return data_qr_terbaca


# ==============================================================================
# TAHAP 5: PUTAR HADAP KE KANAN (YAW 0° / ARAH TIMUR / +X)
# ==============================================================================
def tahap_5_putar_kanan_menghadap_timur(
    drone: DroneController,
    sensor: SensorListener
):
    print("\n" + "=" * 65)
    print(">>> [TAHAP 5] PUTAR HADAP KE KANAN (MENGHADAP TIMUR / +X)")
    print("=" * 65)

    drone.update()
    yaw_awal = drone.yaw_deg
    print(f"[*] Sudut yaw saat ini: {yaw_awal:.1f}°. Memutar 90° ke kanan menghadap Timur...")

    # Memutar relatif 90° searah jarum jam menggunakan kecepatan yaw_rate
    # (Frame-invariant: menjamin moncong drone membelok ke kanan menyusuri lorong timur)
    t_putar = time.time()
    kecepatan_putar_rad = math.radians(45.0)  # 45 deg/s
    durasi_target = 2.0                      # 2 detik * 45 deg/s = 90 deg

    while time.time() - t_putar < durasi_target:
        drone.send_velocity_body(vx=0.0, vy=0.0, vz_up=0.0, yaw_rate=kecepatan_putar_rad)
        drone.update()
        time.sleep(0.04)

    # Rem aktif dan setpoint absolute yaw timur
    drone.brake(0.8)
    drone.set_yaw(0.0, speed_deg_s=45.0, timeout=2.5)
    drone.brake(0.8)

    drone.update()
    print(f"[OK] Putaran selesai! Yaw baru: {drone.yaw_deg:.1f}° | Altitude: {drone.altitude:.2f} m")
    print("[SELESAI TAHAP 5] Drone kini menghadap lurus ke deretan rintangan Timur (+X).")


# ==============================================================================
# TAHAP 6: MENEMBUS SETIAP RINTANGAN DI DEPANNYA (CELAH KOSONG)
# ==============================================================================
def tahap_6_tembus_rintangan_berjendela(
    drone: DroneController,
    sensor: SensorListener
):
    print("\n" + "=" * 65)
    print(">>> [TAHAP 6] MENEMBUS SELURUH RINTANGAN BERJENDELA (CELAH KOSONG)")
    print("=" * 65)

    # Inisialisasi Detektor Belah Ketupat untuk servoing visual
    detector_diamond = BelahKetupatDetector(color="biru")

    # --------------------------------------------------------------------------
    # SUB-TAHAP 6.0: KELUAR DARI LORONG TIMUR MENUJU RUANG RINTANGAN
    # --------------------------------------------------------------------------
    print("\n--- [6.0] Terbang maju menyusuri lorong keluar arah Timur ---")
    # Keluar dari dinding lorong (sekitar 5.5 m maju)
    t_jalan = time.time()
    while time.time() - t_jalan < 6.0:
        drone.update()
        d_kiri = sensor.get_lidar_left()
        d_kanan = sensor.get_lidar_right()
        vy_cmd = 0.0
        if d_kiri < 1.8 and d_kanan < 1.8:
            err = d_kiri - d_kanan
            vy_cmd = float(math.copysign(min(0.25, abs(err * 0.4)), -err))
        drone.send_velocity_body(vx=0.75, vy=vy_cmd, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)
    drone.brake(0.8)
    print(f"[OK] Keluar dari lorong. Altitude: {drone.altitude:.2f} m")

    # --------------------------------------------------------------------------
    # SUB-TAHAP 6.1: RINTANGAN 1 (ORANGE BARRIER - CUBE pada X ≈ -11.84)
    # Jendela Celah Kosong: Z ≈ 3.16 m, Y ≈ 9.16 m (dimensi Z: [2.7 .. 3.6] m)
    # --------------------------------------------------------------------------
    print("\n--- [6.1] Rintangan 1: Orange Barrier (Celah Atas Z ≈ 3.16 m) ---")
    alt_target_1 = 3.16
    delta_z1 = alt_target_1 - drone.altitude
    print(f"[*] Menyelaraskan ketinggian ke celah jendela Z = {alt_target_1:.2f} m (delta: {delta_z1:+.2f} m)...")
    drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=0.0, naik_m=delta_z1, kecepatan_maks=0.7, toleransi=0.10)
    drone.brake(0.5)

    # Menembus celah rintangan 1 (maju sejauh 3.5 m)
    print("[*] Maju menembus celah jendela Orange Barrier...")
    drone.gerak_relatif_presisi(maju_m=3.5, kanan_m=0.0, naik_m=0.0, kecepatan_maks=0.7, toleransi=0.15)
    drone.brake(0.8)
    print(f"[OK] Rintangan 1 BERHASIL DITEMBUS! Altitude: {drone.altitude:.2f} m")

    # --------------------------------------------------------------------------
    # SUB-TAHAP 6.2: RINTANGAN 2 (BELAH KETUPAT BIRU - CUBE_007 pada X ≈ -8.86)
    # Celah Kosong: Pusat Belah Ketupat Biru pada Z ≈ 1.21 m, Y ≈ 8.69 m
    # --------------------------------------------------------------------------
    print("\n--- [6.2] Rintangan 2: Belah Ketupat Biru (Celah Bawah Z ≈ 1.21 m) ---")
    alt_target_2 = 1.21
    delta_z2 = alt_target_2 - drone.altitude
    print(f"[*] Menurunkan ketinggian ke pusat belah ketupat Z = {alt_target_2:.2f} m (delta: {delta_z2:+.2f} m)...")
    drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=0.0, naik_m=delta_z2, kecepatan_maks=0.7, toleransi=0.10)
    drone.brake(0.5)

    # Servoing visual deteksi belah ketupat biru
    detector_diamond.set_color("biru")
    print("[*] Melakukan penyesuaian centering visual Belah Ketupat Biru...")
    t_servo = time.time()
    while time.time() - t_servo < 3.5:
        drone.update()
        frame = sensor.get_frame_front()
        if frame is not None:
            res_blue = detector_diamond.detect(frame, draw=False)
            if res_blue.detected:
                vy_trim = float(math.copysign(min(0.20, abs(res_blue.norm_dx * 0.3)), res_blue.norm_dx))
                vz_trim = float(math.copysign(min(0.20, abs(res_blue.norm_dy * 0.3)), -res_blue.norm_dy))
                drone.send_velocity_body(vx=0.20, vy=vy_trim, vz_up=vz_trim, yaw_rate=0.0)
                if res_blue.is_centered:
                    print(f"[OK] Centered pada Belah Ketupat Biru! dx:{res_blue.dx}, dy:{res_blue.dy}")
                    break
                time.sleep(0.04)
                continue
        drone.send_velocity_body(vx=0.20, vy=0.0, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)

    # Menembus belah ketupat biru (maju sejauh 2.5 m)
    print("[*] Maju menembus celah Belah Ketupat Biru...")
    drone.gerak_relatif_presisi(maju_m=2.5, kanan_m=0.0, naik_m=0.0, kecepatan_maks=0.7, toleransi=0.15)
    drone.brake(0.8)
    print(f"[OK] Rintangan 2 (Biru) BERHASIL DITEMBUS! Altitude: {drone.altitude:.2f} m")

    # --------------------------------------------------------------------------
    # SUB-TAHAP 6.3: RINTANGAN 3 (BELAH KETUPAT HIJAU - CUBE_008 pada X ≈ -6.83)
    # Celah Kosong: Jendela Tinggi pada Z ≈ 4.06 m, Y ≈ 9.78 m
    # --------------------------------------------------------------------------
    print("\n--- [6.3] Rintangan 3: Belah Ketupat Hijau (Celah Tinggi Z ≈ 4.06 m) ---")
    alt_target_3 = 4.06
    delta_z3 = alt_target_3 - drone.altitude
    print(f"[*] Menaikkan ketinggian ke jendela hijau Z = {alt_target_3:.2f} m (delta: {delta_z3:+.2f} m)...")
    drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=0.0, naik_m=delta_z3, kecepatan_maks=0.8, toleransi=0.12)
    drone.brake(0.5)

    # Servoing visual deteksi belah ketupat hijau
    detector_diamond.set_color("hijau")
    print("[*] Melakukan penyesuaian centering visual Belah Ketupat Hijau...")
    t_servo = time.time()
    while time.time() - t_servo < 3.5:
        drone.update()
        frame = sensor.get_frame_front()
        if frame is not None:
            res_green = detector_diamond.detect(frame, draw=False)
            if res_green.detected:
                vy_trim = float(math.copysign(min(0.20, abs(res_green.norm_dx * 0.3)), res_green.norm_dx))
                vz_trim = float(math.copysign(min(0.20, abs(res_green.norm_dy * 0.3)), -res_green.norm_dy))
                drone.send_velocity_body(vx=0.20, vy=vy_trim, vz_up=vz_trim, yaw_rate=0.0)
                if res_green.is_centered:
                    print(f"[OK] Centered pada Belah Ketupat Hijau! dx:{res_green.dx}, dy:{res_green.dy}")
                    break
                time.sleep(0.04)
                continue
        drone.send_velocity_body(vx=0.20, vy=0.0, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)

    # Menembus belah ketupat hijau (maju sejauh 2.8 m)
    print("[*] Maju menembus celah Belah Ketupat Hijau...")
    drone.gerak_relatif_presisi(maju_m=2.8, kanan_m=0.0, naik_m=0.0, kecepatan_maks=0.7, toleransi=0.15)
    drone.brake(0.8)
    print(f"[OK] Rintangan 3 (Hijau) BERHASIL DITEMBUS! Altitude: {drone.altitude:.2f} m")

    # --------------------------------------------------------------------------
    # SUB-TAHAP 6.4: RINTANGAN 4 (BELAH KETUPAT PINK - CUBE_009 pada X ≈ -3.41)
    # Celah Kosong: Pusat Belah Ketupat Pink pada Z ≈ 2.45 m, Y ≈ 8.04 m
    # --------------------------------------------------------------------------
    print("\n--- [6.4] Rintangan 4: Belah Ketupat Pink (Celah Tengah Z ≈ 2.45 m) ---")
    alt_target_4 = 2.45
    delta_z4 = alt_target_4 - drone.altitude
    print(f"[*] Menyelaraskan ketinggian ke belah ketupat pink Z = {alt_target_4:.2f} m (delta: {delta_z4:+.2f} m)...")
    drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=0.0, naik_m=delta_z4, kecepatan_maks=0.7, toleransi=0.10)
    drone.brake(0.5)

    # Servoing visual deteksi belah ketupat pink
    detector_diamond.set_color("pink")
    print("[*] Melakukan penyesuaian centering visual Belah Ketupat Pink...")
    t_servo = time.time()
    while time.time() - t_servo < 3.5:
        drone.update()
        frame = sensor.get_frame_front()
        if frame is not None:
            res_pink = detector_diamond.detect(frame, draw=False)
            if res_pink.detected:
                vy_trim = float(math.copysign(min(0.20, abs(res_pink.norm_dx * 0.3)), res_pink.norm_dx))
                vz_trim = float(math.copysign(min(0.20, abs(res_pink.norm_dy * 0.3)), -res_pink.norm_dy))
                drone.send_velocity_body(vx=0.20, vy=vy_trim, vz_up=vz_trim, yaw_rate=0.0)
                if res_pink.is_centered:
                    print(f"[OK] Centered pada Belah Ketupat Pink! dx:{res_pink.dx}, dy:{res_pink.dy}")
                    break
                time.sleep(0.04)
                continue
        drone.send_velocity_body(vx=0.20, vy=0.0, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)

    # Menembus belah ketupat pink (maju sejauh 3.5 m ke ruang terbuka)
    print("[*] Maju menembus celah Belah Ketupat Pink ke ruang terbuka...")
    drone.gerak_relatif_presisi(maju_m=3.5, kanan_m=0.0, naik_m=0.0, kecepatan_maks=0.7, toleransi=0.15)
    drone.brake(1.0)
    print(f"[OK] Rintangan 4 (Pink) BERHASIL DITEMBUS! Drone tiba di area terbuka.")

    # --------------------------------------------------------------------------
    # SUB-TAHAP 6.5: PENYESUAIAN AKHIR DI RUANG TERBUKA (HANDOFF AREA)
    # --------------------------------------------------------------------------
    # Kembalikan ketinggian jelajah Z ≈ 2.0 m
    delta_z_final = 2.00 - drone.altitude
    drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=0.0, naik_m=delta_z_final, kecepatan_maks=0.6, toleransi=0.10)
    drone.brake(1.0)
    print(f"[SELESAI TAHAP 6] Seluruh rintangan berjendela berhasil dilewati! Altitude: {drone.altitude:.2f} m")


# ==============================================================================
# MAIN EXECUTIVE FUNCTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Navigasi Otonom Jalur Kiri Gamaforce")
    parser.add_argument("--auto-land", action="store_true", help="Mendaratkan drone setelah seluruh rintangan terlewati")
    parser.add_argument("--handoff-misi2", action="store_true", help="Lanjut otomatis ke Misi 2 (Pad Hijau & Ring scenario='left')")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀🚀 [GAMAFORCE] MEMULAI NAVIGASI OTONOM LINTASAN KIRI (FULL PIPELINE) 🚀🚀")
    print("=" * 70)

    # Inisialisasi DroneController & SensorListener
    drone = get_drone()
    sensor = get_sensor()

    # Eksekusi seluruh rangkaian tahap misi kiri secara berurutan:
    # 1. Takeoff & Bergerak ke Kiri menuju obstacle kuning
    tahap_1_takeoff_dan_menuju_kuning(drone, sensor)

    # 2. Lewat di atas rintangan kuning (U terbalik)
    tahap_2_lewati_atas_kuning(drone, sensor)

    # 3. Melewati obstacle merah sembarang (BézierCurve)
    tahap_3_lewati_obstacle_merah(drone, sensor)

    # 4. Menyusuri lorong lurus & membaca QR / ArUco marker
    hasil_qr = tahap_4_susuri_lorong_dan_baca_qr(drone, sensor)

    # 5. Putar hadap ke kanan (menghadap Timur / +X)
    tahap_5_putar_kanan_menghadap_timur(drone, sensor)

    # 6. Bergerak lurus & menembus seluruh rintangan berjendela (celah kosong)
    tahap_6_tembus_rintangan_berjendela(drone, sensor)

    print("\n" + "=" * 70)
    print("🏆🏆🏆 [MISI SELESAI] SELURUH RINTANGAN LINTASAN KIRI SUKSES DILEWATI! 🏆🏆🏆")
    if hasil_qr:
        print(f"📌 KODE ARUCO / QR TERVERIFIKASI: {hasil_qr}")
    print(f"📌 POSISI AKHIR DRONE: Altitude = {drone.altitude:.2f} m | Yaw = {drone.yaw_deg:.1f}°")
    print("=" * 70)

    # Opsi Kelanjutan
    if args.handoff_misi2 and misi2 is not None:
        print("\n>>> [HANDOFF] Melanjutkan ke Misi 2 (Pad Hijau & Ring) [scenario='left']...")
        misi2.misi(drone=drone, sensor=sensor, scenario="left")
    elif args.auto_land:
        print("\n[*] Mendaratkan drone secara aman di area terbuka...")
        drone.land()
    else:
        print("\n[*] Drone melayang stabil (hovering) di udara di area serah-terima.")
        drone.brake(1.0)


if __name__ == '__main__':
    main()

