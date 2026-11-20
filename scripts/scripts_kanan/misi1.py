#!/usr/bin/env python3
"""
================================================================================
MISI 1: NAVIGASI LABIRIN & TEROWONGAN (JALUR KANAN)
================================================================================
Alur Misi:
1. Geser ke KANAN 11.7m presisi (Setpoint EKF) menuju mulut lorong terowongan.
2. Menyusuri 15 tahap rintangan lorong, balok, jendela menggunakan sensor LiDAR 1D.
3. Keluar dari terowongan ke ruang terbuka dan melayang stabil di udara.
================================================================================
"""

import sys
import os
import time
from typing import Optional, Tuple

# Pastikan path modul scripts terdeteksi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)
for p in (SCRIPTS_DIR, CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from core import DroneController, SensorListener, get_drone, get_sensor


def misi(
    drone: Optional[DroneController] = None,
    sensor: Optional[SensorListener] = None,
    auto_land: bool = False
) -> Tuple[DroneController, SensorListener]:
    """
    Eksekusi Misi 1: Labirin & Terowongan.
    """
    if drone is None:
        drone = get_drone()
    if sensor is None:
        sensor = get_sensor()

    print("\n" + "=" * 65)
    print(">>> [MISI 1] MEMULAI NAVIGASI LABIRIN & TEROWONGAN (JALUR KANAN)")
    print("=" * 65)

    drone.takeoff_if_needed(1.35)
    drone.brake(1.0)

    # --------------------------------------------------------------------------
    # TAHAP 1: Geser ke KANAN Presisi (Setpoint EKF 11.7m)
    # --------------------------------------------------------------------------
    print("\n[Tahap 1] Bergerak ke kanan 11.7m menuju mulut lorong...")
    drone.gerak_relatif_presisi(maju_m=0.0, kanan_m=11.7, naik_m=0.0, kecepatan_maks=1.0)
    drone.brake(1.0)

    # --------------------------------------------------------------------------
    # TAHAP 2: Masuk lorong 1 hingga ujung depan terdeteksi
    # --------------------------------------------------------------------------
    print("\n[Tahap 2] Terbang maju masuk ke lorong 1...")
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_depan = sensor.get_lidar_front()
        if d_depan < 1.30:
            print(f"[OK] Ujung lorong 1 terdeteksi ({d_depan:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vx = 0.45 if d_depan < 2.50 else 1.0
        drone.move_timed(vx, 0.0, 0.0, 0.1)

    # --------------------------------------------------------------------------
    # TAHAP 3: Geser ke kiri dalam lorong hingga dinding kiri terdeteksi
    # --------------------------------------------------------------------------
    print("\n[Tahap 3] Geser ke kiri dalam lorong...")
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_kiri = sensor.get_lidar_left()
        if d_kiri < 0.80:
            print(f"[OK] Dinding kiri terdeteksi ({d_kiri:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vy = -0.45 if d_kiri < 2.00 else -1.0
        drone.move_timed(0.0, vy, 0.0, 0.1)

    # --------------------------------------------------------------------------
    # TAHAP 4: Maju lorong 2 hingga ujung depan terdeteksi
    # --------------------------------------------------------------------------
    print("\n[Tahap 4] Terbang maju masuk ke lorong 2...")
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_depan = sensor.get_lidar_front()
        if d_depan < 1.00:
            print(f"[OK] Ujung lorong 2 terdeteksi ({d_depan:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vx = 0.45 if d_depan < 2.50 else 1.0
        drone.move_timed(vx, 0.0, 0.0, 0.1)

    # --------------------------------------------------------------------------
    # TAHAP 5: Naik ketinggian melewati rintangan bawah
    # --------------------------------------------------------------------------
    print("\n[Tahap 5] Naik ketinggian melewati rintangan bawah...")
    t0 = time.time()
    while time.time() - t0 < 15.0:
        d_bawah = sensor.get_lidar_down()
        if d_bawah > 1.30:
            print(f"[OK] Ketinggian aman tercapai ({d_bawah:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        drone.move_timed(0.0, 0.0, 0.40, 0.05)

    # --------------------------------------------------------------------------
    # TAHAP 6: Maju melewati rintangan balok
    # --------------------------------------------------------------------------
    print("\n[Tahap 6] Terbang maju melewati rintangan balok...")
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_depan = sensor.get_lidar_front()
        if d_depan < 1.30:
            print(f"[OK] Ujung lorong terdeteksi ({d_depan:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vx = 0.45 if d_depan < 2.50 else 1.0
        drone.move_timed(vx, 0.0, 0.0, 0.1)

    # --------------------------------------------------------------------------
    # TAHAP 7: Turun kembali ke ketinggian normal
    # --------------------------------------------------------------------------
    print("\n[Tahap 7] Menurunkan ketinggian kembali...")
    t0 = time.time()
    while time.time() - t0 < 20.0:
        d_bawah = sensor.get_lidar_down()
        if 0.60 < d_bawah < 0.90:
            print(f"[OK] Ketinggian normal tercapai ({d_bawah:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        drone.move_timed(0.0, 0.0, -0.45, 0.05)

    # --------------------------------------------------------------------------
    # TAHAP 8: Maju melewati lorong
    # --------------------------------------------------------------------------
    print("\n[Tahap 8] Terbang maju melewati lorong...")
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_depan = sensor.get_lidar_front()
        if d_depan < 1.20:
            print(f"[OK] Dinding depan terdeteksi ({d_depan:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vx = 0.45 if d_depan < 2.50 else 1.0
        drone.move_timed(vx, 0.0, 0.0, 0.1)

    # --------------------------------------------------------------------------
    # TAHAP 9: Naik ke atas melewati rintangan
    # --------------------------------------------------------------------------
    print("\n[Tahap 9] Terbang ke atas melewati rintangan...")
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_bawah = sensor.get_lidar_down()
        if d_bawah > 1.50:
            print(f"[OK] Ketinggian rintangan tercapai ({d_bawah:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vz = 0.40 if d_bawah < 2.50 else 0.80
        drone.move_timed(0.0, 0.0, vz, 0.1)

    # --------------------------------------------------------------------------
    # TAHAP 10: Minggir untuk menyelaraskan lubang
    # --------------------------------------------------------------------------
    print("\n[Tahap 10] Terbang minggir untuk menyelaraskan lubang...")
    drone.move_timed(0.0, -0.25, 0.0, 3.0)
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_depan = sensor.get_lidar_front()
        if d_depan > 1.20:
            print(f"[OK] Bukaan lubang terdeteksi ({d_depan:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vy = -0.30 if d_depan < 2.50 else -0.70
        drone.move_timed(0.0, vy, 0.0, 0.1)

    # --------------------------------------------------------------------------
    # TAHAP 11: Terbang maju menuju lubang
    # --------------------------------------------------------------------------
    print("\n[Tahap 11] Terbang maju mendekati lubang...")
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_depan = sensor.get_lidar_front()
        if d_depan < 1.45:
            print(f"[OK] Mulut lubang tercapai ({d_depan:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vx = 0.40 if d_depan < 2.50 else 0.80
        drone.move_timed(vx, 0.0, 0.0, 0.1)

    # --------------------------------------------------------------------------
    # TAHAP 12: Masuk melintasi lubang
    # --------------------------------------------------------------------------
    print("\n[Tahap 12] Melintasi lubang...")
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_depan = sensor.get_lidar_front()
        if d_depan < 1.50:
            print(f"[OK] Melewati lubang, dinding depan terdeteksi ({d_depan:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vx = 0.40 if d_depan < 2.50 else 0.80
        drone.move_timed(vx, 0.0, 0.0, 0.1)

    # --------------------------------------------------------------------------
    # TAHAP 13: Menurun melewati rintangan bertingkat
    # --------------------------------------------------------------------------
    print("\n[Tahap 13] Terbang menurun untuk melewati rintangan...")
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_depan = sensor.get_lidar_front()
        d_bawah = sensor.get_lidar_down()
        if d_depan < 0.40 and d_bawah < 0.40:
            print(f"[OK] Dinding rintangan terdeteksi (Depan={d_depan:.2f}m, Bawah={d_bawah:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vx = 0.30 if d_depan < 2.50 else 1.00
        vz = -0.30 if d_bawah < 2.00 else -0.80
        drone.move_timed(vx, 0.0, vz, 0.1)

    # --------------------------------------------------------------------------
    # TAHAP 14: Terbang naik melewati rintangan
    # --------------------------------------------------------------------------
    print("\n[Tahap 14] Terbang naik untuk melewati rintangan...")
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_atas = sensor.get_lidar_up()
        if d_atas < 1.20:
            print(f"[OK] Dinding atas terdeteksi ({d_atas:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vz = 0.45 if d_atas < 2.00 else 1.00
        drone.move_timed(0.0, 0.0, vz, 0.1)

    # --------------------------------------------------------------------------
    # TAHAP 15: Terbang maju keluar labirin
    # --------------------------------------------------------------------------
    print("\n[Tahap 15] Terbang maju untuk keluar labirin...")
    t0 = time.time()
    while time.time() - t0 < 25.0:
        d_atas = sensor.get_lidar_up()
        d_kiri = sensor.get_lidar_left()
        d_kanan = sensor.get_lidar_right()
        if d_atas > 5.00:
            print(f"[OK] Atap terowongan terbuka / keluar labirin ({d_atas:.2f}m)! Berhenti.")
            drone.brake(1.0)
            break
        vx = 0.40 if (d_kiri < 2.50 or d_kanan < 2.50) else 1.00
        drone.move_timed(vx, 0.0, 0.0, 0.1)
    
    print("\n>>> MISI SELESAI.")

    print("\n" + "=" * 65)
    print("🎉🎉 [MISI 1 SUKSES] LABIRIN & TEROWONGAN TELAH DILEWATI! 🎉🎉")
    print("=" * 65)

    if auto_land:
        drone.land()
    else:
        drone.brake(1.0)

    return drone, sensor


if __name__ == '__main__':
    misi(auto_land=True)
