#!/usr/bin/env python3
"""
================================================================================
MISI 2: NAVIGASI PAD HIJAU & MENEMBUS DONAT MERAH (DUAL-CAMERA + LIDAR)
================================================================================
Alur Misi:
1. Meluncur aktif ke kiri (sumbu X negatif) mencari lingkaran hijau.
2. Centering presisi posisi X & Y di atas lingkaran hijau (kamera bawah).
3. Centering titik pusat lubang donat merah (anchored di atas lingkaran hijau).
4. Melaju menembus donat merah dengan verifikasi fisik LiDAR 4 arah.
5. Pengereman penuh dan hover stabil di seberang donat merah.
================================================================================
"""

import sys
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# Pastikan path modul scripts terdeteksi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)
for p in (SCRIPTS_DIR, CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from core import DroneController, SensorListener, get_drone, get_sensor


def tahap_1_cari_lingkaran_hijau(drone: DroneController, sensor: SensorListener):
    """Tahap 1: Meluncur aktif ke kiri mencari pad hijau."""
    print("\n" + "=" * 65)
    print(">>> [MISI 2 - TAHAP 1] MELUNCUR KE KIRI MENCARI PAD HIJAU")
    print("=" * 65)

    VY_CARI_KIRI = -1.80
    VX_MAJU_TRIM = 0.40

    while True:
        drone.update()
        green = sensor.get_green()
        if green is not None and green.detected:
            print(f"\n[OK] LINGKARAN HIJAU TERDETEKSI! Area={green.area:.1f}, (cx={green.cx}, cy={green.cy})")
            drone.brake(0.5)
            break
        drone.send_velocity_body(vx=VX_MAJU_TRIM, vy=VY_CARI_KIRI, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)


def tahap_2_centering_lingkaran_hijau(drone: DroneController, sensor: SensorListener):
    """Tahap 2: Centering presisi di atas pad hijau (kamera bawah)."""
    print("\n" + "=" * 65)
    print(">>> [MISI 2 - TAHAP 2] CENTERING PRESISI PAD HIJAU (KAMERA BAWAH)")
    print("=" * 65)

    steady_count = 0
    while True:
        drone.update()
        green = sensor.get_green()
        if green is None or not green.detected:
            drone.send_velocity_body(0.0, 0.0, 0.0, yaw_rate=0.0)
            time.sleep(0.04)
            continue

        dx, dy = green.dx, green.dy
        vx_cmd = np.clip(-0.0055 * dy, -0.85, 0.85)
        vy_cmd = np.clip( 0.0055 * dx, -0.85, 0.85)

        if abs(dx) <= 22 and abs(dy) <= 22:
            steady_count += 1
            drone.send_velocity_body(0.0, 0.0, 0.0, yaw_rate=0.0)
            if steady_count >= 6:
                print(f"[OK] Centering Lingkaran Hijau SELESAI! (dx={dx:+d}, dy={dy:+d})")
                drone.brake(0.6)
                break
        else:
            steady_count = 0
            drone.send_velocity_body(vx_cmd, vy_cmd, 0.0, yaw_rate=0.0)
        time.sleep(0.04)


def tahap_3_centering_donat_merah(drone: DroneController, sensor: SensorListener):
    """Tahap 3: Centering lubang donat merah (anchored di atas pad hijau)."""
    print("\n" + "=" * 65)
    print(">>> [MISI 2 - TAHAP 3] CENTERING LUBANG DONAT (ANCHORED DI PAD HIJAU)")
    print("=" * 65)

    sensor.set_mode("DONUT")
    drone.brake(0.5)

    steady_count = 0
    last_log_time = 0.0
    search_timer = time.time()

    while True:
        drone.update()
        green = sensor.get_green()
        donut = sensor.get_donut()
        alt = drone.altitude

        # Kontrol anchor di atas lingkaran hijau
        if green is not None and green.detected:
            vx_anchor = np.clip(-0.0050 * green.dy, -0.45, 0.45)
            vy_anchor = np.clip( 0.0050 * green.dx, -0.45, 0.45)
            is_over_green = (abs(green.dx) <= 28 and abs(green.dy) <= 28)
        else:
            vx_anchor, vy_anchor = 0.0, 0.0
            is_over_green = True

        # Centering donat merah di kamera depan
        if donut is not None and donut.detected:
            dx_d, dy_d = donut.dx, donut.dy
            vz_cmd = np.clip(-0.0050 * dy_d, -0.55, 0.55)
            yaw_cmd = np.clip(0.0040 * dx_d, -0.45, 0.45)

            is_tight_centered = (abs(dx_d) <= 18 and abs(dy_d) <= 18 and donut.has_hole and alt >= 2.20 and is_over_green)
            if is_tight_centered:
                steady_count += 1
                drone.send_velocity_body(0.0, 0.0, 0.0, yaw_rate=0.0)
                if steady_count >= 15:
                    print(f"\n[OK] CENTERING DONAT MERAH SELESAI! (Offset: dx={dx_d:+d}, dy={dy_d:+d} | Alt={alt:.2f}m)")
                    drone.brake(0.6)
                    break
            else:
                steady_count = 0
                drone.send_velocity_body(vx=vx_anchor, vy=vy_anchor, vz_up=vz_cmd, yaw_rate=yaw_cmd)

            now = time.time()
            if now - last_log_time > 0.4:
                status_str = f"STABIL {steady_count}/15" if is_tight_centered else "CENTERING..."
                print(f"  [{status_str}] Donut(dx={dx_d:+3d}, dy={dy_d:+3d}) | Alt={alt:.2f}m")
                last_log_time = now
        else:
            steady_count = 0
            vz_search = 0.45 if alt < 2.65 else -0.35
            drone.send_velocity_body(vx=vx_anchor, vy=vy_anchor, vz_up=vz_search, yaw_rate=0.0)
            now = time.time()
            if now - search_timer > 0.8:
                print(f"  [*] Naik vertikal mencari lubang donat... (Alt: {alt:.2f}m)")
                search_timer = now

        time.sleep(0.04)


def tahap_4_menembus_donat_merah(drone: DroneController, sensor: SensorListener):
    """Tahap 4: Melaju maju menembus lubang donat dengan verifikasi fisik LiDAR."""
    print("\n" + "=" * 65)
    print(">>> [MISI 2 - TAHAP 4] MELAJU MENEMBUS DONAT MERAH (VERIFIKASI LIDAR)")
    print("=" * 65)

    has_entered_donut = False
    t_start = time.time()
    t_entered = 0.0

    while True:
        drone.update()
        drone.send_velocity_body(vx=2.80, vy=0.0, vz_up=0.0, yaw_rate=0.0)

        d_left = sensor.get_lidar_left()
        d_right = sensor.get_lidar_right()
        d_up = sensor.get_lidar_up()
        now = time.time()

        # Deteksi berada di dalam rongga donat merah (< 1.30 m)
        is_narrow_wall = (d_left < 1.30 or d_right < 1.30 or d_up < 1.30)

        if not has_entered_donut:
            if is_narrow_wall:
                has_entered_donut = True
                t_entered = now
                print(f"\n[LIDAR EVENT] Memasuki rongga donat! Left={d_left:.2f}m, Right={d_right:.2f}m, Up={d_up:.2f}m")
        else:
            # Terkonfirmasi keluar (ruang terbuka > 1.65 m) dan telah melaju >= 0.35 s
            is_space_opened = (d_left > 1.65 and d_right > 1.65 and d_up > 1.65)
            if is_space_opened and (now - t_entered >= 0.35):
                print(f"\n[LIDAR EVENT] Terkonfirmasi keluar dari donat merah! Ruang kembali terbuka.")
                break

        if now - t_start >= 2.8:
            break
        time.sleep(0.03)

    drone.brake(1.0)
    print("[*] Menyesuaikan ketinggian ke level target Bulls Eye (Z ~ 1.5 m)...")
    t_descend = time.time() + 2.0
    while time.time() < t_descend:
        drone.update()
        if drone.altitude <= 1.55:
            break
        drone.send_velocity_body(vx=0.0, vy=0.0, vz_up=-0.50, yaw_rate=0.0)
        time.sleep(0.04)
    drone.brake(0.5)


def misi(
    drone: Optional[DroneController] = None,
    sensor: Optional[SensorListener] = None
) -> Tuple[DroneController, SensorListener]:
    """
    Eksekusi Misi 2 secara terpadu.
    """
    if drone is None:
        drone = get_drone()
    if sensor is None:
        sensor = get_sensor()

    print("\n" + "=" * 65)
    print(">>> [MISI 2] MEMULAI NAVIGASI PAD HIJAU & DONAT MERAH")
    print("=" * 65)

    drone.takeoff_if_needed(1.35)
    drone.brake(0.5)

    tahap_1_cari_lingkaran_hijau(drone, sensor)
    tahap_2_centering_lingkaran_hijau(drone, sensor)
    tahap_3_centering_donat_merah(drone, sensor)
    tahap_4_menembus_donat_merah(drone, sensor)

    print("\n" + "=" * 65)
    print("🎉🎉 [MISI 2 SUKSES] PAD HIJAU & DONAT MERAH BERHASIL DITUNTASKAN! 🎉🎉")
    print("=" * 65)
    return drone, sensor


if __name__ == '__main__':
    misi()
