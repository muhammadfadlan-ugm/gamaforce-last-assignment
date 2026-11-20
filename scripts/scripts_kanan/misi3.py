#!/usr/bin/env python3
"""
================================================================================
GAMAFORCE - MISI 3: CENTERING TITIK PUSAT ANGKA & SERANGAN KAMIKAZE BULLS EYE
================================================================================
Arsitektur Modular:
  - Menerima instance DroneController & SensorListener dari scripts/core/
  - Tahap 1: Scan & Lock Bounding Box Angka Sasaran (Default: Angka #2)
  - Tahap 2: Centering Presisi ke Titik Pusat Bounding Box Angka (Hover vx=0.0)
  - Tahap 3: Kamikaze Strike Sprint Lurus (vx = 7.0 m/s) hingga Impact Terdeteksi
================================================================================
"""

import sys
import os
import time
import argparse
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy

# Otomatis gunakan venv ArduPilot jika modul pymavlink belum terinstal di system
try:
    import pymavlink
except ModuleNotFoundError:
    venv_py = os.path.expanduser('~/venv-ardupilot/bin/python3')
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise

# Setup path untuk core dan deteksi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)
CORE_DIR = os.path.join(SCRIPTS_DIR, "core")
for p in (SCRIPTS_DIR, CORE_DIR, CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from core import DroneController, SensorListener, get_drone, get_sensor


def tahap_1_kunci_target_angka(
    drone: DroneController,
    sensor: SensorListener,
    target_digit: int = 2
):
    """
    Tahap 1: Memindai feed kamera depan mencari target angka (default: 2).
    Mengaktifkan mode sensor KAMIKAZE agar kamera depan fokus pada Bullseye.
    """
    print("\n" + "=" * 65)
    print(f">>> [MISI 3 - TAHAP 1] SCAN & DETEKSI TARGET ANGKA #{target_digit}")
    print("=" * 65)

    drone.brake(0.4)
    sensor.set_mode("KAMIKAZE")
    print(f"[*] Mode sensor dialihkan ke: KAMIKAZE (Deteksi Sasaran #{target_digit})")

    step_cnt = 0
    while rclpy.ok():
        drone.update()
        res_k = sensor.get_kamikaze()

        if res_k is not None:
            # Temukan target dengan digit yang cocok
            matching = [t for t in res_k.all_targets if t.digit == target_digit]
            if res_k.locked_target and res_k.locked_target.digit == target_digit:
                matching.append(res_k.locked_target)

            if matching:
                best = max(matching, key=lambda t: t.area)
                bx, by, bw, bh = best.bbox
                print(f"\n[OK] TARGET ANGKA #{target_digit} TERDETEKSI!")
                print(f"     -> Bounding Box: [X={bx}, Y={by}, W={bw}, H={bh}]")
                print(f"     -> Titik Pusat : ({best.cx}, {best.cy}) | Offset: dx={best.dx:+d}, dy={best.dy:+d}")
                break

        step_cnt += 1
        if step_cnt % 25 == 0:
            print(f"  [*] Memindai kamera depan mencari Bounding Box Angka #{target_digit}...")

        # Maju perlahan jika target belum terlihat di FOV
        drone.send_velocity_body(vx=0.15, vy=0.0, vz_up=0.0, yaw_rate=0.0)
        time.sleep(0.04)

    drone.brake(0.3)


def tahap_2_centering_titik_pusat_box(
    drone: DroneController,
    sensor: SensorListener,
    target_digit: int = 2
):
    """
    Tahap 2: Centering presisi ke titik pusat bounding box angka.
    PENTING: Drone hover di tempat (vx = 0.0) menyelaraskan reticle ke titik pusat angka.
    """
    print("\n" + "=" * 65)
    print(f">>> [MISI 3 - TAHAP 2] CENTERING PRESISI TITIK PUSAT BOX ANGKA #{target_digit}")
    print(">>> (Hover di Tempat vx=0.0 | Visual Servoing Yaw, Vy, Vz)")
    print("=" * 65)

    steady_aim = 0

    while rclpy.ok():
        drone.update()
        res_k = sensor.get_kamikaze()

        target_now = None
        if res_k is not None:
            matching = [t for t in res_k.all_targets if t.digit == target_digit]
            if res_k.locked_target and res_k.locked_target.digit == target_digit:
                matching.append(res_k.locked_target)
            if matching:
                target_now = max(matching, key=lambda t: t.area)

        if target_now is not None:
            dx = target_now.dx
            dy = target_now.dy
            bx, by, bw, bh = target_now.bbox
            tcx, tcy = target_now.cx, target_now.cy

            # Visual Servoing P-Controller:
            # yaw_rate: mengarahkan moncong drone secara horizontal ke titik tengah box
            # vy_cmd  : pergeseran samping lateral agar posisi bodi sejajar garis lurus sasaran
            # vz_cmd  : menyamakan ketinggian kamera tepat dengan titik tengah vertikal box
            yaw_rate_cmd = float(np.clip(0.0055 * dx, -0.80, 0.80))
            vy_cmd = float(np.clip(0.0035 * dx, -0.45, 0.45))
            vz_cmd = float(np.clip(-0.0055 * dy, -0.65, 0.65))

            # Kondisi Centering Presisi (|dx| <= 16 px & |dy| <= 16 px)
            if abs(dx) <= 16 and abs(dy) <= 16:
                steady_aim += 1
                drone.send_velocity_body(0.0, 0.0, 0.0, yaw_rate=0.0)
                if steady_aim >= 5:  # Terkunci stabil selama ~0.2 detik
                    print(f"\n[OK] RETICLE TERKUNCI SEMPURNA PADA TITIK PUSAT BOX ANGKA #{target_digit}!")
                    print(f"     -> Box: [{bx}, {by}, {bw}, {bh}] | Pusat Angka: ({tcx}, {tcy}) | Error: dx={dx:+d}, dy={dy:+d}")
                    drone.brake(0.4)
                    break
            else:
                steady_aim = 0
                drone.send_velocity_body(vx=0.0, vy=vy_cmd, vz_up=vz_cmd, yaw_rate=yaw_rate_cmd)

            if steady_aim == 0 and int(time.time() * 5) % 5 == 0:
                print(f"  [CENTERING ANGKA #{target_digit}] Box: [{bx},{by},{bw}x{bh}] | Pusat: ({tcx},{tcy}) | dx={dx:+3d}, dy={dy:+3d}")
        else:
            # Target sesaat hilang: hover tenang di tempat
            drone.send_velocity_body(vx=0.0, vy=0.0, vz_up=0.0, yaw_rate=0.0)

        time.sleep(0.03)


def tahap_3_kamikaze_strike(
    drone: DroneController,
    sensor: SensorListener,
    target_digit: int = 2
):
    """
    Tahap 3: Melaju lurus dengan kecepatan tinggi (vx = 7.0 m/s) menabrak
    titik tengah sasaran hingga impact fisik terdeteksi oleh LiDAR depan (< 0.45 m).
    """
    print("\n" + "=" * 65)
    print(f"🚀🚀 [MISI 3 - TAHAP 3] SPRINT KAMIKAZE MENABRAK TITIK TENGAH SASARAN #{target_digit}! 🚀🚀")
    print(">>> Kecepatan: vx = 7.0 m/s (Lurus Menuju Pusat Sasaran)")
    print("=" * 65)

    KECEPATAN_TABRAK = 7.0

    while rclpy.ok():
        drone.update()
        jarak_lidar = sensor.get_lidar_front()

        # Deteksi tabrakan fisik via sensor jarak LiDAR depan
        if jarak_lidar <= 0.45:
            print(f"\n💥💥💥 IMPACT TABRAKAN TERJADI TEPAT PADA TITIK TENGAH SASARAN! (LiDAR: {jarak_lidar:.2f} m) 💥💥💥")
            drone.send_velocity_body(vx=KECEPATAN_TABRAK, vy=0.0, vz_up=0.0, yaw_rate=0.0)
            time.sleep(0.2)
            break

        # Koreksi mikro garis lurus saat melesat agar tabrakan persis di tengah
        res_k = sensor.get_kamikaze()
        yaw_trim = 0.0
        vz_trim = 0.0

        if res_k is not None:
            matching = [t for t in res_k.all_targets if t.digit == target_digit]
            if res_k.locked_target and res_k.locked_target.digit == target_digit:
                matching.append(res_k.locked_target)
            if matching:
                t_cur = max(matching, key=lambda t: t.area)
                yaw_trim = float(np.clip(0.0030 * t_cur.dx, -0.6, 0.6))
                vz_trim = float(np.clip(-0.0030 * t_cur.dy, -0.4, 0.4))

        drone.send_velocity_body(vx=KECEPATAN_TABRAK, vy=0.0, vz_up=vz_trim, yaw_rate=yaw_trim)
        time.sleep(0.03)

    drone.brake(0.5)
    print("\n" + "=" * 65)
    print(f"🎉🎉 [MISI 3 SUKSES] DRONE TELAH MENABRAK TITIK TENGAH SASARAN #{target_digit}! 🎉🎉")
    print("=" * 65)


def misi(
    drone: Optional[DroneController] = None,
    sensor: Optional[SensorListener] = None,
    target_digit: int = 2
) -> Tuple[DroneController, SensorListener]:
    """
    Eksekusi Misi 3 secara terpadu.
    """
    if drone is None:
        drone = get_drone()
    if sensor is None:
        sensor = get_sensor(target_digit=target_digit)

    print("\n" + "=" * 65)
    print(f">>> [MISI 3] MEMULAI NAVIGASI & KAMIKAZE TARGET ANGKA #{target_digit}")
    print("=" * 65)

    drone.takeoff_if_needed(1.50)
    drone.brake(0.5)

    tahap_1_kunci_target_angka(drone, sensor, target_digit=target_digit)
    tahap_2_centering_titik_pusat_box(drone, sensor, target_digit=target_digit)
    tahap_3_kamikaze_strike(drone, sensor, target_digit=target_digit)

    return drone, sensor


def run_teleport_preset_5():
    """Helper teleport langsung ke titik awal Misi 3 (di seberang donat merah)."""
    teleport_py = os.path.join(SCRIPTS_DIR, "teleport.py")
    if os.path.exists(teleport_py):
        import subprocess
        x, y, z, yaw = (3.3, 18.5, 1.5, 90.0)
        print(f"[*] Teleporting ke posisi awal Misi 3: X={x}, Y={y}, Z={z}, Yaw={yaw}°...")
        subprocess.run([sys.executable, teleport_py, str(x), str(y), str(z), str(yaw)])
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Misi 3: Centering Titik Pusat Angka & Kamikaze Bulls Eye")
    parser.add_argument("--target", type=int, default=2, help="Nomor sasaran Bulls Eye (default: 2)")
    parser.add_argument("--teleport", action="store_true", help="Teleport ke posisi setelah donat merah sebelum mulai")
    args = parser.parse_args()

    if args.teleport:
        run_teleport_preset_5()

    drone = get_drone()
    sensor = get_sensor(target_digit=args.target)

    try:
        misi(drone=drone, sensor=sensor, target_digit=args.target)
    except KeyboardInterrupt:
        print("\n[!] Misi dihentikan oleh pengguna (Ctrl+C).")
        drone.brake(0.5)


if __name__ == '__main__':
    main()
