#!/usr/bin/env python3
"""
Script Teleportasi Drone (Iris) di Gazebo Sim
Dapat dijalankan dalam 2 mode:
1. Mode Interaktif (Menu CLI):
     python3 scripts/teleport.py
2. Mode Cepat (Argumen Baris Perintah):
     python3 scripts/teleport.py <X> <Y> <Z> [YAW_DERAJAT]
     Contoh: python3 scripts/teleport.py 3.3 12.0 1.5 90
"""

import sys
import math
import subprocess
import argparse

WORLD_NAME = "world_oprec"
MODEL_NAME = "iris"

PRESETS = {
    "1": ("Titik Awal (Spawn Point)", {"x": 0.0, "y": -15.0, "z": 0.2, "yaw": 90}),
    "2": ("Di Depan Track Obstacle & ArUco", {"x": -27.0, "y": -7.0, "z": 0.8, "yaw": 90}),
    "3": ("Di Dalam Lorong Tunnel", {"x": 13.0, "y": -0.1, "z": 1.2, "yaw": 90}),
    "4": ("Di Depan Donat Merah", {"x": 3.3, "y": 9.5, "z": 1.2, "yaw": 90}),
    "5": ("Di Atas Lingkaran Hijau (Centering Ring)", {"x": 3.3, "y": 12.0, "z": 1.5, "yaw": 90}),
    "6": ("Di Depan Target Bulls Eye (1, 2, 3)", {"x": -2.0, "y": 22.0, "z": 1.5, "yaw": 90}),
    "7": ("Di Tengah Arena Maze", {"x": 17.5, "y": 0.0, "z": 1.3, "yaw": 90}),
}


def teleport(x: float, y: float, z: float, yaw_deg: float = 90.0) -> bool:
    """Mengirim pesan set_pose ke Gazebo Sim via gz service."""
    # Konversi sudut yaw derajat ke orientasi quaternion (hanya rotasi sumbu Z)
    yaw_rad = math.radians(yaw_deg)
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)

    service_name = f"/world/{WORLD_NAME}/set_pose"
    req_payload = (
        f'name: "{MODEL_NAME}", '
        f'position: {{x: {x:.3f}, y: {y:.3f}, z: {z:.3f}}}, '
        f'orientation: {{x: 0, y: 0, z: {qz:.4f}, w: {qw:.4f}}}'
    )

    cmd = [
        "gz", "service", "-s", service_name,
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req", req_payload
    ]

    print(f"\n[+] Memindahkan drone ke -> X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f} | Yaw: {yaw_deg:.1f}° ...")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if "data: true" in res.stdout:
            print(f"[OK] Drone berhasil di-teleport ke ({x:.2f}, {y:.2f}, {z:.2f})!")
            return True
        else:
            print(f"[!] Gagal teleportasi. Pastikan Gazebo ({WORLD_NAME}) sedang berjalan.")
            if res.stderr:
                print(f"    Detail error: {res.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("[!] Perintah 'gz' tidak ditemukan. Pastikan ROS 2 / Gazebo environment sudah dimuat.")
        return False
    except subprocess.TimeoutExpired:
        print("[!] Timeout: Gazebo tidak merespons dalam 5 detik.")
        return False


def interactive_menu():
    while True:
        print("\n" + "=" * 55)
        print("         MENU TELEPORTASI MANUAL DRONE (GAZEBO)        ")
        print("=" * 55)
        for key, (label, c) in PRESETS.items():
            print(f" [{key}] {label:<40} -> ({c['x']}, {c['y']}, {c['z']})")
        print(" [8] Input Koordinat Bebas Kustom (X, Y, Z, Yaw)")
        print(" [0] Keluar")
        print("-" * 55)

        choice = input("Pilih lokasi (0-8): ").strip()
        if choice == "0":
            print("Keluar.")
            break
        elif choice in PRESETS:
            _, coords = PRESETS[choice]
            teleport(coords["x"], coords["y"], coords["z"], coords.get("yaw", 90.0))
        elif choice == "8":
            try:
                x = float(input("  Masukkan X : "))
                y = float(input("  Masukkan Y : "))
                z = float(input("  Masukkan Z (ketinggian, misal 1.3) : "))
                yaw_in = input("  Masukkan Yaw derajat (Enter untuk default 90): ").strip()
                yaw = float(yaw_in) if yaw_in else 90.0
                teleport(x, y, z, yaw)
            except ValueError:
                print("[!] Error: Masukan harus berupa angka.")
        else:
            print("[!] Pilihan tidak valid.")


def main():
    parser = argparse.ArgumentParser(
        description="Script Teleportasi Drone Iris di Gazebo",
        epilog="Contoh: python3 scripts/teleport.py 3.3 12.0 1.5 90"
    )
    parser.add_argument("coords", nargs="*", type=float, help="Koordinat: X Y Z [YAW]")
    parser.add_argument("-x", type=float, help="Posisi X")
    parser.add_argument("-y", type=float, help="Posisi Y")
    parser.add_argument("-z", type=float, help="Posisi Z (ketinggian)")
    parser.add_argument("--yaw", type=float, default=90.0, help="Orientasi Yaw dalam derajat (default: 90)")

    args = parser.parse_args()

    # 1. Jika koordinat diberikan lewat argumen posisional: python3 teleport.py 3.3 12.0 1.5 [90]
    if len(args.coords) >= 3:
        x = args.coords[0]
        y = args.coords[1]
        z = args.coords[2]
        yaw = args.coords[3] if len(args.coords) >= 4 else 90.0
        teleport(x, y, z, yaw)

    # 2. Jika koordinat diberikan lewat flag: python3 teleport.py -x 3.3 -y 12.0 -z 1.5
    elif args.x is not None and args.y is not None and args.z is not None:
        teleport(args.x, args.y, args.z, args.yaw)

    # 3. Tanpa argumen -> Buka menu interaktif
    else:
        interactive_menu()


if __name__ == '__main__':
    main()
