
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
    "8": ("misi kanan ketiga", {"x": 3.3, "y": 18.0, "z": 1.3, "yaw": 90})
}


def is_gazebo_running() -> bool:
    """Mengecek apakah proses Gazebo Sim sedang aktif di sistem secara instan."""
    try:
        res = subprocess.run(["pgrep", "-f", "gz sim"], capture_output=True, text=True, timeout=1)
        return len(res.stdout.strip()) > 0
    except Exception:
        return True


def teleport(x: float, y: float, z: float, yaw_deg: float = 90.0) -> bool:
    """Mengirim pesan set_pose ke Gazebo Sim via gz service."""
    # 1. Cek instan apakah proses Gazebo berjalan
    if not is_gazebo_running():
        print(f"\n[!] ERROR: Proses Gazebo Sim tidak terdeteksi berjalan di sistem!")
        print("    -> Jalankan Terminal 1 terlebih dahulu: ./terminal1.sh (atau ./run_all.sh)")
        return False

    print(f"\n[+] Memindahkan drone ke -> X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f} | Yaw: {yaw_deg:.1f}° ...")

    # 2. Konversi sudut yaw derajat ke orientasi quaternion (rotasi sumbu Z)
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
        "--timeout", "5000",
        "--req", req_payload
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if "data: true" in res.stdout:
            print(f"[OK] Drone berhasil di-teleport ke ({x:.2f}, {y:.2f}, {z:.2f})!")
            return True
        else:
            print(f"[!] Gagal teleportasi:\n{res.stderr or res.stdout}")
            return False
    except FileNotFoundError:
        print("[!] Perintah 'gz' tidak ditemukan. Pastikan ROS 2 / Gazebo environment sudah dimuat:")
        print("    source /opt/ros/jazzy/setup.bash")
        return False
    except subprocess.TimeoutExpired:
        print("[!] Timeout: Gazebo tidak merespons dalam 8 detik. Pastikan simulasi tidak dalam keadaan pause.")
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

        try:
            choice = input("Pilih lokasi (0-8): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nKeluar.")
            break

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
            except (KeyboardInterrupt, EOFError):
                print("\nBatal input.")
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

    # 1. Jika koordinat diberikan lewat argumen posisional
    if len(args.coords) >= 3:
        x = args.coords[0]
        y = args.coords[1]
        z = args.coords[2]
        yaw = args.coords[3] if len(args.coords) >= 4 else 90.0
        teleport(x, y, z, yaw)

    # 2. Jika koordinat diberikan lewat flag
    elif args.x is not None and args.y is not None and args.z is not None:
        teleport(args.x, args.y, args.z, args.yaw)

    # 3. Tanpa argumen -> Buka menu interaktif
    else:
        try:
            interactive_menu()
        except KeyboardInterrupt:
            print("\nKeluar.")


if __name__ == '__main__':
    main()
