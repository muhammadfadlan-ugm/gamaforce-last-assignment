#!/usr/bin/env python3
"""
================================================================================
GAMAFORCE - MAIN EXECUTIVE SCRIPT (FULL PIPELINE: MISI 1 -> MISI 2 -> MISI 3)
================================================================================
Program ringkas pengorkestrasi seluruh rangkaian misi:
  - Misi 1: Navigasi Labirin & Terowongan (LiDAR 1D + EKF)
  - Misi 2: Pad Hijau & Menembus Donat Merah (Dual-Camera Servoing + LiDAR)
  - Misi 3: Centering Titik Pusat Angka #2 & Serangan Kamikaze (7.0 m/s)
================================================================================
"""

import sys
import os

# Otomatis gunakan venv ArduPilot jika modul pymavlink belum terinstal di system
try:
    import pymavlink
except ModuleNotFoundError:
    venv_py = os.path.expanduser('~/venv-ardupilot/bin/python3')
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise

# Pastikan path modul scripts dan scripts_kanan terdeteksi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in (CURRENT_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import misi1
import misi2
import misi3


def main():
    print("\n" + "=" * 65)
    print(">>> [MAIN] MEMULAI EKSEKUSI PIPELINE: MISI 1 -> MISI 2 -> MISI 3")
    print("=" * 65)

    # 1. Eksekusi Misi 1: Labirin & Terowongan (auto_land=False agar lanjut ke Misi 2)
    drone, sensor = misi1.misi(auto_land=False)

    # 2. Eksekusi Misi 2: Pad Hijau & Menembus Donat Merah
    misi2.misi(drone=drone, sensor=sensor)

    # 3. Eksekusi Misi 3: Centering Titik Pusat Box Angka #2 & Serangan Kamikaze
    misi3.misi(drone=drone, sensor=sensor, target_digit=2)

    print("\n" + "=" * 65)
    print("🏆🏆 [MAIN] SELURUH RANGKAIAN MISI 1, 2, DAN 3 SUKSES PENUH! 🏆🏆")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    main()
