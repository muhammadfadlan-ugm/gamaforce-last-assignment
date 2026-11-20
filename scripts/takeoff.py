#!/usr/bin/env python3
"""
Script Takeoff Otomatis Drone Iris (ArduPilot SITL)
- Mengubah mode ke GUIDED
- Meng-ARM motor drone (Force Arm jika diperlukan)
- Melakukan Takeoff ke ketinggian target (default: 1.3 meter)
- Menunggu dan memantau ketinggian hingga stabil

Dapat dijalankan langsung:
  python3 scripts/takeoff.py
  python3 scripts/takeoff.py 1.5          # Takeoff ke ketinggian 1.5 meter

Atau di-import ke script lain (misi1.py, misi2.py, dll):
  from takeoff import Drone
  drone = Drone()
  drone.takeoff(1.3)
"""

import sys
import os
import time
import argparse

# Otomatis gunakan venv ArduPilot jika belum aktif
try:
    from pymavlink import mavutil
except ModuleNotFoundError:
    venv_py = os.path.expanduser('~/venv-ardupilot/bin/python3')
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise

DEFAULT_KONEKSI = 'udpin:127.0.0.1:14560'
DEFAULT_TINGGI = 1.3
mavlink = mavutil.mavlink


class Drone:
    def __init__(self, koneksi=DEFAULT_KONEKSI):
        print("=" * 55)
        print(f"[+] Menghubungkan ke SITL: {koneksi} ...")
        self.m = mavutil.mavlink_connection(koneksi)
        self.m.mav.heartbeat_send(0, 0, 0, 0, 0)

        print("[*] Menunggu heartbeat dari SITL...")
        self.m.wait_heartbeat(timeout=10)

        if self.m.target_system == 0:
            raise RuntimeError(
                "Gagal terhubung ke drone. Pastikan SITL berjalan dengan opsi --out 127.0.0.1:14560."
            )

        print(f"[OK] Terhubung ke Drone (System ID: {self.m.target_system})")
        self.d = 0.0
        self.armed = False
        self.dapat_posisi = False

        # Request data stream LOCAL_POSITION_NED (ID 32) pada 10 Hz
        self._minta_data(32, 10)

        print("[*] Menunggu data posisi EKF/GPS siap...")
        if not self._tunggu(lambda: self.dapat_posisi, 60, "data posisi"):
            raise RuntimeError(
                "Tidak ada data posisi. Pastikan status SITL berwarna HIJAU (EKF3 IMU origin set) sebelum takeoff."
            )
        print("[OK] EKF dan data posisi siap!")

    def _minta_data(self, msg_id, hz):
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0
        )

    def _update(self, tunggu=0.05):
        msg = self.m.recv_match(blocking=True, timeout=tunggu)
        while msg is not None:
            t = msg.get_type()
            if t == 'LOCAL_POSITION_NED':
                self.d = msg.z  # Dalam NED, z negatif adalah ketinggian ke atas
                self.dapat_posisi = True
            elif t == 'HEARTBEAT' and msg.get_srcSystem() == self.m.target_system \
                    and msg.type != mavlink.MAV_TYPE_GCS:
                self.armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            msg = self.m.recv_match(blocking=False)

    def _tunggu(self, kondisi, timeout, nama=''):
        t_akhir = time.time() + timeout
        while time.time() < t_akhir:
            self._update(0.05)
            if kondisi():
                return True
        return False

    def ganti_mode(self, mode):
        print(f"[*] Mengubah mode ke {mode}...")
        self.m.set_mode(mode)
        if self._tunggu(lambda: self.m.flightmode == mode, 10, 'mode ' + mode):
            print(f"[OK] Mode berubah ke {mode}")
        else:
            print(f"[!] Gagal mengubah mode ke {mode}")

    def arm(self):
        print("[*] Mencoba ARM motor...")
        for i in range(25):
            # Kirim perintah ARM (param2=21196 adalah force arm di ArduPilot)
            self.m.mav.command_long_send(
                self.m.target_system, self.m.target_component,
                mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1, 21196, 0, 0, 0, 0, 0
            )
            if self._tunggu(lambda: self.armed, 1, 'arm'):
                print("[OK] Drone berhasil di-ARM!")
                return True
            print(f"    Menunggu ARM ({i+1}/25)...")

        raise RuntimeError("Gagal ARM drone setelah 25 kali percobaan.")

    def takeoff(self, tinggi=DEFAULT_TINGGI):
        self.ganti_mode('GUIDED')
        self.arm()

        print(f"[*] Mengirim perintah Takeoff ke ketinggian {tinggi:.2f} meter...")
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, tinggi
        )

        # Pantau ketinggian naik
        print("[*] Drone sedang naik...")
        t_mulai = time.time()
        timeout = 25.0
        while time.time() - t_mulai < timeout:
            self._update(0.1)
            alt_sekarang = -self.d
            sys.stdout.write(f"\r    Ketinggian saat ini: {alt_sekarang:.2f} m / {tinggi:.2f} m")
            sys.stdout.flush()

            if alt_sekarang >= tinggi * 0.90:
                print(f"\n[OK] Drone stabil di ketinggian {alt_sekarang:.2f} meter!")
                return True

        print(f"\n[!] Peringatan: Batas waktu takeoff terlampaui ({alt_sekarang:.2f} m).")
        return False

    def land(self):
        print("\n[*] Mendaratkan drone...")
        self.ganti_mode('LAND')
        self._tunggu(lambda: (not self.armed) or (-self.d < 0.15), 30, 'mendarat')
        print("[OK] Drone telah mendarat.")


def main():
    parser = argparse.ArgumentParser(description="Script Takeoff Otomatis Drone Iris")
    parser.add_argument("tinggi", nargs="?", type=float, default=DEFAULT_TINGGI,
                        help=f"Ketinggian target takeoff dalam meter (default: {DEFAULT_TINGGI})")
    parser.add_argument("--koneksi", default=DEFAULT_KONEKSI,
                        help=f"String koneksi MAVLink (default: {DEFAULT_KONEKSI})")
    parser.add_argument("--land-on-exit", action="store_true",
                        help="Otomatis mendarat saat script dihentikan dengan Ctrl+C")

    args = parser.parse_args()

    drone = None
    try:
        drone = Drone(koneksi=args.koneksi)
        drone.takeoff(args.tinggi)

        print("\n" + "=" * 55)
        print(f"[OK] Takeoff selesai! Drone melayang di ~{args.tinggi:.2f} m.")
        print("     Tekan [ENTER] atau Ctrl+C jika ingin mendaratkan drone...")
        print("=" * 55)
        input()

    except KeyboardInterrupt:
        print("\n[!] Dihentikan oleh pengguna.")
    except Exception as e:
        print(f"\n[!] Error: {e}")
    finally:
        if drone and drone.armed:
            try:
                drone.land()
            except Exception:
                pass


if __name__ == '__main__':
    main()
