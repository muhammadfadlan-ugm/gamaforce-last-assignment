#!/usr/bin/env python3
"""
================================================================================
GAMAFORCE CORE - DRONE CONTROLLER (AUTOPILOT MAVLINK ABSTRACTION)
================================================================================
Modul kontrol tunggal untuk drone Iris berbasis pymavlink.
Menyediakan antarmuka lengkap untuk:
- Arming, Takeoff, Landing
- Kontrol kecepatan Body Frame & Global NED
- Kontrol posisi presisi berbasis EKF P-Controller
- Pengereman aktif (brake) & penyesuaian sudut hadap (yaw)
- Thread-safe dan mendukung shared singleton instance antar-misi.
================================================================================
"""

import sys
import os
import time
import math
from typing import Optional

# Otomatis gunakan venv ArduPilot jika modul pymavlink belum terinstal di system
try:
    from pymavlink import mavutil
except ModuleNotFoundError:
    venv_py = os.path.expanduser('~/venv-ardupilot/bin/python3')
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise

mavlink = mavutil.mavlink
DEFAULT_KONEKSI = 'udpin:127.0.0.1:14560'
TINGGI_DEFAULT = 1.35


class DroneController:
    """
    Kontroler navigasi drone otonom terpadu untuk seluruh rangkaian misi.
    """
    def __init__(self, koneksi: str = DEFAULT_KONEKSI):
        print("=" * 60)
        print(f"[+] Menghubungkan ke SITL Autopilot: {koneksi} ...")
        self.m = mavutil.mavlink_connection(koneksi)
        self.m.mav.heartbeat_send(0, 0, 0, 0, 0)

        print("[*] Menunggu heartbeat autopilot...")
        self.m.wait_heartbeat(timeout=10)
        if self.m.target_system == 0:
            raise RuntimeError(
                "Gagal terhubung ke autopilot! Pastikan SITL berjalan dengan opsi --out 127.0.0.1:14560."
            )

        print(f"[OK] Terhubung ke Drone Iris (System ID: {self.m.target_system})")
        self.pos_x: float = 0.0      # North (m)
        self.pos_y: float = 0.0      # East (m)
        self.pos_z: float = 0.0      # Down (m) -> Altitude = -pos_z
        self.yaw_deg: float = 90.0
        self.yaw_referensi_deg: float = 90.0
        self.armed: bool = False
        self.flightmode: str = "UNKNOWN"

        # Minta stream data telemetri: LOCAL_POSITION_NED (32) dan ATTITUDE (30) pada 25 Hz
        self._minta_stream(32, 25)
        self._minta_stream(30, 25)
        time.sleep(0.5)
        self.update()

    def _minta_stream(self, msg_id: int, hz: int):
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0
        )

    def update(self):
        """Membaca telemetri drone non-blocking."""
        msg = self.m.recv_match(blocking=False)
        while msg is not None:
            t = msg.get_type()
            if t == 'LOCAL_POSITION_NED':
                self.pos_x = msg.x
                self.pos_y = msg.y
                self.pos_z = msg.z
            elif t == 'ATTITUDE':
                deg = math.degrees(msg.yaw)
                if deg < 0:
                    deg += 360.0
                self.yaw_deg = deg
            elif t == 'HEARTBEAT' and msg.get_srcSystem() == self.m.target_system:
                self.armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.flightmode = self.m.flightmode
            msg = self.m.recv_match(blocking=False)

    @property
    def altitude(self) -> float:
        """Ketinggian saat ini di atas tanah (meter)."""
        return -self.pos_z

    def set_mode(self, mode: str, timeout: float = 5.0) -> bool:
        """Mengganti flight mode drone (e.g. 'GUIDED', 'LAND')."""
        self.m.set_mode(mode)
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.update()
            if self.m.flightmode == mode:
                return True
            time.sleep(0.05)
        return False

    def takeoff(self, tinggi: float = TINGGI_DEFAULT):
        """Takeoff ke ketinggian target."""
        self.takeoff_if_needed(tinggi)

    def takeoff_if_needed(self, tinggi: float = TINGGI_DEFAULT):
        """Takeoff jika drone belum terbang atau masih di permukaan tanah."""
        self.update()
        if not self.armed or (-self.pos_z < 0.35):
            print(f"[*] Drone belum di udara. Takeoff ke {tinggi:.2f} m...")
            self.set_mode('GUIDED')
            for _ in range(10):
                self.m.mav.command_long_send(
                    self.m.target_system, self.m.target_component,
                    mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                    1, 21196, 0, 0, 0, 0, 0
                )
                time.sleep(0.3)
                self.update()
                if self.armed:
                    break

            self.m.mav.command_long_send(
                self.m.target_system, self.m.target_component,
                mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                0, 0, 0, 0, 0, 0, tinggi
            )

            while -self.pos_z < tinggi * 0.85:
                self.update()
                time.sleep(0.1)
            print(f"[OK] Ketinggian take off tercapai ({-self.pos_z:.2f} m).")
            self.yaw_referensi_deg = self.yaw_deg
            self.brake(0.8)

    def send_velocity_body(self, vx: float, vy: float, vz_up: float, yaw_rate: float = 0.0):
        """
        Mengirim kecepatan translasi BODY FRAME:
        vx > 0: Maju,      vx < 0: Mundur
        vy > 0: Kanan,     vy < 0: Kiri
        vz_up > 0: Naik,   vz_up < 0: Turun
        """
        vz_ned = -vz_up
        type_mask = 0b010111000111
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            mavlink.MAV_FRAME_BODY_OFFSET_NED,
            type_mask,
            0, 0, 0,
            vx, vy, vz_ned,
            0, 0, 0,
            0, yaw_rate
        )

    def move_timed(self, vx: float, vy: float, vz_up: float, detik: float, yaw_rate: float = 0.0):
        """Menggerakkan drone dengan kecepatan body selama durasi detik tertentu."""
        t_end = time.time() + detik
        while time.time() < t_end:
            self.send_velocity_body(vx, vy, vz_up, yaw_rate)
            self.update()
            time.sleep(0.04)
        self.send_velocity_body(0, 0, 0, 0)

    def move_at_vel(self, vx: float, vy: float, vz_up: float, detik: float):
        """Alias kompatibilitas untuk move_timed."""
        self.move_timed(vx, vy, vz_up, detik)

    def brake(self, detik: float = 0.4):
        """Menghentikan drone secara aktif di posisi saat ini."""
        self.move_timed(0, 0, 0, detik)

    def rem_aktif(self):
        """Alias kompatibilitas untuk brake."""
        self.brake(0.5)

    def gerak_relatif_presisi(
        self,
        maju_m: float = 0.0,
        kanan_m: float = 0.0,
        naik_m: float = 0.0,
        kecepatan_maks: float = 1.0,
        toleransi: float = 0.08
    ):
        """
        Menggerakkan drone ke titik setpoint koordinat EKF lokal tertentu
        dengan pengereman P-Controller dan pengunci sudut yaw.
        """
        self.update()
        yaw_kunci_rad = math.radians(self.yaw_deg)

        target_n = self.pos_x + (maju_m * math.cos(yaw_kunci_rad) - kanan_m * math.sin(yaw_kunci_rad))
        target_e = self.pos_y + (maju_m * math.sin(yaw_kunci_rad) + kanan_m * math.cos(yaw_kunci_rad))
        target_d = self.pos_z - naik_m

        print(f"[*] Setpoint EKF (Maju:{maju_m}m, Kanan:{kanan_m}m) -> Target NED({target_n:.2f}, {target_e:.2f}, {target_d:.2f})")

        dt = 0.033
        kp_xy = 1.2
        kp_z = 1.5

        while True:
            self.update()
            err_n = target_n - self.pos_x
            err_e = target_e - self.pos_y
            err_d = target_d - self.pos_z

            jarak_sisa = math.sqrt(err_n**2 + err_e**2 + err_d**2)
            if jarak_sisa <= toleransi:
                break

            vn = err_n * kp_xy
            ve = err_e * kp_xy
            vd = max(-0.7, min(0.7, err_d * kp_z))

            kecepatan_xy = math.hypot(vn, ve)
            if kecepatan_xy > kecepatan_maks:
                skala = kecepatan_maks / kecepatan_xy
                vn *= skala
                ve *= skala

            type_mask = 0b0000101111000111
            self.m.mav.set_position_target_local_ned_send(
                0, self.m.target_system, self.m.target_component,
                mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                0, 0, 0,
                vn, ve, vd,
                0, 0, 0,
                yaw_kunci_rad, 0
            )
            time.sleep(dt)

        self.brake(0.5)
        print(f"[OK] Tiba di setpoint! NED({self.pos_x:.2f}, {self.pos_y:.2f}, {self.pos_z:.2f}) | Sisa: {jarak_sisa:.3f}m")

    def set_yaw(self, target_yaw_deg: float, speed_deg_s: float = 75.0, timeout: float = 3.0):
        """Memutar hadap yaw drone ke sudut target secara aman."""
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavlink.MAV_CMD_CONDITION_YAW,
            0,
            target_yaw_deg, speed_deg_s, 0, 0, 0, 0, 0
        )
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.update()
            err = (target_yaw_deg - self.yaw_deg + 180.0) % 360.0 - 180.0
            if abs(err) <= 5.0:
                break
            time.sleep(0.05)
        self.brake(0.2)

    def land(self):
        """Mendaratkan drone secara aman."""
        print("[*] Mendaratkan drone...")
        self.set_mode('LAND')
        t_end = time.time() + 30.0
        while time.time() < t_end:
            self.update()
            if not self.armed or (-self.pos_z < 0.15):
                break
            time.sleep(0.2)
        print("[OK] Drone telah mendarat.")

    def mendarat(self):
        """Alias kompatibilitas untuk land."""
        self.land()


# ------------------------------------------------------------------------------
# SHARED INSTANCE HELPER (SINGLETON)
# ------------------------------------------------------------------------------
_global_drone: Optional[DroneController] = None

def get_drone(koneksi: str = DEFAULT_KONEKSI) -> DroneController:
    """Mengembalikan instance DroneController tunggal bersama."""
    global _global_drone
    if _global_drone is None:
        _global_drone = DroneController(koneksi=koneksi)
    return _global_drone

