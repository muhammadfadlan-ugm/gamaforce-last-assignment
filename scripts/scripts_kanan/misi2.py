#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MISI 2: CENTERING DI PAD HIJAU + TERBANG LEWAT RING (GATE NODE)
================================================================================
Implementasi berbasis mentahan algoritma: scripts/centering and entering ring

Alur Misi:
1. SETUP & CLIMB: Menyelaraskan ketinggian jelajah (cruise_alt) & yaw menghadap utara (NORTH).
2. GOTO: Bergerak dari titik serah-terima menuju ENTRY (jalur kanan: 6.3, 12.0) lalu menuju PAD_XY (3.30, 12.00).
3. CENTER_PAD: Visual servoing kamera bawah (ray-casting ke Z=PAD_Z) mengunci pusat pad hijau.
4. ALIGN_RING: Visual servoing kamera depan (ray-casting ke Y=RING_Y) menyelaraskan X dan Z dengan pusat ring.
5. APPROACH_RING & COMMIT: Bergerak maju mendekati ring, mengunci X-Z pada jarak commit, dan menembus ring.
6. EXIT & DONE: Melewati ring hingga jarak aman di belakang bidang ring, siap serah-terima ke Misi 3.

Jalankan Standalone (ROS 2):
    python3 scripts/scripts_kanan/misi2.py --ros-args -p scenario:=right
    ros2 service call /gate_node/start std_srvs/srv/Trigger
Atau dipanggil otomatis dari main.py:
    misi2.misi(drone=drone, sensor=sensor, scenario='right')
================================================================================
"""

import sys
import os
import math
import time
from collections import namedtuple
from typing import Optional, Tuple, Any

import cv2
import numpy as np

# Otomatis gunakan venv ArduPilot jika modul pymavlink/rclpy belum terinstal di system
try:
    import pymavlink
except ModuleNotFoundError:
    venv_py = os.path.expanduser('~/venv-ardupilot/bin/python3')
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

# Setup path untuk modul scripts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)
for p in (SCRIPTS_DIR, CURRENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Integrasi opsional ke Web Dashboard baca_semua_data.py
try:
    from baca_semua_data import data_hub
except Exception:
    data_hub = None


# =============================================================================
# 1. FAKTA ARENA  (diukur dari mesh STL/SDF di repo; frame dunia Gazebo:
#    x = timur, y = utara, z = atas; drone spawn di (0,-15) menghadap utara)
# =============================================================================
PAD_XY = np.array([3.30, 12.00])           # pusat pad hijau (piringan datar, diameter 2.0 m)
PAD_Z = 0.045                              # tinggi permukaan pad
RING_X, RING_Y, RING_Z = 3.301, 16.49, 2.595   # pusat ring; bidang ring tegak lurus sumbu y
SPAWN = np.array([0.0, -15.0, 0.195])      # pose awal drone di world file
NORTH = math.pi / 2.0                      # yaw ENU: 0 = timur, +90 deg = utara

# kotak larangan (xmin, xmax, ymin, ymax) dari bounds mesh collision
KEEPOUT = {
    'maze':         (7.41, 14.46, -12.45, 1.85),
    'tunnel':       (8.08, 13.74, 0.80, 8.85),
    'track_vision': (-21.90, -2.91, -9.00, 11.06),
}
# titik masuk: titik aman di sisi pad tempat drone datang, sebelum lurus ke pad
ENTRY = {'test': (3.3, 7.0), 'right': (6.3, 12.0), 'left': (0.3, 12.0)}
# PERKIRAAN area serah-terima (xmin, xmax, ymin, ymax); hanya untuk peringatan
HANDOFF = {
    'test': (-1.5, 1.5, -16.5, -13.5),
    'right': (6.0, 15.0, 8.8, 20.0),
    'left': (-12.0, 1.0, 11.0, 20.0),
}
# posisi kamera di frame body drone (x depan, y kiri, z atas); dari model.sdf
CAM_OFF = {
    'front': np.array([0.12, 0.0, 0.02]),
    'down': np.array([0.0, 0.0, -0.08]),
}


# =============================================================================
# 2. FUNGSI MURNI (geometri + vision), tanpa ROS, mudah diuji terpisah
# =============================================================================
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clip(x, lim):
    return max(-lim, min(lim, x))


def clip_norm(v, vmax):
    n = float(np.linalg.norm(v))
    return v if n <= vmax else v * (vmax / n)


def quat_to_rot(x, y, z, w):
    """Kuaternion body->dunia menjadi matriks rotasi 3x3."""
    s = 2.0 / (x * x + y * y + z * z + w * w)
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]
    ])


def pixel_ray_world(cam, u, v, width, height, hfov, R_wb, p_wb):
    """Piksel (u,v) -> (titik asal kamera, arah sinar) di frame dunia.
    front: sumbu optik = depan drone.  down: kamera di-pitch +90 deg, jadi
    atas gambar = depan drone dan kanan gambar = kanan drone."""
    f = (width / 2.0) / math.tan(hfov / 2.0)        # fokus dalam piksel (piksel persegi)
    a = (u - (width - 1) / 2.0) / f                  # + = kanan gambar
    b = (v - (height - 1) / 2.0) / f                 # + = bawah gambar
    if cam == 'front':
        d_body = np.array([1.0, -a, -b])
    else:
        d_body = np.array([-b, -a, -1.0])
    return p_wb + R_wb @ CAM_OFF[cam], R_wb @ d_body


def hit_plane_z(o, d, z):
    if d[2] > -1e-3:
        return None
    t = (z - o[2]) / d[2]
    return o + t * d if t > 0 else None


def hit_plane_y(o, d, y):
    if d[1] < 1e-3:
        return None
    t = (y - o[1]) / d[1]
    return o + t * d if t > 0 else None


def seg_hits_box(p0, p1, box, margin):
    """Segmen 2D p0->p1 menyentuh kotak (sudah digelembungkan 'margin')? Metode slab."""
    lo = (box[0] - margin, box[2] - margin)
    hi = (box[1] + margin, box[3] + margin)
    t0, t1 = 0.0, 1.0
    for i in range(2):
        d = p1[i] - p0[i]
        if abs(d) < 1e-9:
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return False
        else:
            ta, tb = (lo[i] - p0[i]) / d, (hi[i] - p0[i]) / d
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return False
    return True


Blob = namedtuple('Blob', 'cx cy area bbox touches hole')
_K3 = np.ones((3, 3), np.uint8)
_K7 = np.ones((7, 7), np.uint8)


def detect_blob(rgb, ranges, min_area):
    """Cari gumpalan warna terbesar. ranges = [(hsv_lo, hsv_hi), ...] (skala OpenCV)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K3)     # buang bintik
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K7)    # tambal lubang kecil/sambung serpihan
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, mask
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = int(st[k, cv2.CC_STAT_AREA])
    if area < min_area:
        return None, mask
    x, y = int(st[k, cv2.CC_STAT_LEFT]), int(st[k, cv2.CC_STAT_TOP])
    w, h = int(st[k, cv2.CC_STAT_WIDTH]), int(st[k, cv2.CC_STAT_HEIGHT])
    H, W = mask.shape
    touches = x <= 0 or y <= 0 or x + w >= W or y + h >= H
    # lubang di dalam gumpalan (untuk ring: pusat lubang tidak bias walau tepi luar terpotong frame)
    hole = None
    cnts, hier = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is not None:
        best = 0.0
        for c, hh in zip(cnts, hier[0]):
            if hh[3] != -1:                                  # punya induk -> kontur lubang
                m = cv2.moments(c)
                if m['m00'] > best and m['m00'] > 0.02 * area:
                    best = m['m00']
                    hole = (m['m10'] / m['m00'], m['m01'] / m['m00'], best)
    return Blob(float(cen[k][0]), float(cen[k][1]), area, (x, y, w, h), touches, hole), mask


# =============================================================================
# 3. NODE ROS 2 (GATE NODE)
# =============================================================================
class GateNode(Node):
    def __init__(self):
        super().__init__('gate_node')
        d = self.declare_parameter
        d('scenario', 'right')        # default jalur kanan (setelah maze + terowongan)
        d('cruise_alt', 3.2)          # tinggi terbang menuju pad (m, frame dunia)
        d('v_cruise', 1.0)            # kecepatan menuju titik masuk (m/s)
        d('v_final', 0.5)             # kecepatan dari titik masuk ke pad
        d('v_fine', 0.35)             # batas kecepatan koreksi halus
        d('v_ring', 0.35)             # kecepatan maju menuju/menembus ring
        d('k_xy', 0.7)                # gain P posisi -> kecepatan (1/s)
        d('k_z', 0.8)
        d('pad_tol', 0.12)            # toleransi centering pad (m)
        d('ring_tol', 0.10)           # toleransi sejajar ring (m)
        d('stable_frames', 15)        # jumlah frame kamera berturut-turut yang harus "lolos"
        d('commit_dist', 2.0)         # di bawah jarak ini koreksi visual dibekukan (m)
        d('exit_dist', 2.6)           # hover sejauh ini di belakang bidang ring (m)
        d('lost_timeout', 4.0)        # detik tanpa deteksi sebelum pakai koordinat nominal
        d('allow_fallback', True)
        d('plaus_radius', 2.0)        # tolak deteksi yang > ini dari posisi nominal (m); 0 = mati
        d('check_keepout', True)
        d('keepout_margin', 0.4)
        d('hold_after_done', True)    # False: berhenti publish setelah selesai (serah ke node lain)
        d('off_x', 0.0)               # dunia = lokal MAVROS + off (jika frame lokal tidak sejajar dunia)
        d('off_y', 0.0)
        d('off_z', 0.0)
        d('hfov', 1.047)
        d('pad_hsv_lo', [45, 100, 50])
        d('pad_hsv_hi', [75, 255, 255])
        d('ring_hsv_lo1', [0, 120, 50])
        d('ring_hsv_hi1', [10, 255, 255])
        d('ring_hsv_lo2', [170, 120, 50])
        d('ring_hsv_hi2', [179, 255, 255])
        d('min_pad_area', 300)
        d('min_ring_area', 800)
        d('debug_dir', '')            # isi path untuk menyimpan PNG deteksi (mis. /ros2_ws/gate_debug)
        d('debug_every', 15)
        g = lambda k: self.get_parameter(k).value
        self.scenario = g('scenario')
        self.cruise_alt, self.v_cruise, self.v_final = g('cruise_alt'), g('v_cruise'), g('v_final')
        self.v_fine, self.v_ring = g('v_fine'), g('v_ring')
        self.k_xy, self.k_z = g('k_xy'), g('k_z')
        self.pad_tol, self.ring_tol, self.stable_frames = g('pad_tol'), g('ring_tol'), g('stable_frames')
        self.commit_dist, self.exit_dist = g('commit_dist'), g('exit_dist')
        self.lost_timeout, self.allow_fallback = g('lost_timeout'), g('allow_fallback')
        self.plaus, self.check_keepout, self.margin = g('plaus_radius'), g('check_keepout'), g('keepout_margin')
        self.hold_after_done = g('hold_after_done')
        self.off = np.array([g('off_x'), g('off_y'), g('off_z')], float)
        self.hfov = g('hfov')
        ints = lambda k: [int(v) for v in g(k)]
        self.pad_ranges = [(ints('pad_hsv_lo'), ints('pad_hsv_hi'))]
        self.ring_ranges = [(ints('ring_hsv_lo1'), ints('ring_hsv_hi1')), (ints('ring_hsv_lo2'), ints('ring_hsv_hi2'))]
        self.min_pad_area, self.min_ring_area = g('min_pad_area'), g('min_ring_area')
        self.debug_dir, self.debug_every = g('debug_dir'), max(1, int(g('debug_every')))

        # ---- state internal
        self.phase = 'WAIT_START'
        self.p_l = None            # posisi lokal MAVROS
        self.R = None              # rotasi body->dunia
        self.yaw = 0.0
        self.pose_t = 0.0
        self.state = None
        self.wps, self.wp_i = [], 0
        self.seq, self.fut, self.fut_tag, self.fut_t0 = 'MODE', None, '', 0.0
        self.next_try, self.fails, self.on_ground, self.z_rest = 0.0, 0, True, 0.0
        self.pad_est, self.pad_t, self.pad_ok = None, 0.0, 0
        self.ring_est, self.ring_t, self.ring_ok = None, 0.0, 0
        self.lost_since, self.fb_ok, self.stable = None, 0, 0
        self.hold_xz, self.z_hold = None, 0.0
        self.dbg_n, self.warned_enc = 0, False

        # ---- ROS I/O
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.on_pose, qos_profile_sensor_data)
        self.create_subscription(State, '/mavros/state', self.on_state,
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Image, '/camera_down', self.on_down, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera_front', self.on_front, qos_profile_sensor_data)
        self.sp_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', qos_profile_sensor_data)
        self.cli_mode = self.create_client(SetMode, '/mavros/set_mode')
        self.cli_arm = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.cli_tko = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        self.create_service(Trigger, '~/start', self.srv_start)
        self.create_service(Trigger, '~/abort', self.srv_abort)
        self.create_timer(0.05, self.tick)      # 20 Hz (waktu dinding)
        self.log(f'GateNode siap. scenario={self.scenario}. Mulai via service ~/start atau otomatis.')

    # ------------------------------------------------------------------ util
    def log(self, s):
        self.get_logger().info(s)

    def now(self):
        return time.monotonic()

    @property
    def pw(self):
        """Posisi drone di frame dunia."""
        return self.p_l + self.off

    def cam_pos_w(self, cam):
        return self.pw + self.R @ CAM_OFF[cam]

    def fail(self, msg):
        self.get_logger().error(msg)
        self.z_hold = float(self.pw[2]) if self.p_l is not None else 0.0
        self.phase = 'ABORT'

    # ------------------------------------------------------------- callbacks
    def on_pose(self, m):
        p, q = m.pose.position, m.pose.orientation
        self.p_l = np.array([p.x, p.y, p.z])
        self.R = quat_to_rot(q.x, q.y, q.z, q.w)
        self.yaw = math.atan2(self.R[1, 0], self.R[0, 0])
        self.pose_t = self.now()

    def on_state(self, m):
        self.state = m

    def to_rgb(self, msg):
        if msg.encoding not in ('rgb8', 'bgr8'):
            if not self.warned_enc:
                self.get_logger().error(f'encoding gambar {msg.encoding} tidak didukung')
                self.warned_enc = True
            return None
        row = msg.width * 3
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :row]
        img = a.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)

    def dump(self, tag, img, mask, blob):
        # Update Web Dashboard buffer jika aktif
        if data_hub is not None:
            try:
                vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                vis[mask > 0] = (255, 0, 255)
                if blob is not None:
                    cv2.circle(vis, (int(blob.cx), int(blob.cy)), 6, (255, 255, 255), -1)
                ret, jpeg = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    with data_hub.lock:
                        if tag == 'pad':
                            data_hub.jpeg_down = jpeg.tobytes()
                            data_hub.has_down = True
                        elif tag == 'ring':
                            data_hub.jpeg_front = jpeg.tobytes()
                            data_hub.has_front = True
            except Exception:
                pass

        if not self.debug_dir:
            return
        self.dbg_n += 1
        if self.dbg_n % self.debug_every or self.dbg_n > 400 * self.debug_every:
            return
        os.makedirs(self.debug_dir, exist_ok=True)
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vis[mask > 0] = (255, 0, 255)
        if blob is not None:
            cv2.circle(vis, (int(blob.cx), int(blob.cy)), 6, (255, 255, 255), -1)
        cv2.imwrite(os.path.join(self.debug_dir, f'{tag}_{self.dbg_n:05d}.png'), vis)

    def on_down(self, msg):
        """Kamera bawah -> estimasi pusat pad di frame dunia."""
        if self.R is None or self.phase not in ('GOTO', 'CENTER_PAD'):
            return
        if self.phase == 'GOTO' and self.wp_i < len(self.wps) - 1:
            return                                   # belum di leg terakhir: abaikan
        img = self.to_rgb(msg)
        if img is None:
            return
        blob, mask = detect_blob(img, self.pad_ranges, self.min_pad_area)
        self.dump('pad', img, mask, blob)
        if blob is None:
            self.pad_ok = 0
            return
        o, dvec = pixel_ray_world('down', blob.cx, blob.cy, msg.width, msg.height, self.hfov, self.R, self.pw)
        hit = hit_plane_z(o, dvec, PAD_Z)
        if hit is None:
            return
        est = hit[:2]
        if self.plaus > 0 and np.linalg.norm(est - PAD_XY) > self.plaus:
            self.get_logger().warn(f'blob hijau {np.round(est, 2)} terlalu jauh dari pad nominal {PAD_XY}; diabaikan '
                                   '(bukan pad? atau off_x/off_y salah?)', throttle_duration_sec=2.0)
            self.pad_ok = 0
            return
        t = self.now()
        fresh = self.pad_est is not None and t - self.pad_t < 1.0
        self.pad_est = 0.6 * self.pad_est + 0.4 * est if fresh else est
        self.pad_t = t
        err = float(np.linalg.norm(self.pad_est - self.pw[:2]))
        self.pad_ok = self.pad_ok + 1 if err < self.pad_tol else 0

    def on_front(self, msg):
        """Kamera depan -> estimasi pusat ring (x, z) pada bidang ring, frame dunia."""
        if self.R is None or self.phase not in ('ALIGN_RING', 'APPROACH_RING'):
            return
        img = self.to_rgb(msg)
        if img is None:
            return
        blob, mask = detect_blob(img, self.ring_ranges, self.min_ring_area)
        self.dump('ring', img, mask, blob)
        if blob is None:
            self.ring_ok = 0
            return
        u, v = (blob.hole[0], blob.hole[1]) if blob.hole else (blob.cx, blob.cy)   # lubang ring > centroid
        o, dvec = pixel_ray_world('front', u, v, msg.width, msg.height, self.hfov, self.R, self.pw)
        hit = hit_plane_y(o, dvec, RING_Y)
        if hit is None:
            return
        est = np.array([hit[0], hit[2]])
        if self.plaus > 0 and np.linalg.norm(est - np.array([RING_X, RING_Z])) > self.plaus:
            self.get_logger().warn(f'blob merah {np.round(est, 2)} terlalu jauh dari ring nominal; diabaikan',
                                   throttle_duration_sec=2.0)
            self.ring_ok = 0
            return
        t = self.now()
        fresh = self.ring_est is not None and t - self.ring_t < 1.0
        self.ring_est = 0.6 * self.ring_est + 0.4 * est if fresh else est
        self.ring_t = t
        err = float(np.linalg.norm(self.ring_est - np.array([self.pw[0], self.pw[2]])))
        self.ring_ok = self.ring_ok + 1 if err < self.ring_tol else 0

    # --------------------------------------------------------------- services
    def srv_start(self, req, res):
        if self.phase != 'WAIT_START':
            res.success, res.message = False, f'sudah berjalan (phase={self.phase})'
            return res
        res.success, res.message = self.prepare()
        self.log(res.message)
        return res

    def srv_abort(self, req, res):
        if self.phase == 'WAIT_START':
            res.success, res.message = False, 'misi belum dimulai, tidak ada yang di-abort'
            return res
        self.fail('ABORT diminta lewat service: hover di tempat')
        res.success, res.message = True, 'hover'
        return res

    def prepare(self):
        sc = self.scenario
        if sc not in ENTRY:
            return False, f"scenario '{sc}' tidak dikenal (test/right/left)"
        if self.R is None:
            return False, 'belum ada /mavros/local_position/pose (T3 hidup?)'
        if self.state is None or not self.state.connected:
            return False, 'MAVROS belum connected ke FCU'
        self.on_ground = not self.state.armed
        if sc == 'test' and self.on_ground:
            if np.linalg.norm(self.pw[:2] - SPAWN[:2]) < 1.5:
                pass                                           # frame lokal (+ off manual) sudah = frame dunia
            elif np.linalg.norm(self.p_l[:2]) < 1.5 and not self.off.any():
                self.off = SPAWN - self.p_l                    # frame lokal relatif titik awal
                self.log(f'frame lokal relatif terdeteksi, off = {np.round(self.off, 3)}')
            else:
                return False, (f'scenario test harus mulai di spawn {SPAWN[:2]}, tapi pose dunia = '
                               f'{np.round(self.pw[:2], 2)}. Taruh drone di spawn atau set off_x/off_y/off_z')
        pw = self.pw
        self.log(f'posisi awal (dunia) = {np.round(pw, 2)}, yaw = {math.degrees(self.yaw):.0f} deg')
        x0, x1, y0, y1 = HANDOFF[sc]
        if not (x0 <= pw[0] <= x1 and y0 <= pw[1] <= y1):
            self.get_logger().warn(f'posisi awal di luar perkiraan area serah-terima {sc} {HANDOFF[sc]}')
        pts = [pw[:2], np.array(ENTRY[sc]), PAD_XY]
        if self.check_keepout:
            for name, box in KEEPOUT.items():
                for a, b in zip(pts[:-1], pts[1:]):
                    if seg_hits_box(a, b, box, self.margin):
                        return False, (f'jalur {np.round(a, 1)} -> {np.round(b, 1)} menembus zona larangan '
                                       f'"{name}" (margin {self.margin} m). Geser titik awal atau set '
                                       'check_keepout:=false')
        self.wps = [np.array(ENTRY[sc]), PAD_XY.copy()]
        self.wp_i = 0
        self.z_rest = float(pw[2])
        self.seq, self.fails, self.next_try = 'MODE', 0, 0.0
        self.phase = 'SETUP'
        return True, f'mulai: scenario={sc}, on_ground={self.on_ground}'

    # ------------------------------------------------------------ setpoint out
    def send(self, vx, vy, vz):
        m = PositionTarget()
        m.header.stamp = self.get_clock().now().to_msg()
        m.coordinate_frame = PositionTarget.FRAME_LOCAL_NED      # MAVROS: nilai kita ENU, ia konversi ke NED
        m.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
                       PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                       PositionTarget.IGNORE_YAW_RATE)           # sisa: kecepatan + yaw absolut
        m.velocity.x, m.velocity.y, m.velocity.z = float(vx), float(vy), float(vz)
        m.yaw = float(NORTH)
        self.sp_pub.publish(m)

    def hold_z(self, z_target, lim=0.6):
        return clip(self.k_z * (z_target - self.pw[2]), lim)

    # ------------------------------------------------------------ state machine
    def tick(self):
        if self.phase == 'WAIT_START':
            return
        t = self.now()
        if self.R is None or t - self.pose_t > 1.5:
            self.get_logger().warn('pose MAVROS lama/tidak ada -> hover', throttle_duration_sec=2.0)
            self.send(0.0, 0.0, 0.0)
            return
        fn = getattr(self, 'ph_' + self.phase.lower())
        v = fn(t)
        if v is not None:
            self.send(*v)
        pw = self.pw
        self.get_logger().info(f'[{self.phase}] pos=({pw[0]:.2f},{pw[1]:.2f},{pw[2]:.2f}) '
                               f'yaw={math.degrees(self.yaw):.0f}', throttle_duration_sec=1.0)

    def go(self, phase, msg=''):
        self.log(f'{self.phase} -> {phase}  {msg}')
        self.phase, self.stable, self.lost_since, self.fb_ok = phase, 0, None, 0

    def pad_fresh(self, t):
        return self.pad_est is not None and t - self.pad_t < 0.8

    def ring_fresh(self, t):
        return self.ring_est is not None and t - self.ring_t < 0.8

    # --- SETUP: GUIDED -> arm -> takeoff (dilewati jika sudah terbang)
    def call(self, tag, cli, req, t):
        if not cli.service_is_ready():
            self.get_logger().warn(f'service {tag} belum siap', throttle_duration_sec=2.0)
            return
        self.fut, self.fut_tag, self.fut_t0, self.next_try = cli.call_async(req), tag, t, t + 1.0

    def ph_setup(self, t):
        if self.fut is not None:
            if not self.fut.done():
                if t - self.fut_t0 > 8.0:
                    self.get_logger().warn(f'service {self.fut_tag} timeout, ulangi')
                    self.fut = None
                return None
            res, tag, self.fut = self.fut.result(), self.fut_tag, None
            ok = res.mode_sent if tag == 'mode' else res.success
            self.log(f'{tag}: {"OK" if ok else "GAGAL"}')
            if not ok:
                self.fails += 1
                self.next_try = t + 2.0
                if self.fails >= 6:
                    self.fail('setup gagal 6x berturut-turut; cek T2/T3 dan pre-arm')
            elif tag == 'takeoff':
                self.seq = 'WAIT_ALT'
            return None
        if t < self.next_try:
            return None
        st = self.state
        if self.seq == 'MODE':
            if st.mode == 'GUIDED':
                self.seq = 'ARM' if self.on_ground else 'END'
            else:
                r = SetMode.Request()
                r.custom_mode = 'GUIDED'
                self.call('mode', self.cli_mode, r, t)
        elif self.seq == 'ARM':
            if st.armed:
                self.seq = 'TAKEOFF'
            else:
                r = CommandBool.Request()
                r.value = True
                self.call('arm', self.cli_arm, r, t)
        elif self.seq == 'TAKEOFF':
            r = CommandTOL.Request()
            r.altitude = float(self.cruise_alt - self.z_rest)   # relatif titik awal (home)
            self.call('takeoff', self.cli_tko, r, t)
        elif self.seq == 'WAIT_ALT':
            if not st.armed:
                self.seq = 'ARM'                                  # ter-disarm otomatis: ulangi
            elif self.pw[2] >= self.cruise_alt - 0.4:
                self.seq = 'END'
        elif self.seq == 'END':
            self.go('CLIMB')
        return None

    # --- CLIMB: pastikan tinggi cruise + yaw utara
    def ph_climb(self, t):
        e = self.cruise_alt - self.pw[2]
        yaw_ok = abs(wrap(NORTH - self.yaw)) < math.radians(6.0)
        self.stable = self.stable + 1 if (abs(e) < 0.15 and yaw_ok) else 0
        if self.stable >= 10:
            self.go('GOTO', f'-> titik masuk {ENTRY[self.scenario]}')
        return 0.0, 0.0, clip(self.k_z * e, 0.8)

    # --- GOTO: titik masuk lalu pad nominal; serahkan ke visi begitu pad terlihat
    def ph_goto(self, t):
        last = self.wp_i == len(self.wps) - 1
        if last and self.pad_fresh(t):
            self.go('CENTER_PAD', 'pad terlihat kamera bawah')
            return 0.0, 0.0, self.hold_z(self.cruise_alt)
        e = self.wps[self.wp_i] - self.pw[:2]
        dist = float(np.linalg.norm(e))
        if dist < (0.5 if last else 0.4):
            if last:
                self.go('CENTER_PAD', 'tiba di pad nominal tanpa deteksi')
            else:
                self.wp_i += 1
                self.log('titik masuk tercapai, lanjut ke pad')
            return 0.0, 0.0, self.hold_z(self.cruise_alt)
        speed = min(self.v_final if last else self.v_cruise, 0.8 * dist)
        v = e / dist * speed
        return v[0], v[1], self.hold_z(self.cruise_alt)

    # --- CENTER_PAD: servo visual di atas pad
    def ph_center_pad(self, t):
        using_fb = False
        if self.pad_fresh(t):
            e = self.pad_est - self.pw[:2]
            self.lost_since = None
        else:
            self.lost_since = self.lost_since or t
            if self.allow_fallback and t - self.lost_since > self.lost_timeout:
                e, using_fb = PAD_XY - self.pw[:2], True
                self.get_logger().warn('pad tidak terdeteksi: pakai koordinat nominal', throttle_duration_sec=3.0)
            else:
                e = np.zeros(2)
                self.get_logger().warn('pad belum terlihat: hover menunggu', throttle_duration_sec=3.0)
        v = clip_norm(self.k_xy * e, self.v_fine)
        if using_fb:
            self.fb_ok = self.fb_ok + 1 if np.linalg.norm(e) < self.pad_tol else 0
        if self.pad_ok >= self.stable_frames or self.fb_ok >= 20:
            self.go('ALIGN_RING', f'pad ter-center di ({self.pw[0]:.2f},{self.pw[1]:.2f})')
        return v[0], v[1], self.hold_z(self.cruise_alt)

    # --- ALIGN_RING: diam, sejajarkan x dan z dengan pusat ring
    def ph_align_ring(self, t):
        using_fb = False
        if self.ring_fresh(t):
            e = self.ring_est - np.array([self.pw[0], self.pw[2]])
            self.lost_since = None
        else:
            self.lost_since = self.lost_since or t
            if self.allow_fallback and t - self.lost_since > self.lost_timeout:
                e, using_fb = np.array([RING_X, RING_Z]) - np.array([self.pw[0], self.pw[2]]), True
                self.get_logger().warn('ring tidak terdeteksi: pakai koordinat nominal', throttle_duration_sec=3.0)
            else:
                e = np.zeros(2)
        if using_fb:
            self.fb_ok = self.fb_ok + 1 if np.linalg.norm(e) < self.ring_tol else 0
        if self.ring_ok >= self.stable_frames or self.fb_ok >= 20:
            self.go('APPROACH_RING', 'sejajar dengan ring')
        return clip(self.k_xy * e[0], self.v_fine), 0.0, clip(self.k_z * e[1], 0.4)

    def dist_to_ring(self):
        return RING_Y - self.cam_pos_w('front')[1]

    # --- APPROACH_RING: maju pelan, kecepatan dikalikan "gate" kesejajaran
    def ph_approach_ring(self, t):
        dist = self.dist_to_ring()
        fresh = self.ring_fresh(t)
        self.lost_since = None if fresh else (self.lost_since or t)
        lost_for = 0.0 if fresh else t - self.lost_since
        if dist <= self.commit_dist or (lost_for > 1.0 and dist < 3.0):
            self.hold_xz = self.ring_est.copy() if self.ring_est is not None else np.array([RING_X, RING_Z])
            self.go('COMMIT', f'jarak ke ring {dist:.2f} m; kunci x,z = {np.round(self.hold_xz, 2)}')
            return 0.0, 0.0, 0.0
        target = None
        if fresh:
            target = self.ring_est
        elif self.allow_fallback and lost_for > self.lost_timeout:
            target = np.array([RING_X, RING_Z])
            self.get_logger().warn('ring hilang: maju dengan koordinat nominal', throttle_duration_sec=3.0)
        if target is None:
            return 0.0, 0.0, 0.0                       # belum ada acuan: diam
        e = target - np.array([self.pw[0], self.pw[2]])
        gate = min(1.0, max(0.0, (0.45 - float(np.linalg.norm(e))) / 0.30))   # 1 bila err<0.15, 0 bila err>0.45
        return clip(self.k_xy * e[0], self.v_fine), self.v_ring * gate, clip(self.k_z * e[1], 0.4)

    # --- COMMIT / EXIT: buta, tahan x,z terkunci, maju lurus menembus ring
    def _blind_forward(self):
        e = self.hold_xz - np.array([self.pw[0], self.pw[2]])
        return clip(self.k_xy * e[0], 0.3), self.v_ring, clip(self.k_z * e[1], 0.3)

    def ph_commit(self, t):
        if self.pw[1] >= RING_Y + 0.9:
            self.go('EXIT', 'ring terlewati')
        return self._blind_forward()

    def ph_exit(self, t):
        if self.pw[1] >= RING_Y + self.exit_dist:
            self.z_hold = float(self.pw[2])
            self.go('DONE', f'hover di y={self.pw[1]:.2f}; serahkan ke node kamikaze')
            return 0.0, 0.0, 0.0
        return self._blind_forward()

    def ph_done(self, t):
        return (0.0, 0.0, self.hold_z(self.z_hold)) if self.hold_after_done else None

    def ph_abort(self, t):
        return 0.0, 0.0, self.hold_z(self.z_hold)


# =============================================================================
# 4. INTEGRASI PIPELINE & ENTRYPOINT
# =============================================================================
def misi(
    drone=None,
    sensor=None,
    scenario: str = 'right',
    timeout_sec: float = 120.0
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Eksekusi Misi 2 (Centering Pad Hijau + Menembus Ring)
    berdasarkan algoritma GateNode mentahan.
    Dapat dipanggil langsung dari main.py.
    """
    if not rclpy.ok():
        rclpy.init()

    print("\n" + "=" * 65)
    print(f">>> [MISI 2] MEMULAI GATE NODE (PAD HIJAU & RING) [Scenario: {scenario}]")
    print("=" * 65)

    node = GateNode()
    node.scenario = scenario
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    # Tunggu koneksi & data pose dari MAVROS
    print("[*] Menunggu data pose & koneksi MAVROS...")
    t_wait = time.time()
    while rclpy.ok() and (node.R is None or node.state is None or not node.state.connected):
        executor.spin_once(timeout_sec=0.1)
        if time.time() - t_wait > 15.0:
            print("[!] Timeout menunggu MAVROS! Pastikan sensors_mavros.launch.py hidup.")
            break

    # Memulai persiapan & eksekusi misi
    ok, msg = node.prepare()
    if not ok and 'zona larangan' in msg:
        node.get_logger().warn(f"Titik serah terima menyentuh margin keepout, menonaktifkan check_keepout...")
        node.check_keepout = False
        ok, msg = node.prepare()

    print(f"[*] Persiapan Misi 2: {msg}")
    if not ok:
        print(f"[!] Gagal memulai Misi 2: {msg}")
        executor.remove_node(node)
        node.destroy_node()
        return drone, sensor

    t_start = time.time()
    while rclpy.ok() and node.phase not in ('DONE', 'ABORT'):
        executor.spin_once(timeout_sec=0.05)
        if time.time() - t_start > timeout_sec:
            node.fail(f"Misi 2 melampaui batas waktu ({timeout_sec}s)!")
            break

    if node.phase == 'DONE':
        print("\n" + "=" * 65)
        print("🎉🎉 [MISI 2 SUKSES] PAD HIJAU & RING BERHASIL DITEMBUS! 🎉🎉")
        print(f"    -> Posisi Hover: X={node.pw[0]:.2f}, Y={node.pw[1]:.2f}, Z={node.pw[2]:.2f}")
        print("=" * 65)
    else:
        print(f"\n[!] Misi 2 berakhir dengan status: {node.phase}")

    executor.remove_node(node)
    node.destroy_node()
    return drone, sensor


def main():
    rclpy.init()
    node = GateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
