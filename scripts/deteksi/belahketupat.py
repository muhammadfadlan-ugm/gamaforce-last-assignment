#!/usr/bin/env python3
"""
================================================================================
DETEKSI & CENTERING BELAH KETUPAT MULTI-WARNA (BIRU / HIJAU / PINK)
================================================================================
Modul tunggal terpadu untuk mendeteksi target berbentuk BELAH KETUPAT
(Diamond/Rhombus) dengan dukungan 3 warna berbeda:
  - BIRU   (Cyan / Blue)
  - HIJAU  (Green)
  - PINK   (Magenta)

Fitur Unggulan:
- Unified Engine: Satu algoritma geometri terpadu (DRY principle).
- Dynamic Color Switching: Dapat berganti warna deteksi kapan saja via detector.set_color().
- Multi-Color Scan: detect_all() mampu memindai ketiga warna sekaligus dalam 1 frame.
- Geometric Polygon Filtering: approxPolyDP 4 vertex, convex test, rasio area bounding box (~0.50).
- Modular & Standalone CLI ROS 2 dengan HUD overlay dinamis sesuai warna target.

Penggunaan Standalone:
  python3 scripts/deteksi/belahketupat.py --color biru --topic /camera_down
  python3 scripts/deteksi/belahketupat.py --color hijau --topic /camera_down
  python3 scripts/deteksi/belahketupat.py --color pink --topic /camera_down
  python3 scripts/deteksi/belahketupat.py --color all --topic /camera_front
================================================================================
"""

import sys
import os
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np


# ==============================================================================
# PRESET WARNA HSV & VISUALISASI
# ==============================================================================
COLOR_PRESETS = {
    "biru": {
        "name": "BIRU",
        "lower": np.array([90, 80, 50], dtype=np.uint8),
        "upper": np.array([130, 255, 255], dtype=np.uint8),
        "bgr": (255, 200, 0),       # Cyan / Biru Muda
        "hud_bg": (180, 80, 0),
    },
    "hijau": {
        "name": "HIJAU",
        "lower": np.array([35, 70, 50], dtype=np.uint8),
        "upper": np.array([85, 255, 255], dtype=np.uint8),
        "bgr": (0, 255, 0),         # Hijau Terang
        "hud_bg": (0, 140, 0),
    },
    "pink": {
        "name": "PINK",
        "lower": np.array([135, 60, 50], dtype=np.uint8),
        "upper": np.array([175, 255, 255], dtype=np.uint8),
        "bgr": (255, 0, 255),       # Magenta / Pink
        "hud_bg": (140, 0, 140),
    },
}


@dataclass
class BelahKetupatResult:
    detected: bool
    color_name: str                # "BIRU", "HIJAU", atau "PINK"
    cx: int
    cy: int
    area: float
    bbox: Tuple[int, int, int, int]# (x, y, w, h)
    vertices: Optional[np.ndarray] # 4 Titik sudut poligon belah ketupat
    dx: int                        # Offset horizontal dari pusat kamera (cx - w/2)
    dy: int                        # Offset vertikal dari pusat kamera (cy - h/2)
    norm_dx: float                 # Offset X ternormalisasi [-1.0 .. 1.0]
    norm_dy: float                 # Offset Y ternormalisasi [-1.0 .. 1.0]
    error_dist: float              # Jarak euclidian offset (pixel)
    is_centered: bool              # True jika |dx| <= tolerance dan |dy| <= tolerance
    annotated_frame: Optional[np.ndarray] = None


class BelahKetupatDetector:
    """
    Detektor bentuk belah ketupat berbasis Pure OpenCV dengan dukungan warna dinamis.
    """
    def __init__(
        self,
        color: str = "biru",
        min_area: float = 180.0,
        tolerance_px: int = 25,
    ):
        self.min_area = min_area
        self.tolerance_px = tolerance_px
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        self.set_color(color)

    def set_color(self, color: str):
        """Mengganti target warna aktif ('biru', 'hijau', 'pink')."""
        color_key = color.lower().strip()
        if color_key not in COLOR_PRESETS and color_key != "all":
            valid = list(COLOR_PRESETS.keys())
            raise ValueError(f"Warna '{color}' tidak dikenal. Pilihan: {valid} atau 'all'")
        self.current_color = color_key

    def _extract_diamond(
        self, mask: np.ndarray, color_key: str, center_x: int, center_y: int
    ) -> Optional[dict]:
        """Ekstraksi kontur belah ketupat terbaik dari mask biner."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_data = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue

            # Aproksimasi poligon
            approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

            # Syarat belah ketupat: 4 titik sudut & konveks
            is_diamond = False
            if len(approx) == 4 and cv2.isContourConvex(approx):
                bx, by, bw, bh = cv2.boundingRect(approx)
                box_area = bw * bh
                if box_area > 0:
                    fill_ratio = area / float(box_area)
                    if 0.35 <= fill_ratio <= 0.75:
                        is_diamond = True
            elif len(approx) in (3, 5):
                # Fallback toleran jika resolusi kamera membulatkan ujung sudut
                is_diamond = True

            if is_diamond and area > best_area:
                best_area = area
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    cx = bx + bw // 2
                    cy = by + bh // 2

                bx, by, bw, bh = cv2.boundingRect(cnt)
                dx = cx - center_x
                dy = cy - center_y
                norm_dx = float(dx) / float(center_x) if center_x > 0 else 0.0
                norm_dy = float(dy) / float(center_y) if center_y > 0 else 0.0
                error_dist = float(np.hypot(dx, dy))
                is_centered = (abs(dx) <= self.tolerance_px and abs(dy) <= self.tolerance_px)

                best_data = {
                    "color_key": color_key,
                    "cnt": cnt,
                    "approx": approx,
                    "cx": cx,
                    "cy": cy,
                    "area": float(area),
                    "bbox": (bx, by, bw, bh),
                    "dx": dx,
                    "dy": dy,
                    "norm_dx": norm_dx,
                    "norm_dy": norm_dy,
                    "error_dist": error_dist,
                    "is_centered": is_centered,
                }

        return best_data

    def detect(
        self,
        frame_bgr: np.ndarray,
        color: Optional[str] = None,
        draw: bool = True
    ) -> BelahKetupatResult:
        """
        Mendeteksi belah ketupat warna aktif (atau warna spesifik yang di-override).
        """
        target_color = (color.lower().strip() if color else self.current_color)
        if target_color == "all":
            # Jika mode 'all', pilih deteksi dengan area terbesar dari semua warna
            all_res = self.detect_all(frame_bgr, draw=draw)
            detected_items = [r for r in all_res.values() if r.detected]
            if detected_items:
                return max(detected_items, key=lambda x: x.area)
            target_color = "biru"

        preset = COLOR_PRESETS[target_color]
        h, w = frame_bgr.shape[:2]
        center_x = w // 2
        center_y = h // 2

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        mask = cv2.inRange(blurred, preset["lower"], preset["upper"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        best = self._extract_diamond(mask, target_color, center_x, center_y)
        annotated = frame_bgr.copy() if draw else None

        if best is not None:
            if draw and annotated is not None:
                self._draw_single_overlay(annotated, best, center_x, center_y)

            return BelahKetupatResult(
                detected=True,
                color_name=preset["name"],
                cx=best["cx"],
                cy=best["cy"],
                area=best["area"],
                bbox=best["bbox"],
                vertices=best["approx"],
                dx=best["dx"],
                dy=best["dy"],
                norm_dx=best["norm_dx"],
                norm_dy=best["norm_dy"],
                error_dist=best["error_dist"],
                is_centered=best["is_centered"],
                annotated_frame=annotated
            )

        if draw and annotated is not None:
            self._draw_no_detection_overlay(annotated, preset["name"], center_x, center_y)

        return BelahKetupatResult(
            detected=False,
            color_name=preset["name"],
            cx=center_x,
            cy=center_y,
            area=0.0,
            bbox=(0, 0, 0, 0),
            vertices=None,
            dx=0,
            dy=0,
            norm_dx=0.0,
            norm_dy=0.0,
            error_dist=0.0,
            is_centered=False,
            annotated_frame=annotated
        )

    def detect_all(
        self,
        frame_bgr: np.ndarray,
        draw: bool = True
    ) -> Dict[str, BelahKetupatResult]:
        """
        Memindai ketiga warna belah ketupat (biru, hijau, pink) secara simultan.
        """
        results = {}
        annotated = frame_bgr.copy() if draw else None
        h, w = frame_bgr.shape[:2]
        center_x = w // 2
        center_y = h // 2

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        found_any = False
        for c_key, preset in COLOR_PRESETS.items():
            mask = cv2.inRange(blurred, preset["lower"], preset["upper"])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

            best = self._extract_diamond(mask, c_key, center_x, center_y)
            if best is not None:
                found_any = True
                if draw and annotated is not None:
                    self._draw_single_overlay(annotated, best, center_x, center_y, multi_mode=True)
                results[c_key] = BelahKetupatResult(
                    detected=True,
                    color_name=preset["name"],
                    cx=best["cx"],
                    cy=best["cy"],
                    area=best["area"],
                    bbox=best["bbox"],
                    vertices=best["approx"],
                    dx=best["dx"],
                    dy=best["dy"],
                    norm_dx=best["norm_dx"],
                    norm_dy=best["norm_dy"],
                    error_dist=best["error_dist"],
                    is_centered=best["is_centered"],
                    annotated_frame=annotated
                )
            else:
                results[c_key] = BelahKetupatResult(
                    detected=False,
                    color_name=preset["name"],
                    cx=center_x,
                    cy=center_y,
                    area=0.0,
                    bbox=(0, 0, 0, 0),
                    vertices=None,
                    dx=0,
                    dy=0,
                    norm_dx=0.0,
                    norm_dy=0.0,
                    error_dist=0.0,
                    is_centered=False,
                    annotated_frame=annotated
                )

        if draw and annotated is not None and not found_any:
            self._draw_no_detection_overlay(annotated, "SEMUA WARNA", center_x, center_y)

        return results

    def _draw_single_overlay(self, img, data, fx, fy, multi_mode: bool = False):
        h, w = img.shape[:2]
        c_key = data["color_key"]
        preset = COLOR_PRESETS[c_key]
        bgr = preset["bgr"]
        cx, cy = data["cx"], data["cy"]
        dx, dy = data["dx"], data["dy"]
        dist = data["error_dist"]
        is_centered = data["is_centered"]

        # 1. Garis kontur poligon
        poly = data["approx"] if data["approx"] is not None else data["cnt"]
        cv2.drawContours(img, [poly], -1, bgr, 2)
        cv2.circle(img, (cx, cy), 6, bgr, -1)

        # 2. Crosshair kamera
        cv2.line(img, (fx - 15, fy), (fx + 15, fy), (200, 200, 200), 1)
        cv2.line(img, (fx, fy - 15), (fx, fy + 15), (200, 200, 200), 1)
        cv2.circle(img, (fx, fy), self.tolerance_px, (255, 255, 255), 1)

        # 3. Vektor error
        vec_col = (0, 255, 0) if is_centered else (0, 255, 255)
        cv2.arrowedLine(img, (fx, fy), (cx, cy), vec_col, 2, tipLength=0.15)

        # 4. Banner HUD
        status_bg = (0, 150, 0) if is_centered else preset["hud_bg"]
        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (160, 36), status_bg, -1)

        status_txt = "CENTERED" if is_centered else "TRACKING"
        cv2.putText(img, status_txt, (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        info_txt = f"Belah Ketupat {preset['name']} | Err: dX={dx:+d} dY={dy:+d} (dist={dist:.1f}px) | Area={data['area']:.0f}"
        cv2.putText(img, info_txt, (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, bgr, 1)

    def _draw_no_detection_overlay(self, img, color_title, fx, fy):
        h, w = img.shape[:2]
        cv2.line(img, (fx - 15, fy), (fx + 15, fy), (100, 100, 100), 1)
        cv2.line(img, (fx, fy - 15), (fx, fy + 15), (100, 100, 100), 1)
        cv2.circle(img, (fx, fy), self.tolerance_px, (80, 80, 80), 1)

        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (160, 36), (0, 0, 160), -1)
        cv2.putText(img, "SEARCHING", (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, f"Belah Ketupat {color_title} Belum Terdeteksi", (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)


# ==============================================================================
# ALIAS CONVENIENCE (KOMPATIBILITAS SKRIP LAMA)
# ==============================================================================
class BelahKetupatBiruDetector(BelahKetupatDetector):
    def __init__(self, **kwargs):
        super().__init__(color="biru", **kwargs)

class BelahKetupatHijauDetector(BelahKetupatDetector):
    def __init__(self, **kwargs):
        super().__init__(color="hijau", **kwargs)

class BelahKetupatPinkDetector(BelahKetupatDetector):
    def __init__(self, **kwargs):
        super().__init__(color="pink", **kwargs)


# ==============================================================================
# ROS 2 NODE UNTUK PENGUJIAN STANDALONE
# ==============================================================================
def run_ros2_node(color: str, topic: str, headless: bool = False):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
    except ImportError:
        print("[!] ROS 2 (rclpy/sensor_msgs) tidak terdeteksi.")
        return

    rclpy.init()
    node = Node("detector_belahketupat_unified")
    detector = BelahKetupatDetector(color=color)

    latest_frame = None
    has_display = bool(os.environ.get("DISPLAY")) and not headless

    def image_callback(msg: Image):
        nonlocal latest_frame
        try:
            if msg.encoding in ("bgr8", "rgb8"):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else arr
            else:
                return
            latest_frame = frame
        except Exception:
            pass

    node.create_subscription(Image, topic, image_callback, 10)
    print(f"[OK] BelahKetupatDetector (Color={color.upper()}) subscribe ke: {topic}")
    print("[*] Tekan Ctrl+C di terminal (atau 'q' di jendela GUI) untuk berhenti.")

    last_log_time = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            if latest_frame is not None:
                if color == "all":
                    all_res = detector.detect_all(latest_frame, draw=True)
                    detected = [r for r in all_res.values() if r.detected]
                    res = detected[0] if detected else list(all_res.values())[0]
                else:
                    res = detector.detect(latest_frame, draw=True)

                now = time.time()
                if now - last_log_time >= 0.5:
                    if res.detected:
                        status = "✅ CENTERED" if res.is_centered else "🔍 TRACKING"
                        print(f"[{status}] {res.color_name} Pos=({res.cx},{res.cy}) dx={res.dx:+d}, dy={res.dy:+d} | Area={res.area:.0f}")
                    else:
                        print(f"[-] Belah Ketupat {color.upper()}: Belum Terdeteksi...")
                    last_log_time = now

                if has_display and res.annotated_frame is not None:
                    cv2.imshow(f"Deteksi Belah Ketupat ({color.upper()})", res.annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
    except KeyboardInterrupt:
        print("\n[!] Dihentikan oleh user.")
    finally:
        if has_display:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


def main(default_color: str = "biru"):
    parser = argparse.ArgumentParser(description="Deteksi Belah Ketupat Multi-Warna (Biru/Hijau/Pink/All)")
    parser.add_argument("--color", default=default_color, choices=["biru", "hijau", "pink", "all"],
                        help="Warna target belah ketupat (default: biru)")
    parser.add_argument("--topic", default="/camera_down", help="Topik kamera ROS 2 (default: /camera_down)")
    parser.add_argument("--headless", action="store_true", help="Jalankan tanpa window GUI OpenCV")
    args = parser.parse_args()

    run_ros2_node(args.color, args.topic, args.headless)


if __name__ == "__main__":
    main()

