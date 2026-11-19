#!/usr/bin/env python3
"""
DETEKSI & CENTERING LINGKARAN HIJAU (CAMERA DOWN)
==================================================
Modul untuk mendeteksi lingkaran/centering ring hijau di bawah drone (kamera bawah).
Menghitung titik pusat (cx, cy) dan error offset (dx, dy) relatif terhadap pusat frame
kamera sehingga drone dapat melakukan posisi hovering/centering presisi atau landing.

Fitur:
- Pure OpenCV (HSV color segmentation, morphological filtering, circularity test, moment centroid).
- Modular: bisa di-import oleh skrip misi (from deteksi.centering_hijau import GreenRingDetector).
- Standalone: dapat dijalankan langsung melalui CLI dengan visualisasi live OpenCV / HUD.
- ROS 2 Image decoder aman (tanpa dependensi cv_bridge yang rentan ABI crash).

Penggunaan Standalone:
  python3 scripts/deteksi/centering_hijau.py
  python3 scripts/deteksi/centering_hijau.py --topic /camera_down --headless
"""

import sys
import os
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class GreenCenteringResult:
    detected: bool
    cx: int
    cy: int
    radius: float
    area: float
    dx: int            # Offset X dari tengah frame (pixel): cx - frame_w/2
    dy: int            # Offset Y dari tengah frame (pixel): cy - frame_h/2
    norm_dx: float     # Offset X ternormalisasi [-1.0 .. 1.0]
    norm_dy: float     # Offset Y ternormalisasi [-1.0 .. 1.0]
    error_dist: float  # Jarak euclidian offset (pixel)
    circularity: float # Nilai sirkularitas kontur [0.0 .. 1.0]
    is_centered: bool  # True jika |dx| <= tolerance dan |dy| <= tolerance
    annotated_frame: Optional[np.ndarray] = None


class GreenRingDetector:
    """
    Detektor lingkaran / ring hijau berbasis OpenCV HSV Color Masking & Contour Analysis.
    """
    def __init__(
        self,
        lower_hsv: Tuple[int, int, int] = (35, 60, 40),
        upper_hsv: Tuple[int, int, int] = (85, 255, 255),
        min_area: float = 200.0,
        min_circularity: float = 0.35,
        tolerance_px: int = 25,
    ):
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.min_area = min_area
        self.min_circularity = min_circularity
        self.tolerance_px = tolerance_px

        # Kernel morfologi
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame_bgr: np.ndarray, draw: bool = True) -> GreenCenteringResult:
        """
        Mendeteksi lingkaran hijau pada frame gambar BGR.
        """
        h, w = frame_bgr.shape[:2]
        center_frame_x = w // 2
        center_frame_y = h // 2

        # 1. Konversi ke HSV dan Blur untuk reduksi noise
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 2. Thresholding warna hijau
        mask = cv2.inRange(blurred, self.lower_hsv, self.upper_hsv)

        # 3. Operasi Morfologi (menghapus noise kecil dan mengisi lubang)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        # 4. Pencarian Kontur
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_area = 0.0
        best_circularity = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity >= self.min_circularity:
                if area > best_area:
                    best_area = area
                    best_cnt = cnt
                    best_circularity = circularity

        # Pastikan kontur yang terdeteksi memenuhi syarat bentuk lingkaran/ring
        annotated = frame_bgr.copy() if draw else None

        if best_cnt is not None and best_circularity >= self.min_circularity:
            # Hitung titik tengah menggunakan Spatial Moments
            M = cv2.moments(best_cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                (cx_f, cy_f), _ = cv2.minEnclosingCircle(best_cnt)
                cx, cy = int(cx_f), int(cy_f)

            (_, _), radius = cv2.minEnclosingCircle(best_cnt)

            # Hitung offset
            dx = cx - center_frame_x
            dy = cy - center_frame_y
            norm_dx = float(dx) / float(center_frame_x) if center_frame_x > 0 else 0.0
            norm_dy = float(dy) / float(center_frame_y) if center_frame_y > 0 else 0.0
            error_dist = float(np.hypot(dx, dy))
            is_centered = (abs(dx) <= self.tolerance_px and abs(dy) <= self.tolerance_px)

            if draw and annotated is not None:
                self._draw_overlay(
                    annotated, best_cnt, cx, cy, radius, best_area,
                    center_frame_x, center_frame_y, dx, dy, error_dist, is_centered
                )

            return GreenCenteringResult(
                detected=True,
                cx=cx,
                cy=cy,
                radius=float(radius),
                area=float(best_area),
                dx=dx,
                dy=dy,
                norm_dx=norm_dx,
                norm_dy=norm_dy,
                error_dist=error_dist,
                circularity=float(best_circularity),
                is_centered=is_centered,
                annotated_frame=annotated
            )

        # Jika tidak terdeteksi
        if draw and annotated is not None:
            self._draw_no_detection_overlay(annotated, center_frame_x, center_frame_y)

        return GreenCenteringResult(
            detected=False,
            cx=center_frame_x,
            cy=center_frame_y,
            radius=0.0,
            area=0.0,
            dx=0,
            dy=0,
            norm_dx=0.0,
            norm_dy=0.0,
            error_dist=0.0,
            circularity=0.0,
            is_centered=False,
            annotated_frame=annotated
        )

    def _draw_overlay(
        self, img, cnt, cx, cy, radius, area,
        fx, fy, dx, dy, dist, is_centered
    ):
        h, w = img.shape[:2]

        # 1. Gambar kontur dan lingkaran target
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), int(radius), (0, 230, 100), 1)
        cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)

        # 2. Gambar crosshair tengah kamera
        cv2.line(img, (fx - 15, fy), (fx + 15, fy), (200, 200, 200), 1)
        cv2.line(img, (fx, fy - 15), (fx, fy + 15), (200, 200, 200), 1)
        cv2.circle(img, (fx, fy), self.tolerance_px, (255, 255, 255), 1)

        # 3. Garis vektor error dari tengah frame ke target
        vec_color = (0, 255, 255) if not is_centered else (0, 255, 0)
        cv2.arrowedLine(img, (fx, fy), (cx, cy), vec_color, 2, tipLength=0.15)

        # 4. HUD Banner
        status_bg = (0, 150, 0) if is_centered else (0, 80, 180)
        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (160, 36), status_bg, -1)

        status_txt = "CENTERED" if is_centered else "TRACKING"
        cv2.putText(img, status_txt, (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        info_txt = f"Err: dX={dx:+d} dY={dy:+d} (dist={dist:.1f}px) | R={radius:.1f} Area={area:.0f}"
        cv2.putText(img, info_txt, (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

    def _draw_no_detection_overlay(self, img, fx, fy):
        h, w = img.shape[:2]
        # Crosshair frame
        cv2.line(img, (fx - 15, fy), (fx + 15, fy), (100, 100, 100), 1)
        cv2.line(img, (fx, fy - 15), (fx, fy + 15), (100, 100, 100), 1)

        # Top banner
        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (160, 36), (0, 0, 160), -1)
        cv2.putText(img, "SEARCHING", (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, "Target Hijau Belum Terdeteksi", (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)


# ==============================================================================
# ROS 2 NODE UNTUK PENGUJIAN & INTEGRASI
# ==============================================================================
def run_ros2_node(topic: str, headless: bool = False):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
    except ImportError:
        print("[!] ROS 2 (rclpy/sensor_msgs) tidak terdeteksi.")
        return

    rclpy.init()
    node = Node("detector_centering_hijau")
    detector = GreenRingDetector()

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
    print(f"[OK] GreenRingDetector subscribe ke: {topic}")
    print("[*] Tekan Ctrl+C di terminal (atau 'q' di jendela GUI) untuk berhenti.")

    last_log_time = time.time()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)

            if latest_frame is not None:
                res = detector.detect(latest_frame, draw=True)

                now = time.time()
                if now - last_log_time >= 0.5:
                    if res.detected:
                        status = "✅ CENTERED" if res.is_centered else "🔍 TRACKING"
                        print(f"[{status}] cx={res.cx:3d}, cy={res.cy:3d} | dx={res.dx:+3d}, dy={res.dy:+3d} | Radius={res.radius:5.1f} | Area={res.area:6.0f}")
                    else:
                        print("[-] Target Hijau: Belum Terdeteksi...")
                    last_log_time = now

                if has_display and res.annotated_frame is not None:
                    cv2.imshow("Centering Lingkaran Hijau", res.annotated_frame)
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


def main():
    parser = argparse.ArgumentParser(description="Deteksi & Centering Lingkaran Hijau")
    parser.add_argument("--topic", default="/camera_down", help="Topic kamera ROS 2 (default: /camera_down)")
    parser.add_argument("--headless", action="store_true", help="Jalankan tanpa window GUI OpenCV")
    args = parser.parse_args()

    run_ros2_node(args.topic, args.headless)


if __name__ == "__main__":
    main()

