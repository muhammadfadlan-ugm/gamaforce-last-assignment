#!/usr/bin/env python3
"""
================================================================================
DETEKSI & CENTERING OBSTACLE MERAH (CAMERA FRONT / DOWN)
================================================================================
Modul untuk mendeteksi rintangan / balok / dinding penghalang berwarna MERAH
pada arena (kamera depan atau kamera bawah drone).

Fitur:
- Pure OpenCV: Segmentasi dual-HSV merah (wrap-around 0-10 & 170-180) + Morfologi.
- Analisis Kontur Rintangan:
  - Bounding Box tegak (x, y, w, h) & Rotated Bounding Box (minAreaRect) beserta sudut orientasi.
  - Menghitung titik pusat (cx, cy), dimensi lebar/tinggi, luas area, dan error offset (dx, dy).
  - Melacak seluruh obstacle merah yang terlihat sekaligus (multi-obstacle tracking).
- Visualisasi HUD: Bounding box merah, titik pusat obstacle, vektor error, dan banner info.
- Standalone CLI ROS 2.

Penggunaan Standalone:
  python3 scripts/deteksi/obstacle_merah.py --topic /camera_front
  python3 scripts/deteksi/obstacle_merah.py --topic /camera_down --headless
================================================================================
"""

import sys
import os
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np


@dataclass
class ObstacleMerahResult:
    detected: bool
    cx: int
    cy: int
    area: float
    bbox: Tuple[int, int, int, int]          # (x, y, w, h) obstacle utama/terbesar
    width: int
    height: int
    angle: float                             # Sudut rotasi obstacle (derajat)
    dx: int                                  # Offset horizontal dari pusat kamera (cx - w/2)
    dy: int                                  # Offset vertikal dari pusat kamera (cy - h/2)
    norm_dx: float                           # Offset X ternormalisasi [-1.0 .. 1.0]
    norm_dy: float                           # Offset Y ternormalisasi [-1.0 .. 1.0]
    error_dist: float                        # Jarak euclidian offset (pixel)
    is_centered: bool                        # True jika |dx| <= tolerance dan |dy| <= tolerance
    obstacle_count: int                      # Jumlah obstacle merah terdeteksi
    all_bboxes: List[Tuple[int, int, int, int]]
    annotated_frame: Optional[np.ndarray] = None


class ObstacleMerahDetector:
    """
    Detektor obstacle/balok/penghalang merah berbasis Dual-HSV Thresholding & Contour Analysis.
    """
    def __init__(
        self,
        lower_red1: Tuple[int, int, int] = (0, 90, 50),
        upper_red1: Tuple[int, int, int] = (10, 255, 255),
        lower_red2: Tuple[int, int, int] = (170, 90, 50),
        upper_red2: Tuple[int, int, int] = (180, 255, 255),
        min_area: float = 250.0,
        tolerance_px: int = 25,
    ):
        self.lower_red1 = np.array(lower_red1, dtype=np.uint8)
        self.upper_red1 = np.array(upper_red1, dtype=np.uint8)
        self.lower_red2 = np.array(lower_red2, dtype=np.uint8)
        self.upper_red2 = np.array(upper_red2, dtype=np.uint8)
        self.min_area = min_area
        self.tolerance_px = tolerance_px

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    def detect(self, frame_bgr: np.ndarray, draw: bool = True) -> ObstacleMerahResult:
        """
        Mendeteksi obstacle merah pada frame gambar BGR.
        """
        h, w = frame_bgr.shape[:2]
        center_frame_x = w // 2
        center_frame_y = h // 2

        # 1. Konversi ke HSV & Blur
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 2. Dual-HSV Masking untuk warna merah
        mask1 = cv2.inRange(blurred, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(blurred, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        # 3. Pencarian Kontur
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_obstacles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                valid_obstacles.append((area, cnt, (bx, by, bw, bh)))

        annotated = frame_bgr.copy() if draw else None

        if valid_obstacles:
            # Urutkan berdasarkan luas area terbesar
            valid_obstacles.sort(key=lambda x: x[0], reverse=True)
            best_area, best_cnt, (bx, by, bw, bh) = valid_obstacles[0]

            # Hitung centroid
            M = cv2.moments(best_cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = bx + bw // 2
                cy = by + bh // 2

            # Rotated bounding rect untuk sudut orientasi
            rect = cv2.minAreaRect(best_cnt)
            angle = float(rect[2])

            dx = cx - center_frame_x
            dy = cy - center_frame_y
            norm_dx = float(dx) / float(center_frame_x) if center_frame_x > 0 else 0.0
            norm_dy = float(dy) / float(center_frame_y) if center_frame_y > 0 else 0.0
            error_dist = float(np.hypot(dx, dy))
            is_centered = (abs(dx) <= self.tolerance_px and abs(dy) <= self.tolerance_px)

            all_bboxes = [obs[2] for obs in valid_obstacles]

            if draw and annotated is not None:
                self._draw_overlay(
                    annotated, best_cnt, rect, cx, cy, bx, by, bw, bh, best_area,
                    center_frame_x, center_frame_y, dx, dy, error_dist, is_centered,
                    all_bboxes
                )

            return ObstacleMerahResult(
                detected=True,
                cx=cx,
                cy=cy,
                area=float(best_area),
                bbox=(bx, by, bw, bh),
                width=bw,
                height=bh,
                angle=angle,
                dx=dx,
                dy=dy,
                norm_dx=norm_dx,
                norm_dy=norm_dy,
                error_dist=error_dist,
                is_centered=is_centered,
                obstacle_count=len(valid_obstacles),
                all_bboxes=all_bboxes,
                annotated_frame=annotated
            )

        # Tidak Terdeteksi
        if draw and annotated is not None:
            self._draw_no_detection_overlay(annotated, center_frame_x, center_frame_y)

        return ObstacleMerahResult(
            detected=False,
            cx=center_frame_x,
            cy=center_frame_y,
            area=0.0,
            bbox=(0, 0, 0, 0),
            width=0,
            height=0,
            angle=0.0,
            dx=0,
            dy=0,
            norm_dx=0.0,
            norm_dy=0.0,
            error_dist=0.0,
            is_centered=False,
            obstacle_count=0,
            all_bboxes=[],
            annotated_frame=annotated
        )

    def _draw_overlay(
        self, img, best_cnt, rect, cx, cy, bx, by, bw, bh, area,
        fx, fy, dx, dy, dist, is_centered, all_bboxes
    ):
        h, w = img.shape[:2]

        # 1. Gambar kotak semua obstacle merah lain (garis tipis)
        for (ox, oy, ow, oh) in all_bboxes[1:]:
            cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (0, 100, 255), 1)

        # 2. Gambar Rotated Bounding Box obstacle utama
        box_pts = cv2.boxPoints(rect)
        box_pts = np.int32(box_pts)
        cv2.drawContours(img, [box_pts], 0, (0, 0, 255), 2)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (50, 50, 255), 1)
        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)

        # 3. Crosshair tengah kamera
        cv2.line(img, (fx - 15, fy), (fx + 15, fy), (200, 200, 200), 1)
        cv2.line(img, (fx, fy - 15), (fx, fy + 15), (200, 200, 200), 1)
        cv2.circle(img, (fx, fy), self.tolerance_px, (255, 255, 255), 1)

        # 4. Vektor error
        vec_color = (0, 255, 255) if not is_centered else (0, 255, 0)
        cv2.arrowedLine(img, (fx, fy), (cx, cy), vec_color, 2, tipLength=0.15)

        # 5. Top Banner HUD
        status_bg = (0, 150, 0) if is_centered else (0, 0, 180)
        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (160, 36), status_bg, -1)

        status_txt = "CENTERED" if is_centered else "AVOID/TRACK"
        cv2.putText(img, status_txt, (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        info_txt = f"Obstacle Merah | {bw}x{bh}px | Err: dX={dx:+d} dY={dy:+d} (dist={dist:.1f}px) | Total={len(all_bboxes)}"
        cv2.putText(img, info_txt, (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 1)

    def _draw_no_detection_overlay(self, img, fx, fy):
        h, w = img.shape[:2]
        cv2.line(img, (fx - 15, fy), (fx + 15, fy), (100, 100, 100), 1)
        cv2.line(img, (fx, fy - 15), (fx, fy + 15), (100, 100, 100), 1)
        cv2.circle(img, (fx, fy), self.tolerance_px, (80, 80, 80), 1)

        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (160, 36), (0, 0, 160), -1)
        cv2.putText(img, "SEARCHING", (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, "Obstacle Merah Belum Terdeteksi", (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)


# ==============================================================================
# ROS 2 NODE UNTUK PENGUJIAN STANDALONE
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
    node = Node("detector_obstacle_merah")
    detector = ObstacleMerahDetector()

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
    print(f"[OK] ObstacleMerahDetector subscribe ke: {topic}")
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
                        print(f"[{status}] Merah Pos=({res.cx},{res.cy}) dx={res.dx:+d}, dy={res.dy:+d} | Box={res.width}x{res.height} | Total={res.obstacle_count}")
                    else:
                        print("[-] Obstacle Merah: Belum Terdeteksi...")
                    last_log_time = now

                if has_display and res.annotated_frame is not None:
                    cv2.imshow("Deteksi Obstacle Merah", res.annotated_frame)
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
    parser = argparse.ArgumentParser(description="Deteksi & Centering Obstacle Merah")
    parser.add_argument("--topic", default="/camera_front", help="Topik kamera ROS 2 (default: /camera_front)")
    parser.add_argument("--headless", action="store_true", help="Jalankan tanpa window GUI OpenCV")
    args = parser.parse_args()

    run_ros2_node(args.topic, args.headless)


if __name__ == "__main__":
    main()

