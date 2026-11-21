#!/usr/bin/env python3
"""
================================================================================
DETEKSI & CENTERING OBSTACLE KUNING (CAMERA FRONT / DOWN)
================================================================================
Modul untuk mendeteksi rintangan / balok / gawang U terbalik berwarna KUNING
pada arena jalur kiri (kamera depan atau kamera bawah drone).

Fitur:
- Pure OpenCV: Segmentasi HSV warna kuning/oranye (Hue 12-38) + Morfologi.
- Analisis Kontur Rintangan:
  - Bounding Box (x, y, w, h) & Rotated Bounding Box (minAreaRect).
  - Menghitung titik pusat (cx, cy), dimensi lebar/tinggi, luas area, dan offset error (dx, dy).
- Visualisasi HUD: Bounding box kuning, titik pusat obstacle, vektor error, dan banner info.
- Standalone CLI ROS 2.

Penggunaan Standalone:
  python3 scripts/deteksi/obstacle_kuning.py --topic /camera_front
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
class ObstacleKuningResult:
    detected: bool
    cx: int
    cy: int
    area: float
    bbox: Tuple[int, int, int, int]          # (x, y, w, h)
    width: int
    height: int
    dx: int                                  # Offset horizontal dari pusat kamera (cx - w/2)
    dy: int                                  # Offset vertikal dari pusat kamera (cy - h/2)
    norm_dx: float                           # Offset X ternormalisasi [-1.0 .. 1.0]
    norm_dy: float                           # Offset Y ternormalisasi [-1.0 .. 1.0]
    error_dist: float                        # Jarak euclidian offset (pixel)
    is_centered: bool                        # True jika |dx| <= tolerance dan |dy| <= tolerance
    annotated_frame: Optional[np.ndarray] = None


class ObstacleKuningDetector:
    """
    Detektor obstacle/gawang kuning berbasis HSV Thresholding & Contour Analysis.
    """
    def __init__(
        self,
        lower_yellow: Tuple[int, int, int] = (10, 80, 50),
        upper_yellow: Tuple[int, int, int] = (38, 255, 255),
        min_area: float = 200.0,
        tolerance_px: int = 25,
    ):
        self.lower_yellow = np.array(lower_yellow, dtype=np.uint8)
        self.upper_yellow = np.array(upper_yellow, dtype=np.uint8)
        self.min_area = min_area
        self.tolerance_px = tolerance_px

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    def detect(self, frame_bgr: np.ndarray, draw: bool = True) -> ObstacleKuningResult:
        """
        Mendeteksi obstacle kuning pada frame gambar BGR.
        """
        h, w = frame_bgr.shape[:2]
        center_frame_x = w // 2
        center_frame_y = h // 2

        # 1. Konversi ke HSV & Blur
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 2. Thresholding Warna Kuning
        mask = cv2.inRange(blurred, self.lower_yellow, self.upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        # 3. Pencarian Kontur
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated = frame_bgr.copy() if draw else None

        if not contours:
            return ObstacleKuningResult(
                detected=False, cx=center_frame_x, cy=center_frame_y, area=0.0,
                bbox=(0, 0, 0, 0), width=0, height=0, dx=0, dy=0,
                norm_dx=0.0, norm_dy=0.0, error_dist=0.0, is_centered=False,
                annotated_frame=annotated
            )

        # Ambil kontur dengan area terbesar
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        if not valid_contours:
            return ObstacleKuningResult(
                detected=False, cx=center_frame_x, cy=center_frame_y, area=0.0,
                bbox=(0, 0, 0, 0), width=0, height=0, dx=0, dy=0,
                norm_dx=0.0, norm_dy=0.0, error_dist=0.0, is_centered=False,
                annotated_frame=annotated
            )

        best_cnt = max(valid_contours, key=cv2.contourArea)
        area = float(cv2.contourArea(best_cnt))
        x, y, bw, bh = cv2.boundingRect(best_cnt)

        # Hitung titik tengah
        M = cv2.moments(best_cnt)
        if M["m00"] > 1e-5:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + bw // 2
            cy = y + bh // 2

        dx = cx - center_frame_x
        dy = cy - center_frame_y
        norm_dx = float(dx / (w / 2.0))
        norm_dy = float(dy / (h / 2.0))
        error_dist = float(np.hypot(dx, dy))
        is_centered = (abs(dx) <= self.tolerance_px) and (abs(dy) <= self.tolerance_px)

        if draw and annotated is not None:
            # Gambar bounding box & centroid
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 220, 255), 2)
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
            cv2.drawMarker(annotated, (center_frame_x, center_frame_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.line(annotated, (center_frame_x, center_frame_y), (cx, cy), (0, 255, 255), 1)

            # Banner info
            status_txt = "CENTERED" if is_centered else "TRACKING"
            color_txt = (0, 255, 0) if is_centered else (0, 220, 255)
            cv2.putText(annotated, f"OBSTACLE KUNING: {status_txt}", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_txt, 2)
            cv2.putText(annotated, f"dx:{dx:+4d} dy:{dy:+4d} Area:{int(area)}", (x, y + bh + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return ObstacleKuningResult(
            detected=True, cx=cx, cy=cy, area=area,
            bbox=(x, y, bw, bh), width=bw, height=bh, dx=dx, dy=dy,
            norm_dx=norm_dx, norm_dy=norm_dy, error_dist=error_dist,
            is_centered=is_centered, annotated_frame=annotated
        )


def main():
    parser = argparse.ArgumentParser(description="Deteksi Obstacle Kuning ROS 2")
    parser.add_argument("--topic", type=str, default="/camera_front", help="Topik kamera")
    parser.add_argument("--headless", action="store_true", help="Jalankan tanpa GUI OpenCV")
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node("deteksi_obstacle_kuning_node")
    detector = ObstacleKuningDetector()

    latest_frame = [None]

    def cam_cb(msg: Image):
        try:
            if msg.encoding in ("bgr8", "rgb8"):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                if msg.encoding == "rgb8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                latest_frame[0] = frame
        except Exception:
            pass

    node.create_subscription(Image, args.topic, cam_cb, 10)
    print(f"[*] Menunggu citra dari topik {args.topic}...")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            if latest_frame[0] is not None:
                res = detector.detect(latest_frame[0], draw=not args.headless)
                if not args.headless and res.annotated_frame is not None:
                    cv2.imshow("Deteksi Obstacle Kuning", res.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                elif res.detected:
                    print(f"\r[KUNING] cx={res.cx:3d}, cy={res.cy:3d}, dx={res.dx:+3d}, dy={res.dy:+3d}, Area={res.area:.0f}", end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

