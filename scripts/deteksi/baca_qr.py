#!/usr/bin/env python3
"""
================================================================================
DETEKSI & PEMBACAAN KODE QR & MARKER ARUCO (OPENCV)
================================================================================
Modul untuk mendeteksi dan membaca informasi kode QR (serta ArUco marker fallback)
dari feed kamera drone (kamera depan / bawah).

Fitur:
- Pure OpenCV: cv2.QRCodeDetector() untuk membaca QR Code standar.
- Fallback Terpadu: cv2.aruco.ArucoDetector (DICT_4X4_50) untuk membaca marker arena OPREC.
- Menghitung bounding polygon, titik pusat (cx, cy), dan offset error (dx, dy)
  relatif terhadap titik tengah frame kamera.
- Visualisasi HUD: status decoding, string data hasil scan, vektor error, dan reticle.
- Modular & Standalone CLI ROS 2.

Penggunaan Standalone:
  python3 scripts/deteksi/baca_qr.py --topic /camera_front
  python3 scripts/deteksi/baca_qr.py --topic /camera_down --headless
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
class QRResult:
    detected: bool
    data: str                      # String teks isi decoding QR atau ID Marker
    code_type: str                 # "QR_CODE" atau "ARUCO"
    cx: int
    cy: int
    dx: int                        # Offset horizontal dari pusat kamera (cx - w/2)
    dy: int                        # Offset vertikal dari pusat kamera (cy - h/2)
    norm_dx: float                 # Offset X ternormalisasi [-1.0 .. 1.0]
    norm_dy: float                 # Offset Y ternormalisasi [-1.0 .. 1.0]
    error_dist: float              # Jarak euclidian error (pixel)
    bbox: Tuple[int, int, int, int]# (x, y, w, h)
    polygon: Optional[np.ndarray]  # 4 Titik pojok kontur/poligon
    is_centered: bool              # True jika |dx| <= tolerance dan |dy| <= tolerance
    annotated_frame: Optional[np.ndarray] = None


class QRCodeReader:
    """
    Detektor & Decoder QR Code serta ArUco Marker berbasis Pure OpenCV.
    """
    def __init__(self, tolerance_px: int = 25):
        self.tolerance_px = tolerance_px
        self.qr_detector = cv2.QRCodeDetector()

        # Inisialisasi ArUco detector (Dictionary DICT_4X4_50 sesuai arena simulasi)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict)
        else:
            self.aruco_detector = None

    def detect(self, frame_bgr: np.ndarray, draw: bool = True) -> QRResult:
        """
        Mendeteksi dan mendekode QR code atau Marker pada frame gambar BGR.
        """
        h, w = frame_bgr.shape[:2]
        center_frame_x = w // 2
        center_frame_y = h // 2

        annotated = frame_bgr.copy() if draw else None

        # ----------------------------------------------------------------------
        # 1. Coba Deteksi QR Code Standar (OpenCV QRCodeDetector)
        # ----------------------------------------------------------------------
        val, points, _ = self.qr_detector.detectAndDecode(frame_bgr)
        if val and points is not None and len(points) > 0:
            pts = points[0].astype(int)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            x, y, bw, bh = cv2.boundingRect(pts)

            dx = cx - center_frame_x
            dy = cy - center_frame_y
            norm_dx = float(dx) / float(center_frame_x) if center_frame_x > 0 else 0.0
            norm_dy = float(dy) / float(center_frame_y) if center_frame_y > 0 else 0.0
            error_dist = float(np.hypot(dx, dy))
            is_centered = (abs(dx) <= self.tolerance_px and abs(dy) <= self.tolerance_px)

            if draw and annotated is not None:
                self._draw_overlay(
                    annotated, pts, cx, cy, x, y, bw, bh, val, "QR_CODE",
                    center_frame_x, center_frame_y, dx, dy, error_dist, is_centered
                )

            return QRResult(
                detected=True,
                data=val,
                code_type="QR_CODE",
                cx=cx,
                cy=cy,
                dx=dx,
                dy=dy,
                norm_dx=norm_dx,
                norm_dy=norm_dy,
                error_dist=error_dist,
                bbox=(x, y, bw, bh),
                polygon=pts,
                is_centered=is_centered,
                annotated_frame=annotated
            )

        # ----------------------------------------------------------------------
        # 2. Fallback: Deteksi ArUco Marker (Bila QR code tidak ada/menggunakan ArUco)
        # ----------------------------------------------------------------------
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = (
            self.aruco_detector.detectMarkers(gray)
            if self.aruco_detector is not None
            else cv2.aruco.detectMarkers(gray, self.aruco_dict)
        )

        if ids is not None and len(ids) > 0:
            best_idx = 0
            pts = corners[best_idx][0].astype(int)
            marker_id = int(ids[best_idx][0])
            val_str = f"ARUCO_ID_{marker_id}"

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            x, y, bw, bh = cv2.boundingRect(pts)

            dx = cx - center_frame_x
            dy = cy - center_frame_y
            norm_dx = float(dx) / float(center_frame_x) if center_frame_x > 0 else 0.0
            norm_dy = float(dy) / float(center_frame_y) if center_frame_y > 0 else 0.0
            error_dist = float(np.hypot(dx, dy))
            is_centered = (abs(dx) <= self.tolerance_px and abs(dy) <= self.tolerance_px)

            if draw and annotated is not None:
                self._draw_overlay(
                    annotated, pts, cx, cy, x, y, bw, bh, val_str, "ARUCO",
                    center_frame_x, center_frame_y, dx, dy, error_dist, is_centered
                )

            return QRResult(
                detected=True,
                data=val_str,
                code_type="ARUCO",
                cx=cx,
                cy=cy,
                dx=dx,
                dy=dy,
                norm_dx=norm_dx,
                norm_dy=norm_dy,
                error_dist=error_dist,
                bbox=(x, y, bw, bh),
                polygon=pts,
                is_centered=is_centered,
                annotated_frame=annotated
            )

        # ----------------------------------------------------------------------
        # 3. Target Tidak Terdeteksi
        # ----------------------------------------------------------------------
        if draw and annotated is not None:
            self._draw_no_detection_overlay(annotated, center_frame_x, center_frame_y)

        return QRResult(
            detected=False,
            data="",
            code_type="NONE",
            cx=center_frame_x,
            cy=center_frame_y,
            dx=0,
            dy=0,
            norm_dx=0.0,
            norm_dy=0.0,
            error_dist=0.0,
            bbox=(0, 0, 0, 0),
            polygon=None,
            is_centered=False,
            annotated_frame=annotated
        )

    def _draw_overlay(
        self, img, pts, cx, cy, x, y, bw, bh, val, code_type,
        fx, fy, dx, dy, dist, is_centered
    ):
        h, w = img.shape[:2]

        # 1. Gambar poligon tepi QR/Marker
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)

        # 2. Gambar Crosshair & Toleransi Tengah Kamera
        cv2.line(img, (fx - 15, fy), (fx + 15, fy), (200, 200, 200), 1)
        cv2.line(img, (fx, fy - 15), (fx, fy + 15), (200, 200, 200), 1)
        cv2.circle(img, (fx, fy), self.tolerance_px, (255, 255, 255), 1)

        # 3. Vektor error
        vec_color = (0, 255, 255) if not is_centered else (0, 255, 0)
        cv2.arrowedLine(img, (fx, fy), (cx, cy), vec_color, 2, tipLength=0.15)

        # 4. Label teks di dekat objek
        cv2.putText(img, f"{code_type}: {val}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # 5. Top Banner HUD
        status_bg = (0, 150, 0) if is_centered else (0, 100, 200)
        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (160, 36), status_bg, -1)

        status_txt = "CENTERED" if is_centered else "LOCKED"
        cv2.putText(img, status_txt, (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        info_txt = f"[{code_type}] '{val}' | Err: dX={dx:+d} dY={dy:+d} (dist={dist:.1f}px)"
        cv2.putText(img, info_txt, (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    def _draw_no_detection_overlay(self, img, fx, fy):
        h, w = img.shape[:2]
        cv2.line(img, (fx - 15, fy), (fx + 15, fy), (100, 100, 100), 1)
        cv2.line(img, (fx, fy - 15), (fx, fy + 15), (100, 100, 100), 1)
        cv2.circle(img, (fx, fy), self.tolerance_px, (80, 80, 80), 1)

        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (160, 36), (0, 0, 160), -1)
        cv2.putText(img, "SEARCHING", (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, "QR Code / Marker Belum Terdeteksi", (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)


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
    node = Node("detector_baca_qr")
    detector = QRCodeReader()

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
    print(f"[OK] QRCodeReader subscribe ke topik: {topic}")
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
                        print(f"[{status}] Type={res.code_type} Data='{res.data}' | Pos=({res.cx},{res.cy}) dx={res.dx:+d}, dy={res.dy:+d}")
                    else:
                        print("[-] QR / Marker: Belum Terdeteksi...")
                    last_log_time = now

                if has_display and res.annotated_frame is not None:
                    cv2.imshow("Scanner QR & Marker", res.annotated_frame)
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
    parser = argparse.ArgumentParser(description="Deteksi & Pembacaan Kode QR / ArUco")
    parser.add_argument("--topic", default="/camera_front", help="Topik kamera ROS 2 (default: /camera_front)")
    parser.add_argument("--headless", action="store_true", help="Jalankan tanpa window GUI OpenCV")
    args = parser.parse_args()

    run_ros2_node(args.topic, args.headless)


if __name__ == "__main__":
    main()

