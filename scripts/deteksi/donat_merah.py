#!/usr/bin/env python3
"""
DETEKSI & ALIGNMENT DONAT MERAH (CAMERA FRONT)
==============================================
Modul untuk mendeteksi rintangan donat/torus merah di depan drone (kamera depan).
Mendeteksi cincin luar (outer torus) dan lubang tengah (inner hole).
Menghitung titik tengah lubang (cx, cy) serta error offset (dx, dy) terhadap pusat kamera
agar drone dapat meluruskan arah terbang (alignment yaw & altitude) dan menembus donat.

Fitur:
- Pure OpenCV (Dual-range HSV Red Color Masking, Contour Hierarchy RETR_CCOMP untuk hole detection).
- Menghitung titik tembus optimal (pusat lubang cincin).
- Modular: bisa di-import oleh skrip misi (from deteksi.donat_merah import RedDonutDetector).
- Standalone: visualisasi real-time OpenCV HUD dengan indikator kesiapan lolos (ready to pass).
- ROS 2 Image decoder aman tanpa dependensi cv_bridge.

Penggunaan Standalone:
  python3 scripts/deteksi/donat_merah.py
  python3 scripts/deteksi/donat_merah.py --topic /camera_front
  python3 scripts/deteksi/donat_merah.py --topic /camera_front --headless
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
class RedDonutResult:
    detected: bool
    cx: int               # Titik tengah lubang tembus (pixel X)
    cy: int               # Titik tengah lubang tembus (pixel Y)
    outer_radius: float   # Radius luar donat (pixel)
    inner_radius: float   # Radius lubang donat (pixel)
    outer_area: float     # Luas kontur donat (pixel^2)
    dx: int               # Offset X dari tengah frame: cx - frame_w/2
    dy: int               # Offset Y dari tengah frame: cy - frame_h/2
    norm_dx: float        # Offset X ternormalisasi [-1.0 .. 1.0]
    norm_dy: float        # Offset Y ternormalisasi [-1.0 .. 1.0]
    error_dist: float     # Jarak euclidian offset (pixel)
    has_hole: bool        # True jika lubang tengah berhasil terdeteksi jelas
    is_aligned: bool      # True jika drone sudah sejajar dengan lubang (|dx| & |dy| <= tolerance)
    ready_to_pass: bool   # True jika sudah aligned dan ukuran donat cukup dekat
    annotated_frame: Optional[np.ndarray] = None


class RedDonutDetector:
    """
    Detektor donat/torus merah berbasis dual-range HSV masking dan contour hierarchy.
    """
    def __init__(
        self,
        lower_red1: Tuple[int, int, int] = (0, 80, 50),
        upper_red1: Tuple[int, int, int] = (10, 255, 255),
        lower_red2: Tuple[int, int, int] = (160, 80, 50),
        upper_red2: Tuple[int, int, int] = (180, 255, 255),
        min_outer_area: float = 800.0,
        tolerance_px: int = 30,
        min_pass_radius: float = 60.0,
    ):
        self.lower_red1 = np.array(lower_red1, dtype=np.uint8)
        self.upper_red1 = np.array(upper_red1, dtype=np.uint8)
        self.lower_red2 = np.array(lower_red2, dtype=np.uint8)
        self.upper_red2 = np.array(upper_red2, dtype=np.uint8)
        self.min_outer_area = min_outer_area
        self.tolerance_px = tolerance_px
        self.min_pass_radius = min_pass_radius

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def detect(self, frame_bgr: np.ndarray, draw: bool = True) -> RedDonutResult:
        """
        Mendeteksi donat merah dan lubang tembusnya pada frame gambar BGR.
        """
        h, w = frame_bgr.shape[:2]
        center_frame_x = w // 2
        center_frame_y = h // 2

        # 1. Konversi HSV dan reduksi noise
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 2. Dual-range thresholding untuk warna merah (wrap around Hue 0 dan 180)
        mask1 = cv2.inRange(blurred, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(blurred, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 3. Morfologi untuk menghilangkan bercak dan merapatkan kontur donat
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        # 4. Pencarian Kontur 2-level Hierarchy (RETR_CCOMP)
        # hierarchy: [Next, Previous, First_Child, Parent]
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        annotated = frame_bgr.copy() if draw else None

        if not contours or hierarchy is None:
            if draw and annotated is not None:
                self._draw_no_detection_overlay(annotated, center_frame_x, center_frame_y)
            return self._build_empty_result(center_frame_x, center_frame_y, annotated)

        hier = hierarchy[0]
        best_outer_idx = -1
        best_outer_area = 0.0

        # Cari kontur luar donat terbesar (Parent == -1)
        for idx, h_info in enumerate(hier):
            if h_info[3] == -1:  # Kontur eksternal
                area = cv2.contourArea(contours[idx])
                if area > best_outer_area and area >= self.min_outer_area:
                    best_outer_area = area
                    best_outer_idx = idx

        if best_outer_idx == -1:
            if draw and annotated is not None:
                self._draw_no_detection_overlay(annotated, center_frame_x, center_frame_y)
            return self._build_empty_result(center_frame_x, center_frame_y, annotated)

        outer_cnt = contours[best_outer_idx]
        (outer_cx_f, outer_cy_f), outer_radius = cv2.minEnclosingCircle(outer_cnt)

        # 5. Cari lubang donat (Child contour di dalam best_outer_idx)
        best_hole_idx = -1
        best_hole_area = 0.0
        child_idx = hier[best_outer_idx][2]

        while child_idx != -1:
            hole_area = cv2.contourArea(contours[child_idx])
            if hole_area > best_hole_area and hole_area >= 50:
                best_hole_area = hole_area
                best_hole_idx = child_idx
            child_idx = hier[child_idx][0]  # Next sibling

        has_hole = (best_hole_idx != -1)
        inner_radius = 0.0

        if has_hole:
            # Titik tengah diambil dari pusat lubang donat
            hole_cnt = contours[best_hole_idx]
            M_hole = cv2.moments(hole_cnt)
            if M_hole["m00"] > 0:
                cx = int(M_hole["m10"] / M_hole["m00"])
                cy = int(M_hole["m01"] / M_hole["m00"])
            else:
                (hcx_f, hcy_f), _ = cv2.minEnclosingCircle(hole_cnt)
                cx, cy = int(hcx_f), int(hcy_f)
            (_, _), inner_radius = cv2.minEnclosingCircle(hole_cnt)
        else:
            # Fallback ke titik tengah kontur luar
            M_outer = cv2.moments(outer_cnt)
            if M_outer["m00"] > 0:
                cx = int(M_outer["m10"] / M_outer["m00"])
                cy = int(M_outer["m01"] / M_outer["m00"])
            else:
                cx, cy = int(outer_cx_f), int(outer_cy_f)

        # 6. Hitung error offset terhadap pusat frame kamera
        dx = cx - center_frame_x
        dy = cy - center_frame_y
        norm_dx = float(dx) / float(center_frame_x) if center_frame_x > 0 else 0.0
        norm_dy = float(dy) / float(center_frame_y) if center_frame_y > 0 else 0.0
        error_dist = float(np.hypot(dx, dy))
        is_aligned = (abs(dx) <= self.tolerance_px and abs(dy) <= self.tolerance_px)
        ready_to_pass = is_aligned and (outer_radius >= self.min_pass_radius or best_outer_area >= 6000.0)

        if draw and annotated is not None:
            hole_cnt_draw = contours[best_hole_idx] if has_hole else None
            self._draw_overlay(
                annotated, outer_cnt, hole_cnt_draw, cx, cy,
                outer_radius, inner_radius, best_outer_area,
                center_frame_x, center_frame_y, dx, dy, error_dist,
                has_hole, is_aligned, ready_to_pass
            )

        return RedDonutResult(
            detected=True,
            cx=cx,
            cy=cy,
            outer_radius=float(outer_radius),
            inner_radius=float(inner_radius),
            outer_area=float(best_outer_area),
            dx=dx,
            dy=dy,
            norm_dx=norm_dx,
            norm_dy=norm_dy,
            error_dist=error_dist,
            has_hole=has_hole,
            is_aligned=is_aligned,
            ready_to_pass=ready_to_pass,
            annotated_frame=annotated
        )

    def _build_empty_result(self, fx, fy, annotated):
        return RedDonutResult(
            detected=False,
            cx=fx,
            cy=fy,
            outer_radius=0.0,
            inner_radius=0.0,
            outer_area=0.0,
            dx=0,
            dy=0,
            norm_dx=0.0,
            norm_dy=0.0,
            error_dist=0.0,
            has_hole=False,
            is_aligned=False,
            ready_to_pass=False,
            annotated_frame=annotated
        )

    def _draw_overlay(
        self, img, outer_cnt, hole_cnt, cx, cy,
        outer_r, inner_r, area,
        fx, fy, dx, dy, dist,
        has_hole, is_aligned, ready_to_pass
    ):
        h, w = img.shape[:2]

        # 1. Gambar cincin luar (merah)
        cv2.drawContours(img, [outer_cnt], -1, (0, 0, 255), 2)
        cv2.circle(img, (int(cx), int(cy)), int(outer_r), (0, 0, 200), 1)

        # 2. Gambar lubang dalam jika ada (cyan)
        if hole_cnt is not None:
            cv2.drawContours(img, [hole_cnt], -1, (255, 255, 0), 2)
            cv2.circle(img, (cx, cy), int(inner_r), (255, 200, 0), 1)

        # 3. Titik tengah lubang tembus
        cv2.circle(img, (cx, cy), 7, (0, 255, 255), -1)
        cv2.circle(img, (cx, cy), 3, (0, 0, 0), -1)

        # 4. Crosshair frame kamera
        cv2.line(img, (fx - 15, fy), (fx + 15, fy), (200, 200, 200), 1)
        cv2.line(img, (fx, fy - 15), (fx, fy + 15), (200, 200, 200), 1)
        cv2.circle(img, (fx, fy), self.tolerance_px, (255, 255, 255), 1)

        # 5. Garis panah error alignment
        vec_col = (0, 255, 0) if is_aligned else (0, 165, 255)
        cv2.arrowedLine(img, (fx, fy), (cx, cy), vec_col, 2, tipLength=0.15)

        # 6. Status Banner
        if ready_to_pass:
            status_bg = (0, 180, 0)
            status_txt = "PASS READY"
        elif is_aligned:
            status_bg = (0, 140, 70)
            status_txt = "ALIGNED"
        else:
            status_bg = (0, 60, 200)
            status_txt = "ALIGNING"

        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (160, 36), status_bg, -1)
        cv2.putText(img, status_txt, (12, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        hole_status = "HOLE: OK" if has_hole else "HOLE: ?"
        info_txt = f"{hole_status} | dX={dx:+d} dY={dy:+d} (dist={dist:.1f}px) | R_out={outer_r:.0f} R_in={inner_r:.0f}"
        cv2.putText(img, info_txt, (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    def _draw_no_detection_overlay(self, img, fx, fy):
        h, w = img.shape[:2]
        cv2.line(img, (fx - 15, fy), (fx + 15, fy), (100, 100, 100), 1)
        cv2.line(img, (fx, fy - 15), (fx, fy + 15), (100, 100, 100), 1)

        cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (160, 36), (0, 0, 160), -1)
        cv2.putText(img, "SEARCHING", (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, "Donat Merah Belum Terdeteksi", (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)


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
    node = Node("detector_donat_merah")
    detector = RedDonutDetector()

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
    print(f"[OK] RedDonutDetector subscribe ke: {topic}")
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
                        status = "🚀 PASS READY" if res.ready_to_pass else ("🎯 ALIGNED" if res.is_aligned else "🔍 ALIGNING")
                        hole_txt = "Hole OK" if res.has_hole else "Outer Center"
                        print(f"[{status}] {hole_txt} | cx={res.cx:3d}, cy={res.cy:3d} | dx={res.dx:+3d}, dy={res.dy:+3d} | R_out={res.outer_radius:5.1f} R_in={res.inner_radius:4.1f}")
                    else:
                        print("[-] Donat Merah: Belum Terdeteksi...")
                    last_log_time = now

                if has_display and res.annotated_frame is not None:
                    cv2.imshow("Deteksi Donat Merah", res.annotated_frame)
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
    parser = argparse.ArgumentParser(description="Deteksi & Alignment Donat Merah")
    parser.add_argument("--topic", default="/camera_front", help="Topic kamera ROS 2 (default: /camera_front)")
    parser.add_argument("--headless", action="store_true", help="Jalankan tanpa window GUI OpenCV")
    args = parser.parse_args()

    run_ros2_node(args.topic, args.headless)


if __name__ == "__main__":
    main()

