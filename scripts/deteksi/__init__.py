"""
Package Deteksi Vision OpenCV untuk Misi Drone Oprec Gamaforce
"""

from .centering_hijau import GreenRingDetector, GreenCenteringResult
from .centering_pink import PinkRingDetector, PinkCenteringResult
from .donat_merah import RedDonutDetector, RedDonutResult
from .kamikaze import KamikazeDetector, KamikazeResult, DetectedTarget

from .baca_qr import QRCodeReader, QRResult
from .belahketupat import (
    BelahKetupatDetector,
    BelahKetupatResult,
    BelahKetupatBiruDetector,
    BelahKetupatHijauDetector,
    BelahKetupatPinkDetector,
    COLOR_PRESETS,
)
from .obstacle_merah import ObstacleMerahDetector, ObstacleMerahResult
from .obstacle_kuning import ObstacleKuningDetector, ObstacleKuningResult

__all__ = [
    # Detektor Ring & Bulls Eye Asli
    "GreenRingDetector",
    "GreenCenteringResult",
    "PinkRingDetector",
    "PinkCenteringResult",
    "RedDonutDetector",
    "RedDonutResult",
    "KamikazeDetector",
    "KamikazeResult",
    "DetectedTarget",
    # Detektor QR & Marker
    "QRCodeReader",
    "QRResult",
    # Detektor Belah Ketupat (Unified & Aliases)
    "BelahKetupatDetector",
    "BelahKetupatResult",
    "BelahKetupatBiruDetector",
    "BelahKetupatHijauDetector",
    "BelahKetupatPinkDetector",
    "COLOR_PRESETS",
    # Detektor Obstacle Merah & Kuning
    "ObstacleMerahDetector",
    "ObstacleMerahResult",
    "ObstacleKuningDetector",
    "ObstacleKuningResult",
]
