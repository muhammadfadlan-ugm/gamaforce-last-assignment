"""
Package Deteksi Vision OpenCV untuk Misi Drone Oprec Gamaforce
"""

from .centering_hijau import GreenRingDetector, GreenCenteringResult
from .centering_pink import PinkRingDetector, PinkCenteringResult
from .donat_merah import RedDonutDetector, RedDonutResult
from .kamikaze import KamikazeDetector, KamikazeResult, DetectedTarget

__all__ = [
    "GreenRingDetector",
    "GreenCenteringResult",
    "PinkRingDetector",
    "PinkCenteringResult",
    "RedDonutDetector",
    "RedDonutResult",
    "KamikazeDetector",
    "KamikazeResult",
    "DetectedTarget",
]

