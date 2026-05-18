"""
blurface — Face blur with tracking + prediction.

blurface/
├── __init__.py      # Package init
├── cli.py           # Argument parsing
├── detector.py      # YOLO face detection
├── tracker.py       # Face tracker (centroid + smoothing + prediction)
├── renderer.py      # Blur and debug-draw
├── encoder.py       # ffmpeg H.264 encoding
└── profiler.py      # Per-phase timing
"""

from .cli import parse_args, get_thresh, parse_time_thresh, parse_exclude_ids
from .detector import FaceDetector
from .tracker import FaceTracker
from .renderer import apply_blur, draw_debug_box, color_for
from .encoder import FFmpegEncoder
from .profiler import Profiler
