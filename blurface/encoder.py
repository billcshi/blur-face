"""
blurface.encoder — ffmpeg pipe wrapper for H.264 encoding with audio.

Supports both CPU (libx264) and GPU (h264_nvenc) encoding.
"""
import sys
import os
import subprocess
import shutil
import imageio_ffmpeg


def _find_ffmpeg() -> str:
    """Find ffmpeg binary, handling PyInstaller frozen bundles."""
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (FileNotFoundError, RuntimeError, ImportError):
        pass

    # PyInstaller frozen: search bundled location
    if getattr(sys, 'frozen', False):
        bundle_dir = os.path.dirname(sys.executable)
        # imageio_ffmpeg bundles versioned names like ffmpeg-win64-v4.2.2.exe
        import glob as _glob
        for base in [
            os.path.join(bundle_dir, "imageio_ffmpeg", "bin"),
        ]:
            ffmpeg_candidates = _glob.glob(os.path.join(base, "ffmpeg*"))
            if ffmpeg_candidates:
                return ffmpeg_candidates[0]

    # Fallback: try system ffmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    raise RuntimeError("ffmpeg not found. Install ffmpeg or imageio-ffmpeg.")


def _has_nvenc(ffmpeg_exe: str) -> bool:
    """Check if ffmpeg was built with NVENC support."""
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-encoders"], capture_output=True, text=True, timeout=5
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


class FFmpegEncoder:
    """
    Opens a subprocess pipe to ffmpeg for H.264 encoding.

    Usage:
        enc = FFmpegEncoder("output.mp4", 1920, 1080, 30, "input.mov")
        enc.write_frame(frame_bytes)  # call per frame
        enc.close()                   # finalize
    """

    def __init__(self, output_path: str, width: int, height: int,
                 fps: float, audio_source: str, use_nvenc: bool = True):
        ff = _find_ffmpeg()

        if use_nvenc and _has_nvenc(ff):
            vcodec = ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "23"]
            print("[Encoder] NVENC GPU encoding")
        else:
            if use_nvenc:
                print("[Encoder] NVENC not available, falling back to CPU (libx264)")
            vcodec = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]

        self.cmd = [
            ff, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",                       # video from stdin
            "-i", audio_source,              # audio from original
            *vcodec,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-movflags", "+faststart",
            output_path,
        ]
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

    def write(self, data: bytes):
        self.proc.stdin.write(data)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
