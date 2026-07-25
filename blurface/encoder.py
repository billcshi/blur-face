"""Fail-closed FFmpeg encoder with atomic output commit."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import imageio_ffmpeg


class EncoderError(RuntimeError):
    """Raised when FFmpeg cannot produce a verified output."""


def _resolve_ffmpeg(explicit: str | Path | None = None) -> str:
    """Prefer an explicit or system FFmpeg so Windows NVENC can be used."""
    if explicit is not None:
        executable = Path(explicit).expanduser().resolve()
        if not executable.is_file():
            raise EncoderError(f"FFmpeg executable does not exist: {executable}")
        return str(executable)
    system_ffmpeg = shutil.which("ffmpeg")
    return system_ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()


def _can_encode_nvenc(ffmpeg_exe: str) -> bool:
    """Probe the actual NVENC runtime, not just FFmpeg's encoder list."""
    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        "color=size=16x16:rate=1",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=10, check=False)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _has_nvenc(ffmpeg_exe: str) -> bool:
    """Backward-compatible alias for the stronger runtime probe."""
    return _can_encode_nvenc(ffmpeg_exe)


class FFmpegEncoder:
    """Stream BGR frames to FFmpeg and atomically commit a valid result."""

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        audio_source: str | Path,
        use_nvenc: bool = True,
        overwrite: bool = False,
        ffmpeg_exe: str | Path | None = None,
    ):
        if width <= 0 or height <= 0 or fps <= 0:
            raise EncoderError("encoder dimensions and FPS must be positive")
        self.output_path = Path(output_path).expanduser().resolve()
        self.audio_source = Path(audio_source).expanduser().resolve()
        self.overwrite = overwrite
        if self.output_path.exists() and not self.overwrite:
            raise EncoderError(
                f"output already exists: {self.output_path} (use --overwrite)"
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = self.output_path.suffix or ".mp4"
        self.temp_path = self.output_path.parent / (
            f".{self.output_path.stem}.{uuid.uuid4().hex}.partial{suffix}"
        )
        self.ffmpeg = _resolve_ffmpeg(ffmpeg_exe)
        self.codec = (
            "h264_nvenc" if use_nvenc and _can_encode_nvenc(self.ffmpeg) else "libx264"
        )
        if self.codec == "h264_nvenc":
            video_codec = ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "23"]
        else:
            video_codec = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
        self.cmd = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-i",
            str(self.audio_source),
            *video_codec,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-movflags",
            "+faststart",
            str(self.temp_path),
        ]
        # This file intentionally spans the encoder object's lifetime.
        self._stderr = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        try:
            self.proc = subprocess.Popen(
                self.cmd, stdin=subprocess.PIPE, stderr=self._stderr
            )
        except Exception:
            self._stderr.close()
            self.temp_path.unlink(missing_ok=True)
            raise
        self._closed = False
        self._committed = False

    def __enter__(self) -> FFmpegEncoder:  # noqa: PYI034 - Python 3.10 support
        return self

    def write(self, data: bytes) -> None:
        if self._closed or self.proc.stdin is None:
            raise EncoderError("cannot write to a closed encoder")
        try:
            self.proc.stdin.write(data)
        except BrokenPipeError as exc:
            raise EncoderError(
                self._failure_message("FFmpeg stopped accepting frames")
            ) from exc

    def _read_stderr(self) -> str:
        self._stderr.flush()
        self._stderr.seek(0)
        return self._stderr.read().decode("utf-8", errors="replace")[-8000:].strip()

    def _failure_message(self, prefix: str) -> str:
        details = self._read_stderr()
        return f"{prefix}: {details}" if details else prefix

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.proc.stdin is not None and not self.proc.stdin.closed:
                try:
                    self.proc.stdin.close()
                except BrokenPipeError:
                    pass
            return_code = self.proc.wait()
            if return_code != 0:
                raise EncoderError(
                    self._failure_message(f"FFmpeg exited with status {return_code}")
                )
            if not self.temp_path.is_file() or self.temp_path.stat().st_size == 0:
                raise EncoderError("FFmpeg reported success but produced no output")
            if self.output_path.exists() and not self.overwrite:
                raise EncoderError(
                    f"output appeared during processing and was not replaced: "
                    f"{self.output_path}"
                )
            os.replace(self.temp_path, self.output_path)
            self._committed = True
        finally:
            self._stderr.close()
            if not self._committed:
                self.temp_path.unlink(missing_ok=True)

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._stderr.close()
        self.temp_path.unlink(missing_ok=True)

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
