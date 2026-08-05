"""Local-only browser UI for blur-face."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import io
import importlib.util
from http.cookies import CookieError, SimpleCookie
import json
import os
import re
import secrets
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .cli import parse_args
from .config import ImageBatchConfig
from .model_store import (
    MODEL_SPECS,
    OPTIONAL_YOLO_MODELS,
    file_digest,
    model_directories,
)

_PROGRESS = re.compile(r"\b(\d+)/(\d+)\s+\((\d+)%\)")
_UI_FILE = Path(__file__).with_name("ui") / "index.html"


def _is_windows() -> bool:
    """Return the runtime platform without requiring tests to mutate os.name."""
    return os.name == "nt"


class _UiDirectoryAnchor:
    """Keep a UI job parent stable across creation and cleanup."""

    def __init__(self, path: Path, expected: tuple[int, int]):
        self.dir_fd: int | None = None
        self._windows_handle: int | None = None
        if _is_windows():
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_file = kernel32.CreateFileW
            create_file.argtypes = (
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            )
            create_file.restype = wintypes.HANDLE
            handle = create_file(
                str(path),
                0x0001,
                0x0001 | 0x0002,
                None,
                3,
                0x02000000 | 0x00200000,
                None,
            )
            if handle in {None, ctypes.c_void_p(-1).value}:
                raise RuntimeError(f"cannot anchor UI job parent: {path}")
            self._windows_handle = int(handle)
            try:
                if JobManager._job_directory_identity(path) != expected:
                    raise RuntimeError(f"UI job parent changed: {path}")
            except Exception:
                self.close()
                raise
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise RuntimeError(f"cannot anchor UI job parent: {path}") from exc
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != expected:
            os.close(descriptor)
            raise RuntimeError(f"UI job parent changed: {path}")
        self.dir_fd = descriptor

    @classmethod
    def open_relative(
        cls,
        parent_fd: int,
        name: str,
        expected: tuple[int, int],
    ) -> _UiDirectoryAnchor:
        anchor = cls.__new__(cls)
        anchor.dir_fd = None
        anchor._windows_handle = None
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            anchor.dir_fd = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise RuntimeError(f"cannot anchor UI job directory: {name}") from exc
        info = os.fstat(anchor.dir_fd)
        if (info.st_dev, info.st_ino) != expected:
            anchor.close()
            raise RuntimeError(f"UI job directory changed: {name}")
        return anchor

    def close(self) -> None:
        if self.dir_fd is not None:
            os.close(self.dir_fd)
            self.dir_fd = None
        if self._windows_handle is not None:
            import ctypes
            from ctypes import wintypes

            close_handle = ctypes.WinDLL(
                "kernel32", use_last_error=True
            ).CloseHandle
            close_handle.argtypes = (wintypes.HANDLE,)
            close_handle.restype = wintypes.BOOL
            close_handle(self._windows_handle)
            self._windows_handle = None


def _number(payload: dict, key: str, cast):
    try:
        return cast(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}") from exc


def discover_local_models() -> list[str]:
    """Return local YuNet ONNX models from every configured model directory."""
    models: set[Path] = set()
    for directory in model_directories():
        if directory.is_dir():
            models.update(
                path.resolve()
                for path in directory.glob("*.onnx")
                if path.is_file()
            )
    return [str(path) for path in sorted(models, key=lambda path: path.name.lower())]


def discover_local_yolo_models() -> list[str]:
    """Return only pinned optional YOLO weights with the expected digest."""
    models: set[Path] = set()
    for directory in model_directories():
        if directory.is_dir():
            for path in directory.iterdir():
                name = path.name.lower()
                if (
                    not path.is_file()
                    or path.suffix.lower() != ".pt"
                    or name not in OPTIONAL_YOLO_MODELS
                ):
                    continue
                try:
                    if file_digest(path) == MODEL_SPECS[name][1]:
                        models.add(path.resolve())
                except OSError:
                    continue
    return [str(path) for path in sorted(models, key=lambda path: path.name.lower())]


def optional_yolo_available(models: list[str] | None = None) -> bool:
    """Report a complete opt-in install, not merely an unrelated PT file."""
    candidates = discover_local_yolo_models() if models is None else models
    return bool(candidates) and importlib.util.find_spec("ultralytics") is not None


def discover_local_sam2_models() -> list[str]:
    """Return local Hugging Face SAM 2.1 model directories."""
    models: set[Path] = set()
    for directory in model_directories():
        if not directory.is_dir():
            continue
        for config_file in directory.rglob("config.json"):
            model_directory = config_file.parent
            name = model_directory.name.lower()
            if "sam2" in name or "sam2" in config_file.read_text(
                encoding="utf-8", errors="ignore"
            ).lower():
                models.add(model_directory.resolve())
    return [str(path) for path in sorted(models, key=lambda path: str(path).lower())]


def _payload_inputs(payload: dict) -> tuple[str, ...]:
    if payload.get("media_type") == "images" or "inputs" in payload:
        values = payload.get("inputs", [])
        if not isinstance(values, list) or not values:
            raise ValueError("at least one input image is required")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("every input image path must be non-empty")
        return tuple(value.strip() for value in values)
    value = str(payload.get("input", "")).strip()
    if not value:
        raise ValueError("input path is required")
    return (value,)


def build_command(
    payload: dict, *, input_manifest: Path | None = None
) -> list[str]:
    """Validate a UI request and translate it into the authoritative CLI."""
    inputs = _payload_inputs(payload)
    if not str(payload.get("output", "")).strip():
        raise ValueError("output path is required")
    image_mode = payload.get("media_type") == "images" or "inputs" in payload
    if not image_mode:
        input_path = Path(inputs[0]).expanduser()
        if not input_path.is_file():
            raise ValueError(f"input video does not exist: {input_path}")
    arguments = [
        *inputs,
        "-o",
        str(payload["output"]),
        "--model",
        str(payload.get("model", "face_detection_yunet_2023mar.onnx")),
        "--detector",
        str(payload.get("detector", "yunet")),
        "--thresh",
        str(_number(payload, "threshold", float)),
        "--mask-scale",
        str(_number(payload, "mask_scale", float)),
        "--mask-shape",
        str(payload.get("mask_shape", "")),
        "--blur-strategy",
        str(payload.get("blur_strategy", "")),
        "--blur-kernel",
        str(_number(payload, "blur_kernel", int)),
        "--blur-kernel-min",
        str(_number(payload, "blur_kernel_min", int)),
        "--min-face-size",
        str(_number(payload, "min_face_size", int)),
        "--preset",
        str(payload.get("preset", "quality")),
        "--mask-engine",
        str(payload.get("mask_engine", "geometric")),
        "--sam-mask-expansion",
        str(payload.get("sam_mask_expansion", 0.12)),
        "--segmentation-combine",
        str(payload.get("segmentation_combine", "union")),
        "--sam2-model",
        str(payload.get("sam2_model", "facebook/sam2.1-hiera-base-plus")),
        "--sam2-refresh-interval",
        str(payload.get("sam2_refresh_interval", 15)),
        "--backfill-frames",
        str(payload.get("backfill_frames", 10)),
        "--release-hold-frames",
        str(payload.get("release_hold_frames", 5)),
        "--scene-cut-sensitivity",
        str(payload.get("scene_cut_sensitivity", 0.55)),
        "--temporal-storage-limit-mb",
        str(payload.get("temporal_storage_limit_mb", 4096)),
        "--device",
        str(payload.get("device", "auto")),
    ]
    arguments.append(
        "--temporal-stabilization"
        if bool(payload.get("temporal_stabilization", True))
        else "--no-temporal-stabilization"
    )
    if bool(payload.get("mask_preview", False)):
        arguments.append("--mask-preview")
    if bool(payload.get("overwrite", True)):
        arguments.append("--overwrite")
    if bool(payload.get("offline", False)):
        arguments.append("--offline")
    if bool(payload.get("no_nvenc", False)):
        arguments.append("--no-nvenc")
    parse_errors = io.StringIO()
    with redirect_stderr(parse_errors):
        try:
            request = parse_args(arguments)
        except SystemExit as exc:
            detail = parse_errors.getvalue().strip().splitlines()
            message = (
                detail[-1].removeprefix("blur-face: error: ")
                if detail
                else "invalid processing options or input/output paths"
            )
            raise ValueError(message) from exc
    if image_mode and not isinstance(request, ImageBatchConfig):
        raise ValueError("image mode accepts only supported image inputs")
    if not image_mode and isinstance(request, ImageBatchConfig):
        raise ValueError("video mode accepts one video input")
    emitted_inputs = (
        ["--input-list", str(input_manifest)]
        if input_manifest is not None
        else list(inputs)
    )
    return [
        sys.executable,
        "-u",
        "-m",
        "blurface",
        *emitted_inputs,
        *arguments[len(inputs) :],
    ]


class JobManager:
    """Run one blur job at a time and expose a bounded status snapshot."""

    def __init__(self):
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._status = "idle"
        self._progress = 0
        self._current = 0
        self._total = 0
        self._return_code: int | None = None
        self._media_type = "video"
        self._cancel_requested = False
        self._shutdown_requested = False
        self._worker: threading.Thread | None = None
        self._job_temp: Path | None = None
        self._job_temp_identity: tuple[int, int] | None = None
        self._job_parent_anchor: _UiDirectoryAnchor | None = None
        self._job_anchor: _UiDirectoryAnchor | None = None
        self._logs: deque[str] = deque(maxlen=500)

    def start(self, payload: dict) -> None:
        command = build_command(payload)
        image_mode = payload.get("media_type") == "images" or "inputs" in payload
        output_path = Path(str(payload.get("output", ""))).expanduser().resolve()
        output_parent = output_path if image_mode else output_path.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._status in {"starting", "running", "cancelling"}:
                raise RuntimeError("a video is already being processed")
            self._status = "starting"
            self._progress = self._current = self._total = 0
            self._return_code = None
            self._media_type = "images" if image_mode else "video"
            self._cancel_requested = False
            self._shutdown_requested = False
            self._logs.clear()
        parent_anchor: _UiDirectoryAnchor | None = None
        try:
            parent_identity = self._job_directory_identity(output_parent)
            parent_anchor = _UiDirectoryAnchor(output_parent, parent_identity)
            if parent_anchor.dir_fd is not None:
                for _attempt in range(100):
                    job_name = f".blur-face-job-{secrets.token_hex(16)}"
                    try:
                        os.mkdir(job_name, 0o700, dir_fd=parent_anchor.dir_fd)
                    except FileExistsError:
                        continue
                    job_temp = output_parent / job_name
                    info = os.stat(
                        job_name,
                        dir_fd=parent_anchor.dir_fd,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISDIR(info.st_mode):
                        raise RuntimeError(
                            f"new UI job path is not an ordinary directory: {job_name}"
                        )
                    job_temp_identity = (info.st_dev, info.st_ino)
                    job_anchor = _UiDirectoryAnchor.open_relative(
                        parent_anchor.dir_fd,
                        job_name,
                        job_temp_identity,
                    )
                    break
                else:
                    raise RuntimeError("cannot allocate a unique UI job directory")
            else:
                job_temp = Path(
                    os.path.abspath(
                        tempfile.mkdtemp(
                            prefix=".blur-face-job-", dir=output_parent
                        )
                    )
                )
                job_temp_identity = self._job_directory_identity(job_temp)
                job_anchor = _UiDirectoryAnchor(
                    job_temp, job_temp_identity
                )
        except Exception:
            try:
                if parent_anchor is not None:
                    parent_anchor.close()
            finally:
                with self._lock:
                    self._status = "failed"
            raise
        try:
            if image_mode:
                manifest = job_temp / "inputs.json"
                self._write_job_manifest(
                    manifest,
                    json.dumps(
                        list(_payload_inputs(payload)), ensure_ascii=False
                    ),
                    job_anchor,
                )
                command = build_command(payload, input_manifest=manifest)
        except Exception:
            try:
                self._cleanup_job_temp(
                    job_temp, job_temp_identity, parent_anchor, job_anchor
                )
            finally:
                job_anchor.close()
                parent_anchor.close()
            with self._lock:
                self._status = "failed"
            raise
        command.extend(["--job-temp-dir", str(job_temp)])
        worker = threading.Thread(
            target=self._run,
            args=(
                command,
                job_temp,
                job_temp_identity,
                parent_anchor,
                job_anchor,
            ),
            name="blur-face-ui-job",
            daemon=True,
        )
        with self._lock:
            self._worker = worker
            self._job_temp = job_temp
            self._job_temp_identity = job_temp_identity
            self._job_parent_anchor = parent_anchor
            self._job_anchor = job_anchor
        try:
            worker.start()
        except Exception:
            try:
                self._cleanup_job_temp(
                    job_temp, job_temp_identity, parent_anchor, job_anchor
                )
            finally:
                job_anchor.close()
                parent_anchor.close()
            with self._lock:
                self._worker = None
                self._job_temp = None
                self._job_temp_identity = None
                self._job_parent_anchor = None
                self._job_anchor = None
                self._status = "failed"
            raise

    def _run(
        self,
        command: list[str],
        job_temp: Path,
        job_temp_identity: tuple[int, int],
        parent_anchor: _UiDirectoryAnchor,
        job_anchor: _UiDirectoryAnchor,
    ) -> None:
        creationflags = 0
        kwargs: dict = {}
        if _is_windows():
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        child_environment = os.environ.copy()
        # The reader below is deliberately UTF-8. Match the child's redirected
        # streams so cp1252-representable filenames are not silently mangled.
        child_environment["PYTHONIOENCODING"] = "utf-8:backslashreplace"
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                env=child_environment,
                **kwargs,
            )
            with self._lock:
                self._process = process
                should_cancel = self._shutdown_requested
                self._status = "cancelling" if should_cancel else "running"
                if should_cancel:
                    self._cancel_requested = True
            if should_cancel:
                self._signal_process(process)
                threading.Thread(
                    target=self._force_stop,
                    args=(process,),
                    name="blur-face-ui-shutdown-force",
                    daemon=True,
                ).start()
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                match = _PROGRESS.search(line)
                with self._lock:
                    if line:
                        self._logs.append(line)
                    if match:
                        self._current = int(match.group(1))
                        self._total = int(match.group(2))
                        self._progress = int(match.group(3))
            return_code = process.wait()
            with self._lock:
                self._return_code = return_code
                if self._cancel_requested:
                    self._status = "cancelled"
                elif return_code == 0:
                    self._status = "completed"
                    self._progress = 100
                else:
                    self._status = "failed"
        except Exception as exc:  # noqa: BLE001 - background process boundary
            with self._lock:
                self._logs.append(f"[UI ERROR] {exc}")
                self._status = "failed"
                self._return_code = 1
        finally:
            try:
                self._cleanup_job_temp(
                    job_temp, job_temp_identity, parent_anchor, job_anchor
                )
            finally:
                job_anchor.close()
                parent_anchor.close()
            with self._lock:
                self._process = None
                if self._job_temp == job_temp:
                    self._job_temp = None
                    self._job_temp_identity = None
                    self._job_parent_anchor = None
                    self._job_anchor = None
                if self._worker is threading.current_thread():
                    self._worker = None

    def cancel(self) -> None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                raise RuntimeError("no video is currently being processed")
            self._cancel_requested = True
            self._status = "cancelling"
        self._signal_process(process)
        threading.Thread(
            target=self._force_stop,
            args=(process,),
            name="blur-face-ui-cancel",
            daemon=True,
        ).start()

    @staticmethod
    def _write_job_manifest(
        path: Path,
        contents: str,
        job_anchor: _UiDirectoryAnchor,
    ) -> None:
        encoded = contents.encode("utf-8")
        if job_anchor.dir_fd is None:
            path.write_bytes(encoded)
            return
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=job_anchor.dir_fd,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _signal_process(process: subprocess.Popen[str]) -> None:
        try:
            if _is_windows():
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
        except (OSError, ValueError):
            process.terminate()

    @staticmethod
    def _force_stop(process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            if _is_windows():
                result = subprocess.run(
                    [
                        "taskkill",
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                    ],
                    capture_output=True,
                    check=False,
                )
                if result.returncode != 0 and process.poll() is None:
                    process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()

    def shutdown(self) -> None:
        """Stop an active process tree and synchronously clean private data."""
        with self._lock:
            self._shutdown_requested = True
            process = self._process
            worker = self._worker
            job_temp = self._job_temp
            job_temp_identity = self._job_temp_identity
            parent_anchor = self._job_parent_anchor
            job_anchor = self._job_anchor
            if process is not None and process.poll() is None:
                self._cancel_requested = True
                self._status = "cancelling"
        if process is not None and process.poll() is None:
            self._signal_process(process)
            self._force_stop(process)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=12)
            if worker.is_alive():
                with self._lock:
                    process = self._process
                if process is not None and process.poll() is None:
                    self._force_stop(process)
                worker.join()
        with self._lock:
            job_temp = self._job_temp
            job_temp_identity = self._job_temp_identity
            parent_anchor = self._job_parent_anchor
            job_anchor = self._job_anchor
        if job_temp is not None:
            try:
                self._cleanup_job_temp(
                    job_temp,
                    job_temp_identity,
                    parent_anchor,
                    job_anchor,
                )
            finally:
                if job_anchor is not None:
                    job_anchor.close()
                if parent_anchor is not None:
                    parent_anchor.close()
        elif parent_anchor is not None:
            parent_anchor.close()
        with self._lock:
            self._process = None
            self._worker = None
            self._job_temp = None
            self._job_temp_identity = None
            self._job_parent_anchor = None
            self._job_anchor = None

    @staticmethod
    def _job_directory_identity(path: Path) -> tuple[int, int]:
        candidate = Path(os.path.abspath(path.expanduser()))
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise RuntimeError(f"cannot inspect UI job directory: {candidate}") from exc
        is_junction = getattr(candidate, "is_junction", None)
        file_attributes = getattr(info, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if (
            stat.S_ISLNK(info.st_mode)
            or (is_junction is not None and is_junction())
            or bool(file_attributes & reparse_flag)
            or not stat.S_ISDIR(info.st_mode)
        ):
            raise RuntimeError(f"UI job path is not an ordinary directory: {candidate}")
        return info.st_dev, info.st_ino

    @staticmethod
    def _remove_tree_contents(descriptor: int) -> None:
        for _root, directories, files, root_fd in os.fwalk(
            ".", topdown=False, follow_symlinks=False, dir_fd=descriptor
        ):
            for filename in files:
                os.unlink(filename, dir_fd=root_fd)
            for directory in directories:
                try:
                    os.rmdir(directory, dir_fd=root_fd)
                except NotADirectoryError:
                    os.unlink(directory, dir_fd=root_fd)

    @staticmethod
    def _cleanup_job_temp(
        path: Path,
        expected_identity: tuple[int, int] | None = None,
        parent_anchor: _UiDirectoryAnchor | None = None,
        job_anchor: _UiDirectoryAnchor | None = None,
    ) -> None:
        """Remove only the exact UI-owned job directory."""
        candidate = Path(os.path.abspath(path.expanduser()))
        if not candidate.name.startswith(".blur-face-job-"):
            raise RuntimeError(f"refusing to remove unowned job directory: {candidate}")
        if (
            parent_anchor is not None
            and parent_anchor.dir_fd is not None
            and job_anchor is not None
            and job_anchor.dir_fd is not None
        ):
            info = os.fstat(job_anchor.dir_fd)
            identity = (info.st_dev, info.st_ino)
            if expected_identity is not None and identity != expected_identity:
                raise RuntimeError(
                    f"UI job directory changed before cleanup: {candidate.name}"
                )
            JobManager._remove_tree_contents(job_anchor.dir_fd)
            job_anchor.close()
            try:
                os.rmdir(candidate.name, dir_fd=parent_anchor.dir_fd)
            except FileNotFoundError:
                pass
            return
        if job_anchor is not None and _is_windows():
            if not os.path.lexists(candidate):
                job_anchor.close()
                return
            identity = JobManager._job_directory_identity(candidate)
            if expected_identity is not None and identity != expected_identity:
                raise RuntimeError(
                    f"UI job directory changed before cleanup: {candidate}"
                )
            for child in candidate.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            job_anchor.close()
            candidate.rmdir()
            return
        if not os.path.lexists(candidate):
            return
        identity = JobManager._job_directory_identity(candidate)
        if expected_identity is not None and identity != expected_identity:
            raise RuntimeError(f"UI job directory changed before cleanup: {candidate}")
        shutil.rmtree(candidate)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "progress": self._progress,
                "current": self._current,
                "total": self._total,
                "media_type": self._media_type,
                "return_code": self._return_code,
                "logs": list(self._logs),
            }


def _pick_path(kind: str, language: str = "en") -> str | tuple[str, ...]:
    """Open the operating system's file chooser without importing Tk at startup."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    zh = language.lower().startswith("zh")
    try:
        if kind == "image-input":
            selected = filedialog.askopenfilenames(
                title="选择输入图片" if zh else "Choose input images",
                filetypes=[
                    (
                        "Image files",
                        "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff",
                    ),
                    ("All files", "*.*"),
                ],
            )
        elif kind == "image-output":
            selected = filedialog.askdirectory(
                title="选择图片输出目录" if zh else "Choose image output folder"
            )
        elif kind == "output":
            selected = filedialog.asksaveasfilename(
                title="选择输出视频" if zh else "Choose output video",
                defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
            )
        elif kind == "model":
            selected = filedialog.askopenfilename(
                title="选择本地模型" if zh else "Choose local model",
                filetypes=[
                    ("OpenCV YuNet model", "*.onnx"),
                    ("All files", "*.*"),
                ],
            )
        elif kind == "yolo-model":
            selected = filedialog.askopenfilename(
                title="选择本地 YOLO 模型" if zh else "Choose local YOLO model",
                filetypes=[
                    ("Ultralytics YOLO model", "*.pt"),
                    ("All files", "*.*"),
                ],
            )
        elif kind == "sam2-model":
            selected = filedialog.askdirectory(
                title=(
                    "选择本地 SAM 2.1 模型目录"
                    if zh
                    else "Choose local SAM 2.1 model directory"
                )
            )
        else:
            selected = filedialog.askopenfilename(
                title="选择输入视频" if zh else "Choose input video",
                filetypes=[
                    ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm"),
                    ("All files", "*.*"),
                ],
            )
        return selected
    finally:
        root.destroy()


def _handler(token: str, jobs: JobManager):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BlurFaceUI/1"

        def _security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy", "default-src 'self' 'unsafe-inline'"
            )
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")

        def _authorized(self) -> bool:
            candidates = [
                self.headers.get("X-Blur-Face-Token", ""),
                parse_qs(urlparse(self.path).query).get("token", [""])[0],
            ]
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get("Cookie", ""))
            except CookieError:
                pass
            if "blur_face_token" in cookie:
                candidates.append(cookie["blur_face_token"].value)
            return any(
                supplied and secrets.compare_digest(supplied, token)
                for supplied in candidates
            )

        def _json(self, status: HTTPStatus, body: dict) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self._security_headers()
            self.end_headers()
            self.wfile.write(encoded)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 65_536:
                raise ValueError("request body is too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlparse(self.path).path
            if path == "/":
                if not self._authorized():
                    self._json(
                        HTTPStatus.FORBIDDEN,
                        {
                            "error": (
                                "UI session expired; close this tab and run "
                                "start-ui again"
                            )
                        },
                    )
                    return
                content = _UI_FILE.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header(
                    "Set-Cookie",
                    (
                        f"blur_face_token={token}; Path=/; HttpOnly; "
                        "SameSite=Strict"
                    ),
                )
                self._security_headers()
                self.end_headers()
                self.wfile.write(content)
                return
            if path == "/api/status":
                if not self._authorized():
                    self._json(HTTPStatus.FORBIDDEN, {"error": "invalid UI token"})
                    return
                self._json(HTTPStatus.OK, jobs.snapshot())
                return
            if path == "/api/models":
                if not self._authorized():
                    self._json(HTTPStatus.FORBIDDEN, {"error": "invalid UI token"})
                    return
                self._json(
                    HTTPStatus.OK,
                    {
                        "models": discover_local_models(),
                        "yolo_models": (
                            yolo_models := discover_local_yolo_models()
                        ),
                        "yolo_available": optional_yolo_available(yolo_models),
                        "sam2_models": discover_local_sam2_models(),
                    },
                )
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._authorized():
                self._json(HTTPStatus.FORBIDDEN, {"error": "invalid UI token"})
                return
            path = urlparse(self.path).path
            try:
                if path == "/api/start":
                    jobs.start(self._read_json())
                    self._json(HTTPStatus.ACCEPTED, {"ok": True})
                elif path == "/api/cancel":
                    jobs.cancel()
                    self._json(HTTPStatus.ACCEPTED, {"ok": True})
                elif path == "/api/pick-input":
                    request = self._read_json()
                    language = str(request.get("language", "en"))
                    if request.get("media_type") == "images":
                        picked = _pick_path("image-input", language)
                        selected = (
                            (picked,)
                            if isinstance(picked, str) and picked
                            else tuple(picked)
                            if not isinstance(picked, str)
                            else ()
                        )
                        suggested = (
                            str(Path(selected[0]).parent / "blurred_images")
                            if selected
                            else ""
                        )
                        self._json(
                            HTTPStatus.OK,
                            {"paths": list(selected), "suggested_output": suggested},
                        )
                    else:
                        selected = str(_pick_path("input", language))
                        suggested = (
                            str(
                                Path(selected).with_name(
                                    f"{Path(selected).stem}_blurred.mp4"
                                )
                            )
                            if selected
                            else ""
                        )
                        self._json(
                            HTTPStatus.OK,
                            {"path": selected, "suggested_output": suggested},
                        )
                elif path == "/api/pick-output":
                    request = self._read_json()
                    language = str(request.get("language", "en"))
                    kind = (
                        "image-output"
                        if request.get("media_type") == "images"
                        else "output"
                    )
                    self._json(
                        HTTPStatus.OK,
                        {"path": _pick_path(kind, language)},
                    )
                elif path == "/api/pick-model":
                    request = self._read_json()
                    language = str(request.get("language", "en"))
                    kind = (
                        "yolo-model"
                        if request.get("detector") == "yolo"
                        else "model"
                    )
                    self._json(
                        HTTPStatus.OK,
                        {"path": _pick_path(kind, language)},
                    )
                elif path == "/api/pick-sam2-model":
                    language = str(self._read_json().get("language", "en"))
                    self._json(
                        HTTPStatus.OK,
                        {"path": _pick_path("sam2-model", language)},
                    )
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except RuntimeError as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - HTTP boundary
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, _format: str, *_args) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local blur-face web UI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")
    token = secrets.token_urlsafe(24)
    jobs = JobManager()
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        _handler(token, jobs),
    )
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/?token={token}"
    print("Blur Face UI is local-only.")
    print(f"Open: {url}")
    print("Press Ctrl+C to stop the UI.")
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nUI stopped.")
    finally:
        server.server_close()
        jobs.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
