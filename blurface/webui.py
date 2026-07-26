"""Local-only browser UI for blur-face."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import AppConfig
from .model_store import model_directories

_PROGRESS = re.compile(r"\b(\d+)/(\d+)\s+\((\d+)%\)")
_UI_FILE = Path(__file__).with_name("ui") / "index.html"


def _number(payload: dict, key: str, cast):
    try:
        return cast(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}") from exc


def discover_local_models() -> list[str]:
    """Return local PyTorch models from every configured model directory."""
    models: set[Path] = set()
    for directory in model_directories():
        if directory.is_dir():
            models.update(
                path.resolve()
                for path in directory.glob("*.pt")
                if path.is_file()
            )
    return [str(path) for path in sorted(models, key=lambda path: path.name.lower())]


def build_command(payload: dict) -> list[str]:
    """Validate a UI request and translate it into the authoritative CLI."""
    input_path = Path(str(payload.get("input", ""))).expanduser()
    output_path = Path(str(payload.get("output", ""))).expanduser()
    if not input_path.is_file():
        raise ValueError(f"input video does not exist: {input_path}")
    if not str(payload.get("output", "")).strip():
        raise ValueError("output path is required")

    config = AppConfig(
        input=input_path,
        output=output_path,
        overwrite=bool(payload.get("overwrite", True)),
        model=str(payload.get("model", "yolov11m-face.pt")),
        threshold=_number(payload, "threshold", float),
        mask_scale=_number(payload, "mask_scale", float),
        mask_shape=str(payload.get("mask_shape", "")),
        blur_strategy=str(payload.get("blur_strategy", "")),
        blur_kernel=_number(payload, "blur_kernel", int),
        blur_kernel_min=_number(payload, "blur_kernel_min", int),
        min_face_size=_number(payload, "min_face_size", int),
        preset=str(payload.get("preset", "quality")),
        offline=bool(payload.get("offline", False)),
        use_nvenc=not bool(payload.get("no_nvenc", False)),
    ).validate()
    command = [
        sys.executable,
        "-u",
        "-m",
        "blurface",
        str(config.input),
        "-o",
        str(config.output),
        "--model",
        config.model,
        "--thresh",
        str(config.threshold),
        "--mask-scale",
        str(config.mask_scale),
        "--mask-shape",
        config.mask_shape,
        "--blur-strategy",
        config.blur_strategy,
        "--blur-kernel",
        str(config.blur_kernel),
        "--blur-kernel-min",
        str(config.blur_kernel_min),
        "--min-face-size",
        str(config.min_face_size),
        "--preset",
        config.preset,
    ]
    if config.overwrite:
        command.append("--overwrite")
    if config.offline:
        command.append("--offline")
    if not config.use_nvenc:
        command.append("--no-nvenc")
    return command


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
        self._cancel_requested = False
        self._logs: deque[str] = deque(maxlen=500)

    def start(self, payload: dict) -> None:
        command = build_command(payload)
        with self._lock:
            if self._status in {"starting", "running", "cancelling"}:
                raise RuntimeError("a video is already being processed")
            self._status = "starting"
            self._progress = self._current = self._total = 0
            self._return_code = None
            self._cancel_requested = False
            self._logs.clear()
        threading.Thread(
            target=self._run,
            args=(command,),
            name="blur-face-ui-job",
            daemon=True,
        ).start()

    def _run(self, command: list[str]) -> None:
        creationflags = 0
        kwargs: dict = {}
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
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
                **kwargs,
            )
            with self._lock:
                self._process = process
                self._status = "running"
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
            with self._lock:
                self._process = None

    def cancel(self) -> None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                raise RuntimeError("no video is currently being processed")
            self._cancel_requested = True
            self._status = "cancelling"
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
        except (OSError, ValueError):
            process.terminate()
        threading.Thread(
            target=self._force_stop,
            args=(process,),
            name="blur-face-ui-cancel",
            daemon=True,
        ).start()

    @staticmethod
    def _force_stop(process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "progress": self._progress,
                "current": self._current,
                "total": self._total,
                "return_code": self._return_code,
                "logs": list(self._logs),
            }


def _pick_path(kind: str, language: str = "en") -> str:
    """Open the operating system's file chooser without importing Tk at startup."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    zh = language.lower().startswith("zh")
    try:
        if kind == "output":
            selected = filedialog.asksaveasfilename(
                title="选择输出视频" if zh else "Choose output video",
                defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
            )
        elif kind == "model":
            selected = filedialog.askopenfilename(
                title="选择本地模型" if zh else "Choose local model",
                filetypes=[
                    ("PyTorch model", "*.pt"),
                    ("All files", "*.*"),
                ],
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
            supplied = self.headers.get("X-Blur-Face-Token", "")
            if not supplied:
                supplied = parse_qs(urlparse(self.path).query).get("token", [""])[0]
            return secrets.compare_digest(supplied, token)

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
                content = _UI_FILE.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
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
                self._json(HTTPStatus.OK, {"models": discover_local_models()})
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
                    language = str(self._read_json().get("language", "en"))
                    selected = _pick_path("input", language)
                    suggested = (
                        str(Path(selected).with_name(f"{Path(selected).stem}_blurred.mp4"))
                        if selected
                        else ""
                    )
                    self._json(
                        HTTPStatus.OK,
                        {"path": selected, "suggested_output": suggested},
                    )
                elif path == "/api/pick-output":
                    language = str(self._read_json().get("language", "en"))
                    self._json(
                        HTTPStatus.OK,
                        {"path": _pick_path("output", language)},
                    )
                elif path == "/api/pick-model":
                    language = str(self._read_json().get("language", "en"))
                    self._json(
                        HTTPStatus.OK,
                        {"path": _pick_path("model", language)},
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
