import tempfile
import threading
import unittest
import os
import signal
import subprocess
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import Mock, patch

from blurface.webui import (
    _UI_FILE,
    JobManager,
    _handler,
    build_command,
    discover_local_models,
    discover_local_yolo_models,
    discover_local_sam2_models,
    optional_yolo_available,
)


class WebUiTests(unittest.TestCase):
    def test_ui_owned_job_directory_is_removed_after_forced_job(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / ".blur-face-job-test"
            job.mkdir()
            (job / "private-mask.png").write_bytes(b"mask")
            (job / ".output.partial.mp4").write_bytes(b"partial")
            JobManager._cleanup_job_temp(job)
            self.assertFalse(job.exists())
            with self.assertRaises(RuntimeError):
                JobManager._cleanup_job_temp(Path(directory))

    def test_ui_cleanup_rejects_symlink_without_touching_target(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symbolic links are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / ".blur-face-job-victim"
            victim.mkdir()
            private = victim / "private-mask.png"
            private.write_bytes(b"private")
            link = root / ".blur-face-job-link"
            try:
                link.symlink_to(victim, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"cannot create symbolic link: {exc}")
            with self.assertRaisesRegex(RuntimeError, "ordinary directory"):
                JobManager._cleanup_job_temp(link)
            self.assertEqual(private.read_bytes(), b"private")
            self.assertTrue(link.is_symlink())

    def test_ui_cleanup_rejects_replaced_owned_directory_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / ".blur-face-job-owned"
            job.mkdir()
            identity = JobManager._job_directory_identity(job)
            original = root / ".blur-face-job-original"
            job.rename(original)
            job.mkdir()
            with self.assertRaisesRegex(RuntimeError, "changed before cleanup"):
                JobManager._cleanup_job_temp(job, identity)
            self.assertTrue(job.is_dir())

    @unittest.skipIf(os.name == "nt", "POSIX dirfd creation race regression")
    def test_ui_job_creation_does_not_resolve_a_replaced_symlink(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symbolic links are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "image.png"
            source.touch()
            output = root / "output"
            output.mkdir()
            victim = root / ".blur-face-job-victim"
            victim.mkdir()
            private = victim / "unowned.txt"
            private.write_bytes(b"private")

            real_mkdir = os.mkdir

            def replaced_job(path, mode=0o777, *, dir_fd=None):
                if dir_fd is None:
                    return real_mkdir(path, mode)
                selected = output / str(path)
                original = output / ".blur-face-job-original"
                real_mkdir(selected)
                selected.rename(original)
                try:
                    selected.symlink_to(victim, target_is_directory=True)
                except OSError as exc:
                    self.skipTest(f"cannot create symbolic link: {exc}")

            payload = {
                "media_type": "images",
                "inputs": [str(source)],
                "output": str(output),
                "threshold": 0.3,
                "mask_scale": 1.5,
                "mask_shape": "rounded-rect",
                "blur_strategy": "adaptive",
                "blur_kernel": 251,
                "blur_kernel_min": 101,
                "min_face_size": 30,
                "preset": "quality",
                "mask_engine": "geometric",
                "sam_mask_expansion": 0.12,
            }
            with patch("blurface.webui.os.mkdir", side_effect=replaced_job):
                with self.assertRaisesRegex(RuntimeError, "ordinary directory"):
                    JobManager().start(payload)
            self.assertEqual(private.read_bytes(), b"private")

    def test_force_stop_kills_the_whole_process_group(self):
        process = Mock(pid=12345)
        process.wait.side_effect = [
            subprocess.TimeoutExpired("blur-face", 8),
            0,
        ]
        if os.name == "nt":
            with patch("blurface.webui.subprocess.run") as run:
                run.return_value.returncode = 0
                JobManager._force_stop(process)
            self.assertIn("taskkill", run.call_args.args[0])
        else:
            with patch("blurface.webui.os.killpg") as kill_group:
                JobManager._force_stop(process)
            kill_group.assert_called_once_with(12345, signal.SIGKILL)

    def test_ui_child_output_encoding_matches_utf8_reader(self):
        process = Mock(stdout=[])
        process.wait.return_value = 0
        jobs = JobManager()
        parent_anchor = Mock()
        job_anchor = Mock()
        with (
            patch("blurface.webui.subprocess.Popen", return_value=process) as popen,
            patch.object(jobs, "_cleanup_job_temp"),
        ):
            jobs._run(
                ["blur-face"],
                Path(".blur-face-job-test"),
                (1, 2),
                parent_anchor,
                job_anchor,
            )
        self.assertEqual(
            popen.call_args.kwargs["env"]["PYTHONIOENCODING"],
            "utf-8:backslashreplace",
        )
        job_anchor.close.assert_called_once_with()
        parent_anchor.close.assert_called_once_with()

    def test_starting_state_recovers_when_parent_anchor_setup_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.mp4"
            payload = {"output": str(output)}

            identity_jobs = JobManager()
            with (
                patch("blurface.webui.build_command", return_value=["blur-face"]),
                patch.object(
                    identity_jobs,
                    "_job_directory_identity",
                    side_effect=RuntimeError("identity failure"),
                ) as identity,
            ):
                for _attempt in range(2):
                    with self.assertRaisesRegex(RuntimeError, "identity failure"):
                        identity_jobs.start(payload)
                    self.assertEqual(identity_jobs.snapshot()["status"], "failed")
            self.assertEqual(identity.call_count, 2)

            anchor_jobs = JobManager()
            with (
                patch("blurface.webui.build_command", return_value=["blur-face"]),
                patch(
                    "blurface.webui._UiDirectoryAnchor",
                    side_effect=RuntimeError("anchor failure"),
                ) as anchor,
            ):
                for _attempt in range(2):
                    with self.assertRaisesRegex(RuntimeError, "anchor failure"):
                        anchor_jobs.start(payload)
                    self.assertEqual(anchor_jobs.snapshot()["status"], "failed")
            self.assertEqual(anchor.call_count, 2)

    def test_shutdown_stops_active_job_and_cleans_owned_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            job_temp = Path(directory) / ".blur-face-job-shutdown"
            job_temp.mkdir()
            (job_temp / "private-mask.png").write_bytes(b"mask")
            process = Mock(pid=12345)
            process.poll.return_value = None
            worker = Mock()
            worker.is_alive.return_value = False
            jobs = JobManager()
            jobs._process = process
            jobs._worker = worker
            jobs._job_temp = job_temp
            jobs._status = "running"

            with (
                patch.object(jobs, "_signal_process") as signal_process,
                patch.object(jobs, "_force_stop") as force_stop,
            ):
                jobs.shutdown()

            signal_process.assert_called_once_with(process)
            force_stop.assert_called_once_with(process)
            worker.join.assert_called_once_with(timeout=12)
            self.assertFalse(job_temp.exists())
            self.assertEqual(jobs.snapshot()["status"], "cancelling")

    def test_shutdown_reloads_job_ownership_after_worker_join(self):
        with tempfile.TemporaryDirectory() as directory:
            job_temp = Path(directory) / ".blur-face-job-finished"
            job_temp.mkdir()
            jobs = JobManager()
            worker = Mock()
            worker.is_alive.return_value = False
            jobs._worker = worker
            jobs._job_temp = job_temp
            jobs._job_temp_identity = (1, 2)
            jobs._job_parent_anchor = Mock()
            jobs._job_anchor = Mock()

            def worker_finished(*_args, **_kwargs):
                job_temp.rmdir()
                with jobs._lock:
                    jobs._job_temp = None
                    jobs._job_temp_identity = None
                    jobs._job_parent_anchor = None
                    jobs._job_anchor = None

            worker.join.side_effect = worker_finished
            with patch.object(jobs, "_cleanup_job_temp") as cleanup:
                jobs.shutdown()
            cleanup.assert_not_called()
            self.assertFalse(job_temp.exists())

    def test_windows_cleanup_is_idempotent_after_worker_removed_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / ".blur-face-job-removed"
            anchor = Mock(dir_fd=None)
            with patch("blurface.webui._is_windows", return_value=True):
                JobManager._cleanup_job_temp(missing, (1, 2), job_anchor=anchor)
            anchor.close.assert_called_once_with()

    def test_image_job_manifest_is_owned_inside_output_and_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.touch()
            output = root / "out"
            payload = {
                "media_type": "images",
                "inputs": [str(image)],
                "output": str(output),
                "threshold": 0.3,
                "mask_scale": 1.5,
                "mask_shape": "rounded-rect",
                "blur_strategy": "adaptive",
                "blur_kernel": 251,
                "blur_kernel_min": 101,
                "min_face_size": 30,
                "preset": "quality",
                "mask_engine": "geometric",
                "sam_mask_expansion": 0.12,
            }
            worker = Mock()
            worker.is_alive.return_value = False
            jobs = JobManager()
            with patch("blurface.webui.threading.Thread", return_value=worker):
                jobs.start(payload)
            self.assertEqual(jobs.snapshot()["media_type"], "images")
            job_temp = jobs._job_temp
            self.assertIsNotNone(job_temp)
            self.assertEqual(job_temp.parent, output.resolve())
            self.assertTrue((job_temp / "inputs.json").is_file())
            jobs.shutdown()
            self.assertFalse(job_temp.exists())

    @unittest.skipIf(os.name == "nt", "Windows parent handle prevents rename")
    def test_image_manifest_is_cleaned_after_output_parent_is_moved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.touch()
            output = root / "out"
            payload = {
                "media_type": "images",
                "inputs": [str(image)],
                "output": str(output),
                "threshold": 0.3,
                "mask_scale": 1.5,
                "mask_shape": "rounded-rect",
                "blur_strategy": "adaptive",
                "blur_kernel": 251,
                "blur_kernel_min": 101,
                "min_face_size": 30,
                "preset": "quality",
                "mask_engine": "geometric",
                "sam_mask_expansion": 0.12,
            }
            worker = Mock()
            worker.is_alive.return_value = False
            jobs = JobManager()
            with patch("blurface.webui.threading.Thread", return_value=worker):
                jobs.start(payload)
            job_name = jobs._job_temp.name
            self.assertTrue((output / job_name / "inputs.json").is_file())
            moved = root / "moved"
            output.rename(moved)
            jobs.shutdown()
            self.assertFalse((moved / job_name).exists())

    def test_ui_main_shutdown_calls_job_manager_shutdown(self):
        source = Path(__file__).resolve().parents[1] / "blurface" / "webui.py"
        contents = source.read_text(encoding="utf-8")
        self.assertIn("jobs.shutdown()", contents)

    def test_build_command_uses_validated_local_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp4"
            source.touch()
            output = root / "output.mp4"
            command = build_command(
                {
                    "input": str(source),
                    "output": str(output),
                    "model": str(root / "face-model.onnx"),
                    "detector": "yunet",
                    "overwrite": True,
                    "threshold": 0.3,
                    "mask_scale": 1.5,
                    "mask_shape": "rectangle",
                    "blur_strategy": "fixed",
                    "blur_kernel": 301,
                    "blur_kernel_min": 101,
                    "min_face_size": 30,
                    "preset": "quality",
                    "mask_engine": "sam2.1",
                    "sam_mask_expansion": 0.12,
                    "segmentation_combine": "mask-only",
                    "temporal_stabilization": True,
                    "mask_preview": True,
                    "backfill_frames": 12,
                    "release_hold_frames": 4,
                    "scene_cut_sensitivity": 0.6,
                    "temporal_storage_limit_mb": 2048,
                    "device": "cpu",
                    "offline": True,
                    "no_nvenc": True,
                }
            )
            self.assertIn("--mask-shape", command)
            self.assertIn("--model", command)
            self.assertEqual(
                command[command.index("--detector") + 1], "yunet"
            )
            self.assertIn(str(root / "face-model.onnx"), command)
            self.assertIn("rectangle", command)
            self.assertIn("--blur-strategy", command)
            self.assertIn("fixed", command)
            self.assertIn("--mask-engine", command)
            self.assertIn("sam2.1", command)
            self.assertIn("--sam-mask-expansion", command)
            self.assertIn("0.12", command)
            self.assertEqual(
                command[command.index("--segmentation-combine") + 1],
                "mask-only",
            )
            self.assertIn("--sam2-model", command)
            self.assertIn("--sam2-refresh-interval", command)
            self.assertIn("--temporal-stabilization", command)
            self.assertIn("--mask-preview", command)
            self.assertEqual(
                command[command.index("--backfill-frames") + 1], "12"
            )
            self.assertEqual(
                command[command.index("--release-hold-frames") + 1], "4"
            )
            self.assertEqual(
                command[command.index("--scene-cut-sensitivity") + 1], "0.6"
            )
            self.assertEqual(
                command[command.index("--temporal-storage-limit-mb") + 1],
                "2048",
            )
            self.assertEqual(command[command.index("--device") + 1], "cpu")
            self.assertIn("--offline", command)
            self.assertIn("--no-nvenc", command)
            self.assertNotIn("shell=True", command)

    def test_build_command_rejects_unknown_mask_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            source.touch()
            with self.assertRaises(ValueError):
                build_command(
                    {
                        "input": str(source),
                        "output": str(Path(directory) / "output.mp4"),
                        "threshold": 0.3,
                        "mask_scale": 1.5,
                        "mask_shape": "cutout",
                        "blur_strategy": "adaptive",
                        "blur_kernel": 251,
                        "blur_kernel_min": 101,
                        "min_face_size": 30,
                        "preset": "quality",
                        "mask_engine": "geometric",
                        "sam_mask_expansion": 0.12,
                    }
                )

    def test_build_command_supports_sam2_model_id(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            source.touch()
            command = build_command(
                {
                    "input": str(source),
                    "output": str(Path(directory) / "output.mp4"),
                    "threshold": 0.3,
                    "mask_scale": 1.5,
                    "mask_shape": "rounded-rect",
                    "blur_strategy": "adaptive",
                    "blur_kernel": 251,
                    "blur_kernel_min": 101,
                    "min_face_size": 30,
                    "preset": "quality",
                    "mask_engine": "sam2.1",
                    "sam2_model": "facebook/sam2.1-hiera-large",
                    "sam_mask_expansion": 0.12,
                }
            )
            self.assertEqual(
                command[command.index("--sam2-model") + 1],
                "facebook/sam2.1-hiera-large",
            )
            self.assertIn("sam2.1", command)

    def test_build_command_rejects_unknown_mask_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            source.touch()
            with self.assertRaisesRegex(ValueError, "mask-engine"):
                build_command(
                    {
                        "input": str(source),
                        "output": str(Path(directory) / "output.mp4"),
                        "threshold": 0.3,
                        "mask_scale": 1.5,
                        "mask_shape": "rounded-rect",
                        "blur_strategy": "adaptive",
                        "blur_kernel": 251,
                        "blur_kernel_min": 101,
                        "min_face_size": 30,
                        "preset": "quality",
                        "mask_engine": "remote",
                        "sam_mask_expansion": 0.12,
                    }
                )

    def test_build_command_uses_manifest_for_multi_image_ui_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "一.png"
            second = root / "two.jpg"
            first.touch()
            second.touch()
            manifest = root / "inputs.json"
            manifest.write_text("[]", encoding="utf-8")
            command = build_command(
                {
                    "media_type": "images",
                    "inputs": [str(first), str(second)],
                    "output": str(root / "out"),
                    "threshold": 0.3,
                    "mask_scale": 1.5,
                    "mask_shape": "rounded-rect",
                    "blur_strategy": "adaptive",
                    "blur_kernel": 251,
                    "blur_kernel_min": 101,
                    "min_face_size": 30,
                    "preset": "quality",
                    "mask_engine": "geometric",
                    "sam_mask_expansion": 0.12,
                },
                input_manifest=manifest,
            )
            self.assertEqual(
                command[command.index("--input-list") + 1], str(manifest)
            )
            self.assertNotIn(str(first), command)
            self.assertNotIn(str(second), command)

    def test_ui_has_no_remote_assets_or_upload_control(self):
        html = _UI_FILE.read_text(encoding="utf-8")
        self.assertNotIn("https://", html)
        self.assertNotIn('type="file"', html)
        self.assertIn("127.0.0.1 ONLY", html)
        self.assertIn("navigator.languages", html)
        self.assertIn('data-lang-mode="auto"', html)
        self.assertIn('id="model"', html)
        self.assertNotIn('id="detector"', html)
        self.assertIn('["geometric-yolo", "geometricYoloEngine"]', html)
        self.assertIn('["sam2.1-yolo", "sam2YoloEngine"]', html)
        self.assertIn("data.yolo_available", html)
        self.assertIn("selectedDetector()", html)
        self.assertIn("AGPL / Enterprise", html)
        self.assertIn("AGPL / 商业许可", html)
        self.assertIn("GPL-3.0", html)
        self.assertIn("WIDER FACE", html)
        self.assertIn('data-help="thresholdHelp"', html)
        self.assertIn('id="mask-engine"', html)
        self.assertNotIn("face-parsing", html)
        self.assertIn('value="sam2.1"', html)
        self.assertIn('<details class="advanced-panel">', html)
        self.assertNotIn("updateParsingOptions", html)
        self.assertNotIn('$("mask-engine").value = "geometric"', html)
        self.assertNotIn('id="parsing-model"', html)
        self.assertIn('id="sam-mask-expansion"', html)
        self.assertIn('id="segmentation-combine"', html)
        self.assertIn('id="sam2-refresh-interval"', html)
        self.assertIn('id="temporal-stabilization"', html)
        self.assertIn('id="mask-preview"', html)
        self.assertIn('id="media-type"', html)
        self.assertIn('value="images"', html)
        self.assertIn("Images (single or batch)", html)
        self.assertIn("图片（单张或批量）", html)
        self.assertIn("selectedImageInputs", html)
        self.assertIn('media_type: $("media-type").value', html)
        self.assertIn('data.media_type === "images"', html)
        self.assertIn('$("media-type").disabled = running', html)
        self.assertIn("Mask diagnostic video", html)
        self.assertIn("遮挡区域测试视频", html)
        self.assertIn('id="backfill-frames"', html)
        self.assertIn('id="release-hold-frames"', html)
        self.assertIn(
            'id="release-hold-frames" type="number" min="0" max="12" '
            'step="1" value="5"',
            html,
        )
        self.assertIn('id="scene-cut-sensitivity"', html)
        self.assertIn('value="intersection"', html)
        self.assertIn('value="mask-only"', html)
        self.assertIn("SAM contour only", html)
        self.assertIn("仅 SAM 轮廓", html)
        self.assertIn('imageSegmentationCombineHelp: "Union adds the current detected-face geometry', html)
        self.assertIn('imageSegmentationCombineHelp: "并集把当前检测到的人脸几何范围', html)
        self.assertIn('segmentationCombineHelp: "imageSegmentationCombineHelp"', html)
        self.assertIn("keys.segmentationCombineHelp", html)
        self.assertIn('data-help="maskEngineHelp"', html)
        self.assertIn("/api/pick-sam2-model", html)
        self.assertIn("OpenCV YuNet", html)
        self.assertIn("临时存储", html)
        self.assertIn('id="device"', html)
        self.assertIn('id="temporal-storage-limit"', html)
        self.assertIn('idle: "Waiting"', html)
        self.assertIn("history.replaceState", html)

    def test_valid_page_token_creates_refresh_safe_session_cookie(self):
        token = "test-secret"
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), _handler(token, JobManager())
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with self.assertRaises(HTTPError) as denied:
                urlopen(f"{base}/?token=expired", timeout=2)
            self.assertEqual(denied.exception.code, 403)

            with urlopen(f"{base}/?token={token}", timeout=2) as page:
                self.assertEqual(page.status, 200)
                cookie = page.headers["Set-Cookie"].split(";", 1)[0]
            request = Request(
                f"{base}/api/status",
                headers={"Cookie": cookie},
            )
            with urlopen(request, timeout=2) as status:
                self.assertEqual(status.status, 200)
                self.assertIn(b'"status": "idle"', status.read())
            picker = Request(
                f"{base}/api/pick-input",
                data=b'{"language":"en"}',
                headers={
                    "Cookie": cookie,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with (
                patch("blurface.webui._pick_path", return_value="C:/video.mp4"),
                urlopen(picker, timeout=2) as picked,
            ):
                self.assertEqual(picked.status, 200)
                self.assertIn(b"C:/video.mp4", picked.read())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_discovers_models_from_local_model_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.onnx"
            second = root / "second.onnx"
            first.touch()
            second.touch()
            yolo = root / "yolov11m-face.pt"
            yolo.touch()
            (root / "unrelated.pt").touch()
            (root / "yolo11n.pt").touch()
            (root / "face-parsing-yolo.pt").touch()
            (root / "ignore.txt").touch()
            with (
                patch("blurface.webui.model_directories", return_value=(root,)),
                patch(
                    "blurface.webui.file_digest",
                    return_value=(
                        "6ccbe920c1fac95ed84de570519e89fbe24d326d466a7aae"
                        "297960b3ecc6c661"
                    ),
                ),
            ):
                self.assertEqual(
                    discover_local_models(),
                    [str(first.resolve()), str(second.resolve())],
                )
                self.assertEqual(
                    discover_local_yolo_models(), [str(yolo.resolve())]
                )

            with patch(
                "blurface.webui.model_directories", return_value=(root,)
            ):
                self.assertEqual(discover_local_yolo_models(), [])

    def test_yolo_ui_modes_require_supported_weight_and_dependency(self):
        with patch("blurface.webui.importlib.util.find_spec") as find_spec:
            find_spec.return_value = object()
            self.assertTrue(optional_yolo_available(["/models/yolov11m-face.pt"]))
            self.assertFalse(optional_yolo_available([]))
            find_spec.return_value = None
            self.assertFalse(
                optional_yolo_available(["/models/yolov11m-face.pt"])
            )

    def test_build_command_accepts_explicit_yolo_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp4"
            source.touch()
            command = build_command(
                {
                    "input": str(source),
                    "output": str(root / "output.mp4"),
                    "detector": "yolo",
                    "model": str(root / "face.pt"),
                    "threshold": 0.3,
                    "mask_scale": 1.5,
                    "mask_shape": "rounded-rect",
                    "blur_strategy": "adaptive",
                    "blur_kernel": 251,
                    "blur_kernel_min": 101,
                    "min_face_size": 30,
                    "preset": "quality",
                    "mask_engine": "geometric",
                    "sam_mask_expansion": 0.12,
                }
            )
            self.assertEqual(
                command[command.index("--detector") + 1], "yolo"
            )
            self.assertEqual(
                command[command.index("--model") + 1], str(root / "face.pt")
            )

    def test_discovers_local_sam2_model_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "sam2.1-local"
            model.mkdir()
            (model / "config.json").write_text(
                '{"model_type":"sam2"}', encoding="utf-8"
            )
            with patch("blurface.webui.model_directories", return_value=(root,)):
                self.assertEqual(
                    discover_local_sam2_models(), [str(model.resolve())]
                )


if __name__ == "__main__":
    unittest.main()
