import unittest
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from blurface.detector import DetectorError, FaceDetector, _YoloDetector


class DetectorTests(unittest.TestCase):
    def test_yunet_xywh_output_is_converted_to_pipeline_xyxy(self):
        backend = Mock()
        backend.detect.return_value = (
            1,
            np.array(
                [[10, 20, 30, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8]],
                dtype=np.float32,
            ),
        )
        factory = SimpleNamespace(create=Mock(return_value=backend))
        with (
            patch("blurface.detector.resolve_model", return_value="/model.onnx"),
            patch("blurface.detector.cv2.FaceDetectorYN", factory),
        ):
            detector = FaceDetector("model.onnx", allow_download=False)
            result = detector.detect(
                np.zeros((100, 120, 3), dtype=np.uint8), conf=0.4
            )

        factory.create.assert_called_once()
        backend.setInputSize.assert_called_once_with((120, 100))
        backend.setScoreThreshold.assert_called_once_with(0.4)
        np.testing.assert_allclose(
            result, np.array([[10, 20, 40, 60, 0.8]])
        )

    def test_empty_yunet_result_uses_stable_empty_shape(self):
        backend = Mock()
        backend.detect.return_value = (1, None)
        with (
            patch("blurface.detector.resolve_model", return_value="/model.onnx"),
            patch(
                "blurface.detector.cv2.FaceDetectorYN",
                SimpleNamespace(create=Mock(return_value=backend)),
            ),
        ):
            result = FaceDetector("model.onnx").detect(
                np.zeros((10, 12, 3), dtype=np.uint8)
            )
        self.assertEqual(result.shape, (0, 5))

    def test_yolo_is_lazy_explicit_and_uses_accuracy_inference_settings(self):
        xyxy = Mock()
        xyxy.cpu.return_value.numpy.return_value = np.array(
            [[10, 20, 40, 60]], dtype=np.float32
        )
        confidence = Mock()
        confidence.cpu.return_value.numpy.return_value = np.array(
            [0.8], dtype=np.float32
        )
        class Boxes:
            def __init__(self):
                self.xyxy = xyxy
                self.conf = confidence

            def __len__(self):
                return 1

        boxes = Boxes()
        yolo_model = Mock()
        yolo_model.task = "detect"
        yolo_model.names = {0: "face"}
        yolo_model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        yolo_factory = Mock(return_value=yolo_model)
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _index: "Test GPU",
            ),
            version=SimpleNamespace(cuda="12.6"),
            __version__="2.test",
        )
        with (
            patch("blurface.detector.resolve_model", return_value="/model.pt"),
            patch.dict(
                sys.modules,
                {
                    "ultralytics": SimpleNamespace(YOLO=yolo_factory),
                    "torch": torch,
                },
            ),
        ):
            detector = FaceDetector(
                "model.pt",
                backend="yolo",
                device="auto",
                allow_download=True,
            )
            result = detector.detect(
                np.zeros((100, 120, 3), dtype=np.uint8), conf=0.4
            )

        yolo_factory.assert_called_once_with("/model.pt")
        self.assertEqual(detector.device, "cuda:0")
        yolo_model.predict.assert_called_once()
        predict = yolo_model.predict.call_args.kwargs
        self.assertEqual(predict["imgsz"], 1280)
        self.assertEqual(predict["max_det"], 1000)
        self.assertEqual(predict["conf"], 0.4)
        self.assertEqual(predict["classes"], [0])
        np.testing.assert_allclose(
            result, np.array([[10, 20, 40, 60, 0.8]])
        )

    def test_yolo_never_downloads_a_missing_weight_during_processing(self):
        with patch(
            "blurface.detector.resolve_model",
            side_effect=FileNotFoundError("missing"),
        ) as resolve:
            with self.assertRaisesRegex(DetectorError, "install-yolo"):
                FaceDetector(
                    "missing.pt",
                    backend="yolo",
                    allow_download=True,
                )
        resolve.assert_called_once_with("missing.pt", allow_download=False)

    def test_yolo_logs_the_selected_cuda_device_name(self):
        get_device_name = Mock(return_value="Second GPU")
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=get_device_name,
            ),
            version=SimpleNamespace(cuda="12.6"),
            __version__="2.test",
        )
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(_YoloDetector._resolve_device("cuda:1"), "cuda:1")
        get_device_name.assert_called_once_with(1)

    def test_yolo_rejects_generic_detector_without_face_class(self):
        generic_model = Mock(task="detect", names={0: "person", 1: "car"})
        with (
            patch("blurface.detector.resolve_model", return_value="/coco.pt"),
            patch.dict(
                sys.modules,
                {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=generic_model))},
            ),
        ):
            with self.assertRaisesRegex(DetectorError, "no explicit face class"):
                FaceDetector("coco.pt", backend="yolo", device="cpu")

    def test_yolo_rejects_non_detection_task(self):
        segment_model = Mock(task="segment", names={0: "face"})
        with (
            patch("blurface.detector.resolve_model", return_value="/segment.pt"),
            patch.dict(
                sys.modules,
                {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=segment_model))},
            ),
        ):
            with self.assertRaisesRegex(DetectorError, "task must be detect"):
                FaceDetector("segment.pt", backend="yolo", device="cpu")

    def test_unknown_detector_backend_is_rejected(self):
        with self.assertRaisesRegex(DetectorError, "unsupported"):
            FaceDetector("model.bin", backend="remote")


if __name__ == "__main__":
    unittest.main()
