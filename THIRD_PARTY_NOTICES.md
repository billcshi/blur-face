# Third-party notices

Blur Face project-owned source code is licensed under the repository's MIT
license. The following components are not relicensed by this project.

## Default runtime

| Component | Role | License |
|---|---|---|
| OpenCV 4.5+ | Detection, tracking support, image/video operations | Apache-2.0 |
| opencv-python packaging scripts | Python wheel packaging | MIT |
| OpenCV YuNet `face_detection_yunet_2023mar.onnx` | Default face detector model | MIT |
| NumPy | Array operations | BSD-3-Clause |

The official `opencv-python-headless` wheels also contain third-party binary
components. Its upstream notice identifies the included FFmpeg libraries as
LGPL-2.1. Consult the `LICENSE-3RD-PARTY.txt` installed with the exact wheel for
the complete platform-specific list.

Sources:

- <https://opencv.org/license/>
- <https://github.com/opencv/opencv-python#licensing>
- <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- <https://github.com/numpy/numpy/blob/main/LICENSE.txt>

## Optional SAM runtime

| Component | Role | License |
|---|---|---|
| Meta SAM 2 code and official checkpoints | Video object segmentation | Apache-2.0 |
| Hugging Face Transformers | SAM 2 model and processor API | Apache-2.0 |
| safetensors | Model tensor loading | Apache-2.0 |
| PyTorch | Tensor runtime | BSD-style |
| Pillow | Image utilities | HPND |

The CUDA PyTorch installation may download NVIDIA runtime packages under their
own terms. Blur Face does not copy those packages into this repository.

Sources:

- <https://github.com/facebookresearch/sam2>
- <https://github.com/huggingface/transformers/blob/main/LICENSE>
- <https://github.com/huggingface/safetensors/blob/main/LICENSE>
- <https://github.com/pytorch/pytorch/blob/main/LICENSE>
- <https://github.com/python-pillow/Pillow/blob/main/LICENSE>

## Optional YOLO detector runtime

This section applies only when the user explicitly chooses the optional YOLO
installation. It is not part of the default runtime.

| Component | Role | License |
|---|---|---|
| Ultralytics Python package | YOLO inference API | AGPL-3.0 or separate Ultralytics Enterprise license |
| PyTorch / torchvision | Tensor runtime | BSD-style |
| `akanametov/yolo-face` 1.0.0 `yolov11m-face.pt` | Optional face detector weight | GPL-3.0 or an applicable Enterprise license, as stated by the upstream repository |

The setup and standalone installers display this boundary before installation.
Selecting YOLO does not relicense the project-owned MIT source, but using,
modifying, distributing, or providing a network service with the optional
Ultralytics component can create obligations under its license. Obtain
qualified legal advice for a particular commercial distribution or service.
The pinned face weight is published by the `akanametov/yolo-face` 1.0.0
release. That release's training recipe identifies WIDER FACE as its dataset;
the dataset's own terms remain applicable and are not replaced by this
project's MIT license. User-supplied weights likewise retain their own
training-data and model terms.

Sources:

- <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>
- <https://docs.ultralytics.com/#yolo-licenses-how-is-ultralytics-yolo-licensed>
- <https://github.com/akanametov/yolo-face/blob/1.0.0/README.md#license>
- <https://github.com/akanametov/yolo-face/blob/1.0.0/LICENSE>
- <http://shuoyang1213.me/WIDERFACE/>
- <https://github.com/pytorch/pytorch/blob/main/LICENSE>

## External FFmpeg executable

Blur Face invokes an explicitly selected or system `ffmpeg` executable as a
separate process. It does not bundle the `imageio-ffmpeg` executable or an
FFmpeg build. FFmpeg licensing depends on how that executable was configured;
builds enabling GPL codecs such as libx264 have different redistribution
obligations from LGPL builds. Users and downstream distributors must inspect
the selected executable with `ffmpeg -L` and comply with its license.
On Windows, `init.bat` may offer to invoke WinGet for the external
`Gyan.FFmpeg` full build, but only after explicit confirmation; that package is
downloaded from its publisher and is not part of this repository or wheel.

Source: <https://ffmpeg.org/legal.html>

## Model and dependency policy

- Default model downloads are pinned by SHA-256 and have a recorded license.
- Optional YOLO downloads occur only after explicit selection and license notice.
- A user-selected local model remains subject to that model's own license and
  training-data terms.
- Runtime dependencies are installed from lock files rather than copied into
  this source repository.
- Downstream binary or offline-bundle distributors must preserve all applicable
  licenses and notices for the exact artifacts they redistribute.
