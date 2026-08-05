# Blur Face Local

**离线、本地优先的视频与图片人脸匿名化工具，提供本地网页 UI 和 CLI。**

[English](README.md) · [版本记录](CHANGELOG.md) ·
[时序设计](docs/temporal-stabilization.md)

Blur Face 在本机完成视频或图片的解码、分析、遮挡和写入。网页只连接
`127.0.0.1`，不会上传媒体内容。

## 遮挡模式

- **快速几何遮挡**是默认模式。所选检测器检测人脸，本地 tracker
  填补短暂漏检，现有几何 mask 以低启动开销单遍渲染。
- **SAM 2.1 高质量遮挡**使用检测框发现人脸和定期纠偏，再由 SAM
  video memory 产生贴合人脸的轮廓。它使用离线两遍流程，因此可以在后来
  检测到人脸时修补较早帧，并在渲染前对跨 Track 的最终 mask 做运动对齐和稳定。

本项目不再依赖人脸解析/BiSeNet。默认的
`face_detection_yunet_2023mar.onnx` 来自 OpenCV Zoo，采用 MIT 许可；
OpenCV 4.5+ 以及官方 SAM 2 代码和 checkpoint 使用 Apache-2.0。
Ultralytics YOLO 只作为明确选择的可选检测器提供：Ultralytics 默认适用
AGPL-3.0，除非另有 Enterprise 许可；`akanametov/yolo-face` 1.0.0 权重
按上游声明适用 GPL-3.0 或相应 Enterprise 许可。其训练说明使用 WIDER FACE，
该数据集自身条款也继续适用。基础环境不会安装 YOLO，也不会因发现 `.pt`
文件而自动启用。

## 静态图片与批处理

1.2.0 版将相同的检测、几何覆盖、模糊、预览和可选 SAM 策略用于 JPEG、
PNG、WebP、BMP 与 TIFF 图片。CLI 可接收单张图片、多张图片路径，或一个
非递归图片目录。本地工作台增加双语的“视频 / 图片”选择，以及原生的多图
和输出文件夹选择器。整批任务只加载一次检测器和可选 SAM 模型，内存中只
保留当前图片的工作数据。

每张完成的图片都会原子提交；未开启 `--overwrite` 时，处理中途新出现的
目标文件不会被替换。输出图片有意不复制 EXIF 等源文件元数据，也因此移除
其中可能包含的位置和相机信息。PNG、WebP 和 TIFF 的透明通道会保留；若为
带透明通道的图片选择无法保留透明度的输出格式，任务会失败且不会提交该文件。

## 快速开始

需要 Python 3.10 或更高版本，以及系统 FFmpeg。

### Windows

```powershell
git clone https://github.com/billcshi/blur-face.git
cd blur-face
.\init.bat
.\start-ui.bat
```

### Linux / macOS

```bash
git clone https://github.com/billcshi/blur-face.git
cd blur-face
chmod +x init.sh start-ui.sh
./init.sh
./start-ui.sh
```

初始化会分别显示创建环境、安装依赖、下载小型 YuNet 模型以及总耗时。
几何模式不安装 PyTorch。初始化会重建隔离的 `.venv`，但保留模型和用户
文件，从而避免升级后继续启用已删除的旧依赖。Windows 缺少 FFmpeg 时，
`init.bat` 会先说明 Gyan 完整版是体积较大的外部 GPL 软件，再询问是否通过
WinGet 安装；Unix 初始化只提示系统包依赖。

Windows 初始化会分别明确询问是否安装可选 YOLO 检测器和 SAM 2.1 mask
引擎。选择“否”后仍是精简的 MIT/Apache 默认 YuNet 几何环境；CI 会跳过
这两个提示。也可以稍后运行：

```powershell
.\install-yolo.bat
.\install-sam2.bat
```

```bash
./install-yolo.sh
```

可选安装器会再次要求确认上游许可，安装 PyTorch/Ultralytics，并下载经过
SHA-256 固定校验的 `yolov11m-face.pt`。确认前会明确显示该权重的上游
GPL-3.0/Enterprise 边界和 WIDER FACE 训练数据来源。CI 和非交互初始化
默认跳过 YOLO；准确的上游链接见 `THIRD_PARTY_NOTICES.md`。

Linux/macOS 初始化后，或 Windows 提示中选择跳过后，如需 SAM 可运行：

```powershell
.\install-sam2.bat
```

```bash
./install-sam2.sh
```

SAM 安装器在检测到 NVIDIA 时安装已测试的 CUDA 运行时，否则安装 CPU
运行时。命名的 SAM checkpoint 可以在模型初始化阶段、分析开始之前下载；
缓存后可启用离线模式，也可选择本地模型目录。

普通 `git pull` 不需要重新初始化，除非依赖 lock 文件或安装脚本发生变化。

## 本地工作台

主界面保留媒体类型、输入、输出、遮挡模式和隐私预设。图片模式可选择一张
或多张本机图片及输出文件夹，视频模式保持原有单输入/输出流程。没有安装可选 YOLO 时，遮挡
模式中只有“快速几何 · YuNet”和“SAM 高质量 · YuNet”两项；完整安装 YOLO
后，同一个选择框会增加“快速几何 · YOLO”和“SAM · YOLO”。展开
“**高级参数**”后，只显示与所选引擎相关的设置：

- 选择与当前组合模式中检测器匹配的模型；
- 通用模糊、追踪、输出和诊断参数；
- 几何遮挡及 SAM fallback 的形状和范围；
- SAM 模型、设备、纠偏间隔、合并方式、时序窗口、切镜灵敏度和临时存储上限。

隐私预设不会再静默切换遮挡引擎。UI 会自动选择中英文，也可手动选择
`Auto / EN / 中文`。
YOLO 自动发现严格限制为已校验的 `yolov11m-face.pt`；其他本地 YOLO 权重
还必须在运行时通过 `detect` task 和明确的 `face` 类检查。

启用“**遮挡区域测试视频**”后，输出为黑色背景，蓝色像素就是正常结果最终会
模糊的准确范围；其中不包含原视频像素、音频、检测框、Track ID 或文字。

进度百分比会单调覆盖分析、Track 稳定、最终场景 mask 稳定和渲染编码。
SAM 日志会显示模型初始化耗时、设备、视频信息、镜头数、纠偏次数、prompt、
有效/fallback mask、反向补帧、memory reset、各 pass 工作量、整个任务 ETA、
临时存储峰值和各阶段耗时。

## SAM mask 语义

先在每个 Track 内做时序修正，再合并并平滑最终场景 mask：

1. 分析视频并保存有上限的代理分辨率 mask；
2. 在 `(scene_id, track_id)` 内修补和稳定；
3. 应用 SAM 与几何范围的合并策略；
4. 合并所有 Track，并稳定最终场景 mask；
5. 重新读取原视频并统一渲染。

合并方式：

- `union`：将当前 tracker 几何范围加入 SAM 轮廓，推荐；
- `intersection`：将 SAM 轮廓裁剪在当前几何范围内；
- `mask-only`：有效时只渲染 SAM 轮廓，不加入检测框。

稀疏、单轴细条、覆盖不足、破碎、空白、缺失或非有限分数、失败或漂移的
SAM mask，在所有组合方式下都会使用配置的几何 fallback。检测到切镜后会
清除 tracker、SAM memory、反向缓存和时序历史；两人交叉且身份不明确时，
不会用这些 Track ID 提示 SAM。

默认新出现的可靠人脸最多在同一连续镜头内向前修补 10 帧。遇到切镜、真实
边缘进入、不合理运动或缩放、外观证据不足或与他人重叠时立即停止。代理帧、
mask 和 SQLite 元数据默认有 4096 MiB 硬上限，并在成功、失败或 UI 取消后清理。

## CLI

```powershell
.\.venv\Scripts\blur-face.exe input.mov -o output.mp4 --overwrite

.\.venv\Scripts\blur-face.exe input.mov -o output.mp4 `
  --mask-engine sam2.1 `
  --detector yolo `
  --model yolov11m-face.pt `
  --segmentation-combine union `
  --device auto `
  --overwrite
```

```bash
.venv/bin/blur-face input.mov -o output.mp4 --overwrite

.venv/bin/blur-face portrait.jpg -o portrait_blurred.jpg
.venv/bin/blur-face first.jpg second.png -o blurred-images/
.venv/bin/blur-face photos/ -o blurred-images/
```

常用默认值：

| 参数 | 默认值 | 用途 |
|---|---:|---|
| `--detector` | `yunet` | `yunet` 或明确安装的 `yolo` |
| `--model` | `face_detection_yunet_2023mar.onnx` | 本地 YuNet ONNX 模型 |
| `--thresh` | `0.3` | 检测置信度阈值 |
| `--mask-engine` | `geometric` | `geometric` 或 `sam2.1` |
| `--mask-shape` | `rounded-rect` | 几何和 SAM fallback 形状 |
| `--mask-scale` | `1.5` | 几何和 fallback 扩张范围 |
| `--segmentation-combine` | `union` | `union`、`intersection` 或 `mask-only` |
| `--sam-mask-expansion` | `0.12` | SAM 轮廓扩张比例 |
| `--sam2-model` | `facebook/sam2.1-hiera-base-plus` | SAM 目录或模型 ID |
| `--sam2-refresh-interval` | `15` | 所选检测器纠偏间隔 |
| `--device` | `auto` | SAM/YOLO 设备：自动、CPU、CUDA 或 MPS |
| `--[no-]temporal-stabilization` | 开 | 离线两遍时序稳定 |
| `--backfill-frames` | `10` | 同镜头反向修补窗口 |
| `--release-hold-frames` | `5` | 对齐后轮廓过渡窗口 |
| `--scene-cut-sensitivity` | `0.55` | 镜头切换重置灵敏度 |
| `--temporal-storage-limit-mb` | `4096` | 临时存储硬上限 |
| `--mask-preview` | 关 | 黑/蓝最终 mask 测试视频 |

运行 `blur-face --help` 查看全部追踪、模糊、编码和离线参数。

SAM checkpoint 首次使用时可能在分析开始前下载；后续任务会先尝试完整的
本地缓存，不会仅为检查更新而访问 Hub。勾选离线模式可强制只用缓存，也可
选择本地模型目录。UI 的每个任务是独立进程，因此每个任务会加载一次权重；
同一个图片批次内则会复用这一个模型实例。

公开 Hugging Face checkpoint 无需账号。可选执行 Windows 的
`.venv\Scripts\hf.exe auth login`（Linux/macOS 为 `.venv/bin/hf auth login`），
可去除首次下载时的未认证提示并提高 Hub 限速；凭据由 Hugging Face 客户端
管理，本程序不会读取或保存 token。

## 隐私与输出安全

- 自动检测器无法保证 100% 召回；发布敏感视频或图片前必须检查结果。
- 所有模型初始化都在处理第一帧之前完成。
- 编码失败不会替换目标文件。只有 FFmpeg 成功退出且生成非空文件后，才会
  原子提交输出。
- 图片输出逐文件遵循相同的无残缺提交规则。批处理失败或取消时，已经完成的
  图片会保留，但当前未完成图片绝不会提交。
- UI 在目标文件系统创建唯一作业目录（图片任务位于输出文件夹内）；普通异常、
  取消以及强制终止进程树后都会清理其中的 mask、运动代理和 partial 输出。
- FFmpeg 是用户提供的外部系统工具，本项目不捆绑；其许可证取决于用户选择的
  FFmpeg 构建。

## 开发与 CI

```bash
python -m unittest discover -s tests -v
python -m compileall -q blurface tests scripts
git diff --check
```

GitHub Actions 在 Ubuntu 和 Windows 上使用 Python 3.10、3.12 运行 unittest
发现和编译检查；新增的 `tests/test_*.py` 会自动进入 CI。CI 会在视频集成
测试前显式准备外部系统 FFmpeg；默认矩阵不需要模型。维护者也可以在运行
unittest discovery 前，将
`BLUR_FACE_REAL_SAM2_MODEL` 指向本地 checkpoint 目录，以 CPU 实际验证
已安装 Transformers 的 processor/model/session 契约。

## 许可证

项目原创源码采用 MIT，© 2025 Jiechang Shi。依赖和模型资产保留各自许可证，
详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
