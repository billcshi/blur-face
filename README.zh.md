# blur-face

**基于 YOLO + 自研跟踪器的人脸自动打码工具 — 跨帧跟踪、遮挡预测、无闪烁、无漏帧。**

> English version: [README.md](README.md)

## 这个项目为什么存在

我写这个工具的初衷是给**自己的情色视频打码**，然后分享给伴侣。因此每一个设计决策都把隐私和本地执行放在首位：

- **100% 本地运行** — 不上传、不联网、不收集数据。所有处理都在你自己的机器上完成
- **完全开源 (MIT)** — 每一行代码都可以审查，可以自己编译运行
- **零外部 tracker 依赖** — 不使用 supervision、不使用 ultralytics 内置 tracker，不会被第三方库的版本变更影响
- **iPhone 兼容输出** — 视频始终在你自己的设备上，不经过任何云服务

如果你需要处理私密内容，你需要一个能完全信任的工具。不是 SaaS。不是黑盒 App。而是一个可以审计、可以离线跑、可以自己验证的开源工具。

## 为什么不用现有的工具

市面上已有的人脸打码工具（如 deface）逐帧独立检测，存在几个痛点：

| 问题 | 表现 |
|------|------|
| 无跨帧跟踪 | 模糊框每帧重新计算，大小位置跳动，视觉上像在"跳舞" |
| 遮挡漏帧 | 转头、手挡、侧脸的几帧检测不到 → 突然露脸 |
| 无法微调 | 一个阈值跑全程，暗光/强光/远脸通用性差 |
| 无法选择性打码 | 所有脸都糊，不能指定"这张脸不打码" |

blur-face 的核心设计就是解决这四个问题。

## 核心设计

```
视频帧 → YOLO人脸检测 → 自研FaceTracker（跟踪+预测）→ 模糊/标注 → ffmpeg编码H.264
                              │
                ┌─────────────┼─────────────┐
                │ • 质心距离匹配检测框→已有轨迹   │
                │ • EMA指数平滑坐标（不跳变）     │
                │ • 检测不到时保留旧位置（预测）   │
                │ • 超过N帧未检测到→删除轨迹      │
                └─────────────────────────────┘
```

**关键设计决策：每帧都调用 tracker.update()，包括空检测。**

市面上其他工具（包括 deface 的 supervision ByteTrack、ultralytics 的 model.track()）在检测为空时会跳过 tracker，导致遮挡帧无输出。blur-face 的自研 tracker 无论 YOLO 有没有检测到脸，都继续维护所有轨迹——检测不到就用上一帧位置继续模糊。

## 跟 deface 的对比

| | deface | blur-face |
|---|---|---|
| 检测模型 | SCRFD（固定） | YOLO（可换模型） |
| 跨帧跟踪 | ❌ 逐帧独立 | ✅ 自研 tracker |
| 遮挡预测 | ❌ 漏检就露脸 | ✅ 保留位置继续糊 |
| 坐标平滑 | ❌ 每帧跳动 | ✅ EMA 指数平滑 |
| Debug 审查 | ❌ | ✅ `--debug` 彩色框+ID |
| 选择性排除 | ❌ | ✅ `--exclude-ids 2,5` |
| 按时间调阈值 | ❌ | ✅ `--time-thresh "0:0.15,120:0.3"` |
| iPhone 兼容 | ⚠️ | ✅ H.264+AAC+faststart |
| 外部依赖 | ONNX + supervision | 零外部 tracker 依赖 |

## 快速开始

```bash
# Windows — 双击或运行
init.bat

# Linux / macOS
chmod +x init.sh && ./init.sh
```

这会安装 Python 依赖并下载人脸检测模型到 `models/` 目录。

```bash
# Debug 模式：看每个人脸的跟踪ID（用于审查）
python blur-face.py video.mov --debug

# 打码全部人脸
python blur-face.py video.mov -o output.mp4

# 排除特定人脸（ID 2 和 5 不打码）
python blur-face.py video.mov --exclude-ids 2,5

# 不同时间段用不同阈值（前2分钟暗光低阈值，后面正常）
python blur-face.py video.mov --time-thresh "0:0.15,120:0.3"
```

## 微调选项

所有参数都有合理的默认值，但你可以精细控制：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 人脸检测模型 | `yolov11m-face.pt` |
| `--thresh` | 检测置信度阈值（越低越敏感，漏脸更少但误检更多） | `0.2` |
| `--time-thresh` | 分时间段阈值，格式 `"秒:阈值,秒:阈值"` | 无 |
| `--mask-scale` | 模糊区域相对于检测框的放大倍数 | `1.15` |
| `--blur-kernel` | 高斯模糊核大小（奇数，越大越模糊） | `51` |
| `--smooth` | 坐标平滑系数 0-1（0=完全不动, 1=完全不平滑） | `0.7` |
| `--lost-buffer` | 检测丢失后保留轨迹的帧数（约 60 帧 = 2 秒） | `60` |
| `--exclude-ids` | 指定不打码的跟踪ID，逗号分隔 | 无 |
| `--debug` | 审查模式：画彩色框+ID，不模糊 | 关 |
| `--profile` | 显示各阶段耗时（检测/跟踪/模糊/编码） | 关 |
| `--device` | `cuda` 或 `cpu` | `cuda`（自动降级） |

### 典型工作流

```bash
# 1. 先用 debug 模式看哪些脸需要打码
python blur-face.py video.mov --debug

# 2. 发现问题：开头光线暗，ID 3 偶尔漏检
#    尾部 ID 2 是朋友的脸，不需要打码

# 3. 精确控制
python blur-face.py video.mov -o output.mp4 \
    --time-thresh "0:0.15,45:0.2" \   # 前45秒暗光降阈值
    --exclude-ids 2 \                  # 保留朋友的脸
    --blur-kernel 71 \                 # 更重的模糊
    --profile                          # 看性能数据
```

## 性能

| GPU | 模型 | 视频 | 速度 |
|-----|------|------|------|
| RTX 3080 Ti | yolov11m-face.pt | 1080p 30fps | ~34fps（略慢于实时） |
| RTX 3080 Ti | yolo26n-face.pt | 1080p 30fps | ~43fps（快于实时） |
| CPU (i7-11700K) | yolov11m-face.pt | 1080p 30fps | ~6fps（5倍慢于实时） |

可用模型（来自 [yolo-face releases](https://github.com/akanametov/yolo-face/releases)）：

| 模型文件 | 大小 | 速度 | 推荐场景 |
|---------|------|------|---------|
| `yolo26n-face.pt` | 5.6 MB | 最快 | 光线好、正脸为主 |
| `yolov10n-face.pt` | 5.5 MB | 快 | 快速预览 |
| `yolov10s-face.pt` | 15.7 MB | 中等 | 平衡选择 |
| `yolov11m-face.pt` | 38.6 MB | 较慢 | 侧脸、暗光、边缘脸（默认推荐） |
| `yolov11l-face.pt` | 48.8 MB | 最慢 | 追求极致检测率 |

## 项目结构

```
blur-face/
├── blur-face.py          入口（流程编排）
├── init.bat / init.sh    一键安装脚本
├── blur-batch.bat        批量处理当前文件夹所有视频
├── README.md             English
├── README.zh.md          中文
├── LICENSE               MIT
└── blurface/             核心包
    ├── cli.py            命令行参数解析
    ├── detector.py       YOLO 人脸检测封装
    ├── tracker.py        自研多目标人脸跟踪器
    ├── renderer.py       模糊与 debug 标注
    ├── encoder.py        ffmpeg H.264 编码管道
    └── profiler.py       分阶段性能计时
```

每个模块职责单一、独立可测。tracker 完全自研，零外部依赖（不依赖 supervision、不依赖 ultralytics 内置 tracker）。

## 依赖

- Python 3.10+
- CUDA GPU 推荐（CPU 也能跑，慢 5 倍）
- `pip install ultralytics opencv-python imageio-ffmpeg numpy`
- GPU 检测使用 PyTorch CUDA，GPU 编码使用 NVENC。GPU 模糊渲染还需要带 CUDA 的 OpenCV；普通 `opencv-python` wheel 会自动降级到 CPU 渲染。

## License

MIT © 2025 Jiechang Shi
