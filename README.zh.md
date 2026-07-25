# blur-face

面向隐私场景的本地视频人脸检测、跟踪和模糊工具。

blur-face 每帧运行 YOLO 人脸检测，并通过保守的多人跟踪、短时光流补偿和
原子 FFmpeg 输出降低漏糊与损坏输出的风险。处理失败时程序会明确返回错误，
不会把不存在或不完整的输出报告为成功。

## 隐私与联网行为

- 视频帧的解码、检测、模糊和编码全部在本机完成；逐帧处理循环不会上传视频，
  也不会主动发起网络请求。
- 如果本地缺少模型，模型初始化阶段可能在解码第一帧之前下载 YOLO 模型。
- 使用 `--offline` 可要求模型必须已经存在，并禁用上述自动下载路径。
- 任何自动检测器都无法保证找到所有人脸。分享敏感视频前必须人工审查成品。

## 安装

需要 Python 3.10 或更高版本。

```bash
# Linux / macOS
chmod +x init.sh
./init.sh

# Windows
init.bat
```

安装脚本会创建隔离的 `.venv`，安装 `requirements.lock` 中经过测试的直接依赖，
并下载两个模型。模型通过固定的 SHA-256 校验，只有完整下载成功后才会替换目标文件。

## 使用

```bash
# Linux / macOS
.venv/bin/blur-face input.mov -o output.mp4

# Windows
.venv\Scripts\blur-face.exe input.mov -o output.mp4

# 严格离线初始化；模型缺失时直接失败
.venv/bin/blur-face input.mov -o output.mp4 --offline

# 指定支持 NVENC 的 Windows FFmpeg
.venv\Scripts\blur-face.exe input.mov -o output.mp4 --ffmpeg C:\ffmpeg\bin\ffmpeg.exe

# 只画出保守覆盖区域，不模糊
.venv/bin/blur-face input.mov --debug -o review.mp4

# 按时间调整检测阈值
.venv/bin/blur-face input.mov --time-thresh "0:0.15,120:0.3"
```

在当前 Python 环境已经安装依赖时，原有的
`python blur-face.py ...` 入口仍然可用。

## 关键安全语义

对于当前已经检测到的人脸，最终渲染区域是“原始检测框”和“平滑跟踪框”的并集。
因此平滑只能增加覆盖，不能用一个滞后的框替换当前检测结果。近景大脸默认不会被
过滤；只有无效、极小或极端畸形的检测框会被丢弃。

`--exclude-ids` 只适合经过仔细审查的视频。Track ID 是临时运动轨迹，不是生物身份；
人物交叉或重新检测时仍可能换 ID。除非同时传入 `--allow-unsafe-exclusions`
明确接受风险，否则程序会拒绝运行；启用后仍会显示警告。

输出首先写入目标目录中的隐藏临时文件。只有 FFmpeg 正常退出且输出非空时才会
原子替换最终路径。中断或失败会删除临时文件并返回非零状态。已有输出默认不会
被覆盖；只有显式传入 `--overwrite` 才会在新文件完整完成后替换旧文件。

## 主要参数

以 `blur-face --help` 为唯一权威来源。当前关键默认值如下：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | `yolov11m-face.pt` | 本地路径或可下载模型名 |
| `--thresh` | `0.3` | YOLO 置信度阈值，范围 0–1 |
| `--mask-scale` | `1.35` | 在保守跟踪区域外继续扩大 |
| `--blur-kernel` | `51` | 正数高斯核；偶数会自动变为奇数 |
| `--lost-buffer` | `180` | 漏检后继续保留轨迹的帧数 |
| `--smooth` | `0.7` | EMA 中当前检测框的权重 |
| `--min-face-size` | `8` | 最小检测宽度和高度，单位像素 |
| `--max-face-height-ratio` | `1.0` | 人脸高度相对画面高度的上限 |
| `--preset` | `quality` | 光流计算策略 |
| `--offline` | 关闭 | 模型缺失时禁止下载并直接失败 |
| `--ffmpeg` | 系统/内置 | 显式指定 FFmpeg 可执行文件 |

所有影响安全的数值参数都会在加载模型或启动编码器之前验证。

## 模块结构

```text
CLI → AppConfig 校验 → VideoSource 输入探测
                         ↓
                     模型初始化
                         ↓
视频帧 → 检测 → 全局关联与光流 → 隐私覆盖区域
                         ↓
                   CPU/CUDA 渲染
                         ↓
               原子 FFmpeg 编码 → 成品
```

- `config.py`：类型化配置和约束
- `video.py`：经过验证的视频输入生命周期
- `detector.py`：本地优先的 YOLO 初始化
- `tracker.py`：全局匹配、轨迹状态和保守覆盖
- `renderer.py`：统一验证的 CPU/CUDA 模糊后端
- `encoder.py`：真实 NVENC 探测、FFmpeg 错误检查和原子提交
- `pipeline.py`：资源所有权和逐帧流程
- `app.py`：进程级错误与退出码

## 开发验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q blurface tests
```

测试覆盖当前检测框兜底、近景人脸、与检测顺序无关的关联、陈旧光流失效、
配置校验和 FFmpeg 失败处理。

## License

MIT © 2025 Jiechang Shi
