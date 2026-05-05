# 用户使用指南

## 快速开始

### 环境准备

1. **安装 uv 包管理器**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **安装依赖**
   ```bash
   uv sync
   ```

3. **安装 ffmpeg** (音频处理必需)
   ```bash
   sudo apt update && sudo apt install -y ffmpeg
   ```

---

## 基本使用

### 转录音频
```bash
uv run use_whisper.py transcribe 你的音频文件.mp3
```

### 指定模型大小
```bash
uv run use_whisper.py transcribe 你的音频文件.mp3 --model medium
```

### 指定语言和输出目录
```bash
uv run use_whisper.py transcribe 你的音频文件.mp3 --language zh --output ./output
```

### 查看配置
```bash
uv run use_whisper.py info
```

### 模型大小对比
| 模型      | 体积  | 中文效果 | 内存占用 | GPU 显存 | 播客推荐       |
| --------- | ----- | -------- | -------- | -------- | -------------- |
| tiny      | ~1GB  | ⭐⭐       | ~1GB     | <1GB     | ✅✅✅✅✅ 速度优先 |
| base      | ~1GB  | ⭐⭐       | ~1GB     | <1GB     | -              |
| **small** | ~2GB  | ⭐⭐⭐      | ~2GB     | ~2GB     | ✅✅✅✅ 平衡首选  |
| medium    | ~5GB  | ⭐⭐⭐⭐     | ~5GB     | ~5GB     | ✅✅✅ 质量优先   |
| large     | ~10GB | ⭐⭐⭐⭐⭐    | ~10GB    | ~10GB    | ✅✅ 专业级      |

> 💡 **GTX 1050 2GB 用户强烈推荐 tiny 模型** - 温度从 93°C 降到 ~70°C，速度快 3 倍！

---

## 📜 关于 Vosk 方案

Vosk 方案已移除，专注于 Whisper。

**为什么移除 Vosk？** 对于播客场景：
- Whisper 准确率、标点、断句全面碾压
- 省下的转写时间，会在校对时全部赔回去
- 详见：[unused.md - Vosk 方案完整说明与对比](unused.md)

---

## 输出格式说明

### 支持格式

| 格式     | 文件后缀 | 特点                 | 适用场景             |
| -------- | -------- | -------------------- | -------------------- |
| **TXT**  | `.txt`   | 纯文本，无时间戳     | 阅读、文稿整理、搜索 |
| **SRT**  | `.srt`   | 标准字幕，精确到毫秒 | 视频剪辑、精确定位   |
| **LRC**  | `.lrc`   | 歌词格式，秒级精度   | 播客播放、边听边看   |
| **JSON** | `.json`  | 完整元数据           | 二次开发             |

### 格式选择建议

#### 🎧 播客内容
1. **日常收听 + 阅读**: TXT + SRT（推荐）
2. **在播放器中显示歌词式字幕**: LRC
3. **后期剪辑制作**: SRT

#### 📝 会议/访谈记录
- **仅需文字内容**: TXT
- **需要回溯发言时间**: SRT

---

## 常见问题

### Q: 为什么转录速度很慢？
A: Whisper large 模型在 CPU 上确实很慢，建议：
- 使用 GPU（CUDA）加速
- 选择 smaller 模型
- 使用 Vosk 方案

### Q: 中文标点效果不好？
A: 使用 medium 或 large 模型，small 模型标点能力有限。

### Q: 内存不够怎么办？
A: 降低模型大小，tiny/base 占用内存小。

### Q: 支持哪些音频格式？
A: 支持几乎所有主流格式：mp3, m4a, wav, flac, ogg 等。

---

---

## Docker 部署

### 使用已有环境
```bash
# 构建镜像
docker build -t stt-whisper .

# 运行容器（挂载音频目录）
docker run -it --rm -v /path/to/your/audio:/app/audio stt-whisper
```

### Dockerfile 参考
```dockerfile
FROM python:3.11-slim

RUN apt update && apt install -y ffmpeg git curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"
ENV HF_ENDPOINT="https://hf-mirror.com"

WORKDIR /app
COPY . .
RUN uv sync

ENTRYPOINT ["uv", "run", "use_whisper.py"]
```

---

## 示例输出

### TXT 格式
```
大家好，欢迎收听本期播客。
今天我们来聊一聊人工智能的发展。
```

### SRT 格式
```
1
00:00:00,500 --> 00:00:03,200
大家好，欢迎收听本期播客。

2
00:00:04,100 --> 00:00:08,500
今天我们来聊一聊人工智能的发展。
```

### LRC 格式
```
[00:00.50]大家好，欢迎收听本期播客。
[00:04.10]今天我们来聊一聊人工智能的发展。
```
