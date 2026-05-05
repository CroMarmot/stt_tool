# 🎙️ STT - 播客音频转文字工具

基于 OpenAI Whisper 的离线播客音频转文字工具，专为中文优化。

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync

# 安装 ffmpeg
sudo apt update && sudo apt install -y ffmpeg
```

### 2. 基本使用

```bash
# 基础转录
uv run use_whisper.py transcribe 你的播客.mp3 --output ./output

uv run use_whisper.py transcribe 你的播客.mp3 --model tiny --language zh --output ./output

# 查看所有参数
uv run use_whisper.py transcribe --help
```

## 📋 许可证

MIT License
