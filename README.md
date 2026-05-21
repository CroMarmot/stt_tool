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
# 单个文件转录
uv run use_whisper.py transcribe 你的播客.mp3 --output ./output

# 批量处理文件夹（非递归）
uv run use_whisper.py transcribe ./你的文件夹 --output ./output

# 指定模型和语言
uv run use_whisper.py transcribe 你的播客.mp3 --model medium --language zh --output ./output

# 查看环境诊断（GPU/系统信息）
uv run use_whisper.py env

# 查看所有参数
uv run use_whisper.py transcribe --help
```

## ✨ 核心功能

- 📁 **文件夹批量处理** - 非递归处理目录下所有 MP3 文件
- ⏭️ **自动跳过已处理** - 检测同名 SRT 文件自动跳过
- 📊 **Rich 进度显示** - 动态展示处理进度和当前文件
- 🔒 **临时文件夹机制** - 成功处理后才移动到最终位置
- 📝 **仅输出 SRT** - 只生成字幕文件，避免冗余

## 📋 许可证

MIT License
