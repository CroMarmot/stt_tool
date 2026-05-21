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

### 单个文件转录
```bash
uv run use_whisper.py transcribe 你的音频文件.mp3 --output ./output
```

### 批量处理文件夹
```bash
# 非递归处理目录下所有 MP3
uv run use_whisper.py transcribe ./你的文件夹 --output ./output
```

### 指定模型和语言
```bash
uv run use_whisper.py transcribe 你的音频文件.mp3 --model medium --language zh --output ./output
```

### 查看配置
```bash
uv run use_whisper.py info
```

### 环境诊断
```bash
uv run use_whisper.py env
```
查看系统信息、GPU 状态、依赖版本，用于诊断问题。

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

## 核心特性

### 文件夹批量处理
- 支持输入单个 MP3 或整个文件夹
- 非递归处理，只处理指定目录
- 自动按文件名排序处理

### 自动跳过已处理
- 检测输出目录中同名 SRT 文件
- 已处理文件自动跳过，避免重复工作

### Rich 进度显示
- 动态展示总体进度条
- 实时显示当前处理文件
- 自动计算剩余时间

### 临时文件夹机制
- 文件先写入 `.temp` 临时目录
- 成功处理后才移动到最终位置
- 失败时自动清理临时文件

### 仅输出 SRT
- 只生成标准字幕格式
- 避免冗余文件

---

## 常见问题

### Q: 为什么转录速度很慢？
A: Whisper large 模型在 CPU 上确实很慢，建议：
- 使用 GPU（CUDA）加速
- 选择 smaller 模型

### Q: 中文标点效果不好？
A: 使用 medium 或 large 模型，small 模型标点能力有限。

### Q: 内存不够怎么办？
A: 降低模型大小，tiny/base 占用内存小。

### Q: 支持哪些音频格式？
A: 当前仅支持 MP3 格式。
