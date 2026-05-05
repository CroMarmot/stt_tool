# 开发者指南

## 项目架构

```
stt/
├── use_whisper.py      # OpenAI Whisper 主程序（推荐）
├── use_vosk.py         # Vosk 离线识别程序
├── docs/
│   ├── user-guide.md       # 用户文档
│   └── developer-guide.md  # 开发者文档
├── pyproject.toml      # 项目依赖配置
└── Dockerfile          # 容器化部署（待实现）
```

## 运行环境

- **推荐系统**: Ubuntu 22.04 LTS / 20.04 LTS
- **部署方式**: 本地运行 → Docker 容器化
- **依赖安装**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg python3-pip git curl
  ```

---

## 技术栈

| 组件     | 库                 | 用途                  |
| -------- | ------------------ | --------------------- |
| CLI 框架 | `typer`            | 命令行参数解析        |
| 输出美化 | `rich`             | 终端输出美化、进度条  |
| 音频处理 | `ffmpeg` / `pydub` | 音频格式转换          |
| 识别引擎 | `openai-whisper`   | OpenAI 官方 Whisper   |
| 识别引擎 | `vosk`             | 离线语音识别          |
| 标点恢复 | `transformers`     | Hugging Face 标点模型 |

---

## Whisper 模块开发

### 核心流程

```python
1. 验证音频文件路径
2. 加载 Whisper 模型
3. 执行转录（带 word_timestamps=True）
4. 生成各种格式输出
```

### 扩展输出格式

Whisper 内置 writer 支持：
- `txt` - 纯文本
- `srt` - 标准字幕
- `vtt` - WebVTT 字幕
- `json` - 完整结果
- `tsv` - 表格格式
- `audacity` - Audacity 标签


---

## Vosk 模块开发

### 核心流程

```python
1. 加载 Vosk 模型
2. 音频转 16kHz 单声道 WAV
3. 流式识别（4000 bytes/块）
4. 解析 JSON 结果
5. 中文标点恢复（可选）
6. 生成 TXT/SRT 输出
```

### 标点恢复实现

```python
class ChinesePunctuator:
    def __init__(self, model_name="cantonese-zh/punc_ctc_zh_base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
```

---

## 代码规范

### 代码检查与格式化

项目使用 **Ruff** 进行代码检查和格式化，已启用 `ALL` 规则集：

```bash
# 代码检查（自动修复可修复问题）
uv run ruff check . --fix

# 代码格式化
uv run ruff format .
```

配置文件：[ruff.toml](../ruff.toml)
- ✅ 启用所有规则 (`select = ["ALL"]`)
- ✅ 行长度：120
- ✅ 双引号风格
- ✅ 忽略对 CLI 工具过于严格的规则

### 导入顺序
```python
# 标准库
import os
from pathlib import Path

# 第三方库
import typer
import whisper
from rich.console import Console

# 本地模块
# from . import xxx
```

### 日志规范
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("信息")
logging.warning("警告")
logging.error("错误")
logging.exception("异常带堆栈")
```

### 提交前检查
```bash
# 确保代码通过所有检查
uv run ruff check .
uv run ruff format . --check
```

---

## 测试指南

### 快速测试
```bash
# 小文件快速测试（tiny 模型）
uv run use_whisper.py transcribe test.wav --model tiny
```

### 中文测试音频
```bash
# 使用系统录音
ffmpeg -f avfoundation -i :1 -t 5 test.wav
```

---

## 性能优化

### Whisper 加速
1. **GPU 加速** - CUDA 自动检测
2. **量化模型** - 使用 int8 量化
3. **better-transformer** - FlashAttention 加速
4. **批处理** - 批量处理多个文件

### Vosk 加速
1. **多线程** - Vosk 自动使用 CPU 线程
2. **流式处理** - 边读边识别，降低内存

---

## 扩展开发计划

### 待实现功能
- [ ] LRC 歌词格式输出
- [ ] 说话人分离（diarization）
- [ ] Web UI 界面
- [ ] 批量处理目录
- [ ] 进度保存/断点续传
- [ ] Docker 镜像打包

---

## 调试技巧

### 查看完整识别结果
```python
import json
with open("debug.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
```

### 启用详细日志
```python
TRANSCRIBE_VERBOSE = True  # use_whisper.py
```

---

## 贡献指南

1. Fork 本仓库
2. 创建功能分支
3. 提交变更
4. 确保文档同步更新
5. 提交 Pull Request
