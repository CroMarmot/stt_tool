# 开发者指南

## 项目架构

```
stt/
├── use_whisper.py      # OpenAI Whisper 主程序
├── docs/
│   ├── user-guide.md       # 用户文档
│   └── developer-guide.md  # 开发者文档
├── pyproject.toml      # 项目依赖配置
└── ruff.toml          # 代码规范配置
```

## 运行环境

- **推荐系统**: Ubuntu 22.04 LTS / 20.04 LTS
- **依赖安装**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg python3-pip git curl
  ```

---

## 技术栈

| 组件     | 库                  | 用途                 |
| -------- | ------------------- | -------------------- |
| CLI 框架 | `typer`             | 命令行参数解析       |
| 输出美化 | `rich`              | 终端输出美化、进度条 |
| 配置管理 | `pydantic-settings` | 配置项管理           |
| 数据验证 | `pydantic`          | 数据模型验证         |
| 识别引擎 | `openai-whisper`    | OpenAI 官方 Whisper  |

---

## 核心流程

```python
1. 验证输入路径（文件或文件夹）
2. 扫描 MP3 文件
3. 过滤已处理文件
4. 加载 Whisper 模型
5. 逐个处理文件：
   - 转录音频
   - 写入临时文件夹
   - 成功后移动到输出目录
6. 显示进度和结果
```

---

## 代码规范

### 代码检查与格式化

项目使用 **Ruff** 进行代码检查和格式化：

```bash
# 代码检查（自动修复可修复问题）
uv run ruff check . --fix

# 代码格式化
uv run ruff format .
```

配置文件：[ruff.toml](../ruff.toml)

### 导入顺序
```python
# 标准库
import os
import shutil
from pathlib import Path

# 第三方库
import torch
import typer
import whisper
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.progress import Progress
```

---

## 配置管理

使用 `pydantic-settings` 管理配置：

```python
class Settings(BaseSettings):
    whisper_download_root: str = "/data/.cache/whisper"
    transcribe_verbose: bool = False
    default_model_size: str = "small"

settings = Settings()
```

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
# 修改 use_whisper.py
settings = Settings(transcribe_verbose=True)
```

---

## 贡献指南

1. Fork 本仓库
2. 创建功能分支
3. 提交变更
4. 确保文档同步更新
5. 提交 Pull Request


## 本地模拟时，多用户共享文件夹rwx权限

```bash
# setfacl 感觉 需要-R 递归授权，否则没有递归
# 所以还是考虑加用户组
sudo usermod -aG <现在权限主用户组 > <远端测试用户名>
# sudo usermod -aG esakura yexiaorain
```