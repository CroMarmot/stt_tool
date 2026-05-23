#!/usr/bin/env python3
from pathlib import Path

import pydantic
import pydantic_settings
from rich.console import Console

console = Console()


class Settings(pydantic_settings.BaseSettings):
    app_name: str = "cloud-stt"
    ssh_alias: str = "stt-server"
    remote_work_dir: str = "~/stt-remote"

    @property
    def remote_work_path(self) -> Path:
        return Path(self.remote_work_dir)

    @property
    def remote_input_path(self) -> Path:
        return self.remote_work_path / "input"

    @property
    def remote_output_path(self) -> Path:
        return self.remote_work_path / "output"

    model_config = pydantic_settings.SettingsConfigDict(env_prefix="CLOUD_", env_file=".env", env_file_encoding="utf-8")


# Package 级别的单例
settings = Settings()

# 需要上传到远程服务器的本地文件
LOCAL_FILES_TO_UPLOAD = [
    "use_whisper.py",
    "pyproject.toml",
    "utils.py",
    "config.py",
]
