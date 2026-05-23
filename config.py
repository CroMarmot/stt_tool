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
    whisper_download_root: str = "/data/.cache/whisper"
    transcribe_verbose: bool = False
    default_model_size: str = "small"
    temp_threshold: int = 90
    temp_wait_minutes: int = 3
    hf_endpoint: str = "https://hf-mirror.com"
    hf_home: str = "/data/.cache"
    transformers_cache: str = "/data/.cache"
    log_dir: str = "./log"
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5

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
    "log_config.py",
]
