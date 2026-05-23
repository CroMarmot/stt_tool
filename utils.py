import logging
from pathlib import Path

import typer
from rich.console import Console

logger = logging.getLogger("utils")
console = Console()


def get_mp3_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".mp3":
            return [input_path]
        logger.error("输入文件不是 MP3 格式：%s", input_path)
        raise typer.Exit(1)
    if input_path.is_dir():
        mp3_files = list(input_path.glob("*.mp3"))
        mp3_files.sort()
        return mp3_files
    return []


def has_existing_srt(mp3_file: Path, output_dir: Path) -> bool:
    srt_file = output_dir / (mp3_file.stem + ".srt")
    return srt_file.exists()


def filter_existing_files(mp3_files: list[Path], output_dir: Path) -> list[Path]:
    files_to_process = []
    for mp3_file in mp3_files:
        if has_existing_srt(mp3_file, output_dir):
            console.print(f"[dim]跳过已处理：[/dim] {mp3_file.name}")
        else:
            files_to_process.append(mp3_file)
    return files_to_process