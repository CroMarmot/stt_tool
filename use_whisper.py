# uv add torch torchaudio openai-whisper typer rich pydantic pydantic-settings
# 模型: https://huggingface.co/openai/whisper-medium
#     mirror: https://hf-mirror.com/openai/whisper-medium
# huggingface 上,一个B是一个Billion
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pydantic
import pydantic_settings
import torch
import typer
import whisper
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from whisper.utils import WriteSRT

# 设置 Hugging Face 镜像与缓存路径
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/.cache"


class Settings(BaseSettings):
    whisper_download_root: str = "/data/.cache/whisper"
    transcribe_verbose: bool = False
    default_model_size: str = "small"


settings = Settings()

# 初始化 rich
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="whisper-transcribe",
    help="🎙️ 使用官方 OpenAI Whisper 模型转录音频，输出带标点的中文 SRT 字幕",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)


def validate_input_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        logger.error("路径不存在：%s", path)
        raise typer.Exit(1)
    return path


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


def write_srt_only(result, audio_name: str, output_dir: Path):
    srt_writer = WriteSRT(str(output_dir))
    srt_writer(
        result,
        audio_name,
        {
            "highlight_words": False,
            "max_line_width": None,
            "max_line_count": None,
        },
    )


def filter_existing_files(mp3_files: list[Path], output_dir: Path) -> list[Path]:
    files_to_process = []
    for mp3_file in mp3_files:
        if has_existing_srt(mp3_file, output_dir):
            console.print(f"[dim]跳过已处理：[/dim] {mp3_file.name}")
        else:
            files_to_process.append(mp3_file)
    return files_to_process


def load_whisper_model(model_name: str):
    console.print(f"正在加载模型 '{model_name}'...")
    model_obj = whisper.load_model(model_name, download_root=settings.whisper_download_root)

    device_type = model_obj.device.type
    if device_type == "cuda":
        device_name = torch.cuda.get_device_name(model_obj.device)
        device_info = f"GPU: {device_name}"
    elif device_type == "mps":
        device_info = "Apple Silicon MPS"
    else:
        device_info = "CPU"
    console.print(f"计算设备：[cyan]{device_info}[/cyan]")
    return model_obj


def process_files_with_progress(
    files_to_process: list[Path], model_obj, language: str, temp_dir: Path, output_dir: Path
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task("[blue]总体进度", total=len(files_to_process))

        for mp3_file in files_to_process:
            task_id = progress.add_task(f"[yellow]处理中：{mp3_file.name}", total=1)

            try:
                result = model_obj.transcribe(
                    str(mp3_file),
                    language=language,
                    word_timestamps=True,
                    verbose=settings.transcribe_verbose,
                )

                write_srt_only(result, mp3_file.stem + mp3_file.suffix, temp_dir)

                temp_srt = temp_dir / (mp3_file.stem + ".srt")
                final_srt = output_dir / (mp3_file.stem + ".srt")
                shutil.move(str(temp_srt), str(final_srt))

                progress.update(task_id, advance=1, description=f"[green]完成：{mp3_file.name}")
                progress.update(main_task, advance=1)
            except Exception:
                progress.update(task_id, description=f"[red]失败：{mp3_file.name}")
                logger.exception("处理文件失败：%s", mp3_file.name)
                temp_srt = temp_dir / (mp3_file.stem + ".srt")
                if temp_srt.exists():
                    temp_srt.unlink()


@app.command()
def transcribe(
    input_path: str = typer.Argument(..., help="🎧 输入音频文件或文件夹路径"),
    model: str = typer.Option(
        settings.default_model_size,
        "--model",
        "-m",
        help="🧠 Whisper 模型大小",
        case_sensitive=False,
        autocompletion=lambda: [
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v1",
            "large-v2",
        ],
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="📁 输出目录（默认为当前目录）"),
    language: str = typer.Option("zh", "--language", "-l", help="🌐 语言代码（zh=中文）"),
):
    """
    转录音频并生成 SRT 字幕格式
    """
    input_p = validate_input_path(input_path)
    mp3_files = get_mp3_files(input_p)

    if not mp3_files:
        logger.error("未找到 MP3 文件")
        raise typer.Exit(1)

    output_dir = output or Path()
    output_dir.mkdir(exist_ok=True)

    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)

    files_to_process = filter_existing_files(mp3_files, output_dir)

    if not files_to_process:
        console.print("[green]所有文件都已处理完成！[/green]")
        return

    console.print(f"[bold]找到 {len(files_to_process)} 个文件待处理[/bold]")

    try:
        model_obj = load_whisper_model(model)
    except Exception as err:
        logger.exception("加载模型失败：")
        raise typer.Exit(1) from err

    process_files_with_progress(files_to_process, model_obj, language, temp_dir, output_dir)

    console.print("\n[bold green]处理完成！[/bold green]")


@app.command()
def info():
    """📄 显示当前配置信息"""
    info_table = Table("配置项", "值", title="🔧 当前配置", show_lines=True)
    info_table.add_row("HF 镜像", os.environ["HF_ENDPOINT"])
    info_table.add_row("缓存目录", os.environ["HF_HOME"])
    info_table.add_row("Whisper 下载目录", settings.whisper_download_root)
    info_table.add_row("默认模型", settings.default_model_size)
    console.print(info_table)


@app.command()
def env():
    """🖥️ 显示系统和GPU环境诊断信息"""

    # 系统信息
    console.print("[bold cyan]=== 系统信息 ===[/bold cyan]")
    system_table = Table("项目", "值", show_lines=False)
    system_table.add_row("操作系统", platform.platform())
    system_table.add_row("Python版本", sys.version)
    system_table.add_row("Python路径", sys.executable)
    system_table.add_row("工作目录", str(Path.cwd()))
    console.print(system_table)

    # PyTorch 信息
    console.print("\n[bold cyan]=== PyTorch 信息 ===[/bold cyan]")
    torch_table = Table("项目", "值", show_lines=False)
    torch_table.add_row("PyTorch版本", torch.__version__)
    torch_table.add_row("CUDA版本(编译时)", torch.version.cuda or "N/A")
    torch_table.add_row("CUDA可用", "[green]是[/green]" if torch.cuda.is_available() else "[red]否[/red]")
    torch_table.add_row("CUDA设备数量", str(torch.cuda.device_count()) if torch.cuda.is_available() else "0")

    if torch.cuda.is_available():
        torch_table.add_row("当前设备", str(torch.cuda.current_device()))
        torch_table.add_row("设备名称", torch.cuda.get_device_name(0))
        torch_table.add_row("显存总量", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch_table.add_row("显存可用", f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    if hasattr(torch.backends, "mps"):
        torch_table.add_row("MPS可用", "[green]是[/green]" if torch.backends.mps.is_available() else "[red]否[/red]")

    console.print(torch_table)

    # NVIDIA SMI
    console.print("\n[bold cyan]=== NVIDIA GPU 信息 ===[/bold cyan]")
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path:
        try:
            result = subprocess.run([nvidia_smi_path], check=False, capture_output=True, text=True, timeout=10)  # noqa: S603
            if result.returncode == 0:
                console.print(result.stdout)
            else:
                console.print("[yellow]nvidia-smi 执行失败[/yellow]")
                console.print(result.stderr)
        except FileNotFoundError:
            console.print("[yellow]未找到 nvidia-smi 命令[/yellow]")
        except OSError as e:
            console.print(f"[red]nvidia-smi 出错: {e}[/red]")
    else:
        console.print("[yellow]未找到 nvidia-smi 命令[/yellow]")

    # 依赖版本
    console.print("\n[bold cyan]=== 依赖版本 ===[/bold cyan]")
    deps_table = Table("包名", "版本", show_lines=False)
    deps_table.add_row("whisper", whisper.__version__)
    deps_table.add_row("torch", torch.__version__)
    deps_table.add_row("typer", typer.__version__)
    deps_table.add_row("pydantic", pydantic.__version__)
    deps_table.add_row("pydantic-settings", getattr(pydantic_settings, "__version__", "N/A"))
    console.print(deps_table)


if __name__ == "__main__":
    app()
