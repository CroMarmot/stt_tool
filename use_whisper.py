# uv add torch torchaudio faster-whisper typer rich pydantic pydantic-settings
# 模型: https://huggingface.co/openai/whisper-medium
#     mirror: https://hf-mirror.com/openai/whisper-medium
# huggingface 上,一个B是一个Billion
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pydantic
import pydantic_settings
import torch
import torchaudio
import typer
from faster_whisper import WhisperModel
from rich.console import Console
from rich.table import Table

from config import settings
from log_config import setup_logging
from utils import filter_existing_files, format_temperatures, get_max_temperature, get_mp3_files

# 设置 Hugging Face 镜像与缓存路径
os.environ["HF_ENDPOINT"] = settings.hf_endpoint
os.environ["HF_HOME"] = settings.hf_home
os.environ["TRANSFORMERS_CACHE"] = settings.transformers_cache

# 初始化 rich
console = Console()

# 设置日志
setup_logging()
logger = logging.getLogger(f"{settings.app_name}.use_whisper")

app = typer.Typer(
    name="whisper-transcribe",
    help="🎙️ 使用官方 OpenAI Whisper 模型转录音频，输出带标点的中文 SRT 字幕",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)


def validate_input_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        logger.error("路径不存在：%s", path)
        raise typer.Exit(1)
    return path


def write_srt_only(segments, audio_name: str, output_dir: Path):
    """将 faster-whisper 的 segments 写入 SRT 文件"""
    srt_path = output_dir / f"{Path(audio_name).stem}.srt"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            # 格式化时间戳: 00:00:00,000
            start_hours = int(segment.start // 3600)
            start_minutes = int((segment.start % 3600) // 60)
            start_seconds = int(segment.start % 60)
            start_millis = int((segment.start % 1) * 1000)

            end_hours = int(segment.end // 3600)
            end_minutes = int((segment.end % 3600) // 60)
            end_seconds = int(segment.end % 60)
            end_millis = int((segment.end % 1) * 1000)

            start_str = f"{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d},{start_millis:03d}"
            end_str = f"{end_hours:02d}:{end_minutes:02d}:{end_seconds:02d},{end_millis:03d}"

            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{segment.text.strip()}\n\n")


def load_whisper_model(model_name: str):
    console.print(f"正在加载模型 '{model_name}'...")

    # 确定计算设备
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        device_info = f"GPU: {device_name}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_info = "Apple Silicon MPS"
    else:
        device = "cpu"
        device_info = "CPU"

    console.print(f"计算设备：[cyan]{device_info}[/cyan]")

    # GPU 计算类型选择
    # - GTX 10xx (Pascal, compute capability 6.x): 不支持 float16，只能用 float32
    # - RTX 20xx (Turing, compute capability 7.5): 可用 float16
    # - RTX 30xx / A10 / A100 (Ampere, compute capability 8.x): 推荐 int8_float16
    # - RTX 40xx / H100 (Ada/Hopper, compute capability 9.x): 可用 bfloat16
    compute_type = "float32"
    if device == "cuda":
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 8:  # Ampere or newer (8.x, 9.x)  # noqa: PLR2004
            compute_type = "int8_float16"
        elif cap[0] >= 7:  # Volta/Turing (7.x)  # noqa: PLR2004
            compute_type = "float16"

    # 加载 faster-whisper 模型
    model_obj = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=settings.whisper_download_root,
    )

    return model_obj


def get_audio_duration(mp3_path: Path) -> float:
    """获取音频时长(秒)"""
    try:
        waveform, sample_rate = torchaudio.load(str(mp3_path))
        return waveform.shape[1] / sample_rate
    except (OSError, RuntimeError):
        # torchaudio 失败时尝试用 whisper 加载
        audio = whisper.load_audio(str(mp3_path))
        return len(audio) / whisper.audio.SAMPLE_RATE


def format_duration(seconds: float) -> str:
    """格式化时长为可读字符串"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h{minutes}m{secs}s"
    if minutes > 0:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


def process_files_with_progress(
    files_to_process: list[Path], model_obj, language: str, temp_dir: Path, output_dir: Path
):
    total = len(files_to_process)
    total_audio_duration = 0.0
    total_process_time = 0.0

    # 显示初始温度
    initial_max_temp, initial_temps = get_max_temperature()
    initial_temp_str = format_temperatures(initial_temps)
    if initial_temp_str:
        console.print(f"[cyan]硬件初始温度：{initial_temp_str}[/cyan]\n")

    for idx, mp3_file in enumerate(files_to_process, 1):
        current_max_temp, current_temps = get_max_temperature()
        temp_str = format_temperatures(current_temps)

        progress_line = f"\n[bold blue]总体进度：{idx}/{total}[/bold blue]"
        if temp_str:
            progress_line += f" [cyan]{temp_str}[/cyan]"

        console.print(progress_line)
        console.print(f"[yellow]处理中：{mp3_file.name}[/yellow]")

        # 获取音频时长
        try:
            audio_duration = get_audio_duration(mp3_file)
        except (OSError, RuntimeError):
            audio_duration = 0.0

        # 处理并计时
        start_time = time.time()
        try:
            # faster-whisper 的 transcribe 返回 (segments, info)
            segments, info = model_obj.transcribe(
                str(mp3_file),
                language=language,
                word_timestamps=True,
                vad_filter=True,
                log_progress=True,
            )

            # 将 segments 转换为列表（因为 faster-whisper 返回的是生成器）
            segments_list = list(segments)

            write_srt_only(segments_list, mp3_file.stem + mp3_file.suffix, temp_dir)

            temp_srt = temp_dir / (mp3_file.stem + ".srt")
            final_srt = output_dir / (mp3_file.stem + ".srt")
            shutil.move(str(temp_srt), str(final_srt))

            console.print(f"[green]完成：{mp3_file.name}[/green]")
        except Exception:
            console.print(f"[red]失败：{mp3_file.name}[/red]")
            logger.exception("处理文件失败：%s", mp3_file.name)
            temp_srt = temp_dir / (mp3_file.stem + ".srt")
            if temp_srt.exists():
                temp_srt.unlink()
        process_time = time.time() - start_time

        # 更新统计
        total_audio_duration += audio_duration
        total_process_time += process_time

        # 显示统计
        efficiency = total_audio_duration / total_process_time if total_process_time > 0 else 0
        console.print(
            f"[cyan]已处理：音频 {format_duration(total_audio_duration)}，"
            f"耗时 {format_duration(total_process_time)}，"
            f"效率 {efficiency:.1f}x[/cyan]"
        )

        # 检查温度（最后一个文件不需要等待）
        if idx < total:
            current_max_temp, current_temps = get_max_temperature()
            while current_max_temp >= settings.temp_threshold and current_max_temp > 0:
                temp_str = format_temperatures(current_temps)
                console.print(f"[bold red]⚠️ 检测到高温！{temp_str}[/bold red]")
                console.print(f"[bold red]触发过热保护，暂停计算并等待 {settings.temp_wait_minutes} 分钟...[/bold red]")
                time.sleep(settings.temp_wait_minutes * 60)
                current_max_temp, current_temps = get_max_temperature()

    # 显示结束温度
    final_max_temp, final_temps = get_max_temperature()
    final_temp_str = format_temperatures(final_temps)
    if final_temp_str:
        console.print(f"\n[cyan]硬件结束温度：{final_temp_str}[/cyan]")


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
    output: Path = typer.Option("output", "--output", "-o", help="📁 输出目录（默认为当前目录）"),
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

    output_dir = Path(output).expanduser()
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

    # NVCC 信息
    console.print("\n[bold cyan]=== NVCC 信息 ===[/bold cyan]")
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        try:
            result = subprocess.run([nvcc_path, "--version"], check=False, capture_output=True, text=True, timeout=10)  # noqa: S603
            if result.returncode == 0:
                console.print(result.stdout)
            else:
                console.print("[yellow]nvcc 执行失败[/yellow]")
                console.print(result.stderr)
        except FileNotFoundError:
            console.print("[yellow]未找到 nvcc 命令[/yellow]")
        except OSError as e:
            console.print(f"[red]nvcc 出错: {e}[/red]")
    else:
        console.print("[yellow]未找到 nvcc 命令[/yellow]")

    # Triton 信息
    console.print("\n[bold cyan]=== Triton 信息 ===[/bold cyan]")
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.is_available()); import triton; print('Triton OK')"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            console.print(result.stdout)
        else:
            console.print("[yellow]Triton 检查失败[/yellow]")
            console.print(result.stderr)
    except FileNotFoundError:
        console.print("[yellow]未找到 Python 命令[/yellow]")
    except OSError as e:
        console.print(f"[red]Triton 检查出错: {e}[/red]")

    # 依赖版本
    console.print("\n[bold cyan]=== 依赖版本 ===[/bold cyan]")
    deps_table = Table("包名", "版本", show_lines=False)
    try:
        import faster_whisper

        deps_table.add_row("faster-whisper", faster_whisper.__version__)
    except ImportError:
        deps_table.add_row("faster-whisper", "N/A")
    deps_table.add_row("torch", torch.__version__)
    deps_table.add_row("typer", typer.__version__)
    deps_table.add_row("pydantic", pydantic.__version__)
    deps_table.add_row("pydantic-settings", getattr(pydantic_settings, "__version__", "N/A"))
    console.print(deps_table)


if __name__ == "__main__":
    app()
