# uv add torch torchaudio openai-whisper typer rich
# 模型: https://huggingface.co/openai/whisper-medium
#     mirror: https://hf-mirror.com/openai/whisper-medium
# huggingface 上,一个B是一个Billion
import logging
import os
from pathlib import Path

import torch
import typer
import whisper
from rich.console import Console
from rich.table import Table
from whisper.utils import get_writer

# 设置 Hugging Face 镜像与缓存路径
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/.cache"

WHISPER_DOWNLOAD_ROOT = "/data/.cache/whisper"  # default is ~/.cache/whisper
TRANSCRIBE_VERBOSE = False  # None 无输出 false 进度条, True 内容
# 默认模型
MODEL_SIZE = "small"

# ======= #

# 初始化 rich（仅用于 info 命令）
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


def validate_audio_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        logger.error("文件不存在：%s", path)
        raise typer.Exit(1)
    return path


@app.command()
def transcribe(
    audio: str = typer.Argument(..., help="🎧 输入音频文件路径"),
    model: str = typer.Option(
        MODEL_SIZE,
        "--model",
        "-m",
        help="🧠 Whisper 模型大小",
        case_sensitive=False,
        # ✅ 使用 Choice 实现自动补全和校验
        callback=lambda ctx, param, value: value,
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
    audio_path = validate_audio_path(audio)

    # 输出目录
    output_dir = output or Path()
    output_dir.mkdir(exist_ok=True)

    logger.info("开始处理音频：%s | 模型：%s | 语言：%s", audio_path.name, model, language)

    try:
        logger.info("正在加载模型 '%s'...", model)
        model_obj = whisper.load_model(model, download_root=WHISPER_DOWNLOAD_ROOT)

        device_type = model_obj.device.type
        if device_type == "cuda":
            device_name = torch.cuda.get_device_name(model_obj.device)
            device_info = f"GPU: {device_name}"
        elif device_type == "mps":
            device_info = "Apple Silicon MPS"
        else:
            device_info = "CPU"
        logger.info("计算设备：%s", device_info)
    except Exception as err:
        logger.exception("加载模型失败：")
        raise typer.Exit(1) from err

    try:
        logger.info("正在转录音频...")
        result = model_obj.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            verbose=TRANSCRIBE_VERBOSE,
        )
    except Exception as err:
        logger.exception("转录失败：")
        raise typer.Exit(1) from err

    try:
        logger.info("正在生成输出文件...")
        srt_writer = get_writer("srt", str(output_dir))
        srt_writer(
            result,
            audio_path.stem + audio_path.suffix,
            {
                "highlight_words": False,
                "max_line_width": None,
                "max_line_count": None,
            },
        )
    except Exception as err:
        logger.exception("写入文件失败：")
        raise typer.Exit(1) from err

    logger.info("转录完成！输出文件：")
    logger.info("✅ SRT: %s/%s.srt", output_dir, audio_path.stem)


@app.command()
def info():
    """📄 显示当前配置信息"""
    info_table = Table("配置项", "值", title="🔧 当前配置", show_lines=True)
    info_table.add_row("HF 镜像", os.environ["HF_ENDPOINT"])
    info_table.add_row("缓存目录", os.environ["HF_HOME"])
    info_table.add_row("默认模型", MODEL_SIZE)
    console.print(info_table)


if __name__ == "__main__":
    app()
