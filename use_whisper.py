# uv add torch torchaudio openai-whisper typer rich
# 模型: https://huggingface.co/openai/whisper-medium
#     mirror: https://hf-mirror.com/openai/whisper-medium
# huggingface 上,一个B是一个Billion
import logging
import os
from pathlib import Path

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

# 配置 logging，包含时间戳
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = typer.Typer(
    name="whisper-transcribe",
    help="🎙️ 使用官方 OpenAI Whisper 模型转录音频，输出带标点的中文 SRT 字幕",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)


def validate_audio_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        logging.error(f"文件不存在：{path}")
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
    output: Path | None = typer.Option(
        None, "--output", "-o", help="📁 输出目录（默认为当前目录）"
    ),
    language: str = typer.Option(
        "zh", "--language", "-l", help="🌐 语言代码（zh=中文）"
    ),
):
    """
    转录音频并生成 SRT、TXT、JSON 等格式
    """
    audio_path = validate_audio_path(audio)

    # 输出目录
    output_dir = output or Path()
    output_dir.mkdir(exist_ok=True)

    # 开始信息（使用 logging 输出带时间戳）
    logging.info(f"开始处理音频：{audio_path.name} | 模型：{model} | 语言：{language}")

    # 加载模型
    try:
        logging.info(f"正在加载模型 '{model}'...")
        model_obj = whisper.load_model(model, download_root=WHISPER_DOWNLOAD_ROOT)
    except Exception:
        logging.exception("加载模型失败：")
        raise typer.Exit(1)

    # 转录
    try:
        logging.info("正在转录音频...")
        result = model_obj.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            verbose=TRANSCRIBE_VERBOSE,  # None 无输出 false 进度条, True 内容
        )
    except Exception:
        logging.exception("转录失败：")
        raise typer.Exit(1)

    # 写入输出
    try:
        logging.info("正在生成输出文件...")
        # 只输出 txt 和 srt
        txt_writer = get_writer("txt", str(output_dir))
        srt_writer = get_writer("srt", str(output_dir))

        # 写入 txt
        txt_writer(
            result,
            audio_path.stem + audio_path.suffix,
            {
                "highlight_words": False,
                "max_line_width": None,
                "max_line_count": None,
            },
        )
        # 写入 srt
        srt_writer(
            result,
            audio_path.stem + audio_path.suffix,
            {
                "highlight_words": False,
                "max_line_width": None,
                "max_line_count": None,
            },
        )
    except Exception:
        logging.exception("写入文件失败：")
        raise typer.Exit(1)

    # 成功提示（使用 logging）
    logging.info("转录完成！输出文件：")
    logging.info(f"✅ TXT: {output_dir}/{audio_path.stem}.txt")
    logging.info(f"✅ SRT: {output_dir}/{audio_path.stem}.srt")


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
