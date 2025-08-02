# uv add vosk pydub typer rich
import os
import sys
from pathlib import Path
from typing import Optional
import tempfile
import time

import typer
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
import json

# rich 用于美化输出
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import install

install()  # 美化异常 traceback
console = Console()

app = typer.Typer(help="使用本地 Vosk 模型将音频转为文字（支持 SRT 时间轴）")

# =============================================
# 🔧 可配置项（用户可修改）
# =============================================

# 模型文件夹名称（放在程序同级目录或绝对路径）
MODEL_FOLDER = "vosk-model-cn-0.22"  # 推荐使用 0.22 或更大模型

# 临时文件前缀
TEMP_PREFIX = "vosk_transcribe_"

# 采样率
SAMPLE_RATE = 16000

# 标点恢复模型（推荐轻量级）
PUNCTUATOR_MODEL = "cantonese-zh/punc_ctc_zh_base"

# =============================================
# 自动生成临时 WAV 路径
# =============================================


def get_temp_wav_path() -> Path:
    timestamp = int(time.time())
    temp_name = f"{TEMP_PREFIX}{timestamp}.wav"
    temp_path = Path(tempfile.gettempdir()) / temp_name
    return temp_path


TEMP_WAV_PATH = get_temp_wav_path()

# =============================================
# 时间格式化：秒 → SRT 时间
# =============================================


def time_format(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s % 1) * 1000):03}"


# =============================================
# 中文标点恢复（基于 Hugging Face BERT）
# =============================================

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch

    HAS_PUNCTUATOR = True
except ImportError:
    HAS_PUNCTUATOR = False


class ChinesePunctuator:
    def __init__(self, model_name: str = PUNCTUATOR_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label

    def punctuate(self, text: str) -> str:
        words = text.strip().split()
        if not words:
            return text

        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)[0].numpy()

        result = ""
        for i, word in enumerate(words):
            result += word
            label_id = predictions[i + 1]  # 注意：[CLS] 在开头
            label = self.id2label[label_id]
            if label in ["，", "。", "！", "？", "；", "："]:
                result += label
            else:
                result += " "  # 保持词语连接自然
        return result.strip()


# 全局标点恢复器（延迟加载）
_punctuator = None


def restore_punctuation(text: str) -> str:
    global _punctuator
    if not text.strip():
        return text

    try:
        if _punctuator is None:
            console.print("[cyan]📥 正在加载中文标点恢复模型...[/cyan]")
            _punctuator = ChinesePunctuator()
        return _punctuator.punctuate(text)
    except Exception as e:
        console.print(f"[yellow]⚠️ 标点恢复失败，使用原文：{e}[/yellow]")
        return text  # 失败时返回原文


# =============================================
# 主命令
# =============================================


@app.command()
def transcribe(
    audio_path: Path = typer.Argument(..., help="输入音频文件路径"),
    model_path: Optional[Path] = typer.Option(
        MODEL_FOLDER, "--model", "-m", help=f"Vosk 模型路径（默认: {MODEL_FOLDER}）"
    ),
):
    """
    将音频文件转为带标点的文本和 SRT 字幕
    """
    if not audio_path.exists():
        console.print(f"[red]❌ 文件不存在：{audio_path}[/red]")
        raise typer.Exit(1)

    model_path = model_path or Path(MODEL_FOLDER)
    txt_path = audio_path.with_suffix(audio_path.suffix + ".txt")
    srt_path = audio_path.with_suffix(audio_path.suffix + ".srt")

    console.print(f"[bold green]🎙️ 开始处理音频：{audio_path}[/bold green]")
    console.print(f"[cyan]模型路径：{model_path}[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        try:
            # 检查模型
            if not model_path.exists():
                console.print(f"[red]❌ 模型路径不存在：{model_path}[/red]")
                console.print(
                    "[yellow]请下载中文模型：https://alphacephei.com/vosk/models[/yellow]"
                )
                raise typer.Exit(1)

            # Step 1: 加载模型
            progress.add_task(description="加载 Vosk 模型...", total=None)
            model = Model(str(model_path))
            rec = KaldiRecognizer(model, SAMPLE_RATE)

            # Step 2: 转换音频
            progress.remove_task(list(progress._tasks.keys())[-1])
            progress.add_task(description="转换音频格式为 16kHz WAV...", total=None)
            audio = AudioSegment.from_file(str(audio_path))
            audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
            audio.export(str(TEMP_WAV_PATH), format="wav")

            # Step 3: 语音识别
            progress.remove_task(list(progress._tasks.keys())[-1])
            progress.add_task(description="正在识别语音...", total=None)
            results = []
            with open(TEMP_WAV_PATH, "rb") as f:
                f.read(44)  # 跳过 WAV 头
                while True:
                    data = f.read(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        results.append(rec.Result())
                results.append(rec.FinalResult())

            segments = []
            for res in results:
                res_json = json.loads(res)
                if "text" in res_json and res_json["text"].strip():
                    segments.append(res_json)

            if not segments:
                console.print("[yellow]⚠️ 未识别出任何文字[/yellow]")
                raise typer.Exit(0)

            total_duration = len(audio) / 1000.0  # 秒

            # Step 4: 写入 TXT（带标点）
            progress.remove_task(list(progress._tasks.keys())[-1])
            progress.add_task(description="生成带标点文本文件...", total=None)
            with open(txt_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    clean_text = restore_punctuation(seg["text"])
                    f.write(clean_text + "\n")
            console.print(f"[bold]✅ 文本已保存：{txt_path}[/bold]")

            # Step 5: 写入 SRT
            progress.remove_task(list(progress._tasks.keys())[-1])
            progress.add_task(description="生成字幕文件...", total=None)
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments, 1):
                    text = seg["text"]
                    clean_text = restore_punctuation(text)

                    if "result" in seg and seg["result"]:
                        # 精确时间戳
                        words = seg["result"]
                        start_sec = words[0]["start"]
                        end_sec = words[-1]["end"]
                    else:
                        # 兜底：平均分配时间
                        seg_duration = total_duration / len(segments)
                        start_sec = (i - 1) * seg_duration
                        end_sec = i * seg_duration

                    start = time_format(start_sec)
                    end = time_format(end_sec)
                    f.write(f"{i}\n{start} --> {end}\n{clean_text}\n\n")

            console.print(f"[bold]✅ SRT 字幕已保存：{srt_path}[/bold]")

            # 清理临时文件
            if TEMP_WAV_PATH.exists():
                os.unlink(TEMP_WAV_PATH)
                console.print(f"[dim]🗑️ 临时文件已清理：{TEMP_WAV_PATH}[/dim]")

            console.print("[bold green]🎉 转录完成！[/bold green]")

        except Exception as e:
            console.print(f"[red]❌ 处理失败：{str(e)}[/red]")
            if TEMP_WAV_PATH.exists():
                os.unlink(TEMP_WAV_PATH)
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
