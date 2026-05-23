import glob
import logging
import subprocess
from pathlib import Path

import typer
from rich.console import Console

from config import settings

logger = logging.getLogger(f"{settings.app_name}.utils")
console = Console()

# 定义需要重点关注的传感器关键字（不区分大小写）
CRITICAL_SENSORS = ["cpu", "package", "core", "pch", "nvme", "composite", "gpu"]


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


def get_all_temperatures() -> dict[str, float]:
    """遍历 hwmon 接口获取所有关键硬件的温度"""
    temps = {}

    # 尝试通过 nvidia-smi 获取 GPU 温度
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            gpu_temps = result.stdout.strip().split("\n")
            for idx, temp in enumerate(gpu_temps):
                if temp.strip():
                    temps[f"GPU-{idx}"] = float(temp.strip())
    except Exception:
        console.print("[yellow]⚠ 无法通过 nvidia-smi 读取 GPU 温度，继续尝试 hwmon...[/yellow]")
        logger.warning("无法通过 nvidia-smi 读取 GPU 温度")

    # 匹配 /sys/class/hwmon/hwmon*/temp*_input
    temp_files = glob.glob("/sys/class/hwmon/hwmon*/temp*_input")

    for path in temp_files:
        try:
            path_obj = Path(path)
            with path_obj.open("r") as f:
                raw_temp = int(f.read().strip())
                temp_c = raw_temp / 1000.0  # 原始值为毫摄氏度

            # 尝试读取该传感器的名称标签
            label_path = path_obj.with_suffix("").with_name(path_obj.stem.replace("_input", "_label"))
            sensor_name = "Unknown"
            if label_path.exists():
                with label_path.open("r") as f:
                    sensor_name = f.read().strip()

            # 尝试读取设备名称
            name_path = path_obj.parent / "name"
            device_name = ""
            if name_path.exists():
                with name_path.open("r") as f:
                    device_name = f.read().strip()
                    sensor_name = f"{device_name}:{sensor_name}"

            # 过滤出我们关心的核心硬件温度
            name_lower = sensor_name.lower()
            if any(key in name_lower for key in CRITICAL_SENSORS):
                if sensor_name not in temps:
                    temps[sensor_name] = temp_c
        except Exception:
            continue

    return temps


def get_max_temperature() -> tuple[float, dict[str, float]]:
    """获取最高温度和所有温度信息（失败返回 0 和空 dict）"""
    temps = get_all_temperatures()
    if not temps:
        console.print("[yellow]⚠ 未检测到兼容的硬件温度传感器[/yellow]")
        logger.warning("未检测到兼容的硬件温度传感器")
        return 0.0, temps
    max_temp = max(temps.values())
    # 输出每个硬件温度
    for name, temp in temps.items():
        logger.debug("%s: %.1f°C", name, temp)
    logger.info("硬件温度：%s (最高: %.1f°C)", format_temperatures(temps), max_temp)
    return max_temp, temps


def format_temperatures(temps: dict[str, float]) -> str:
    """格式化温度信息用于显示"""
    if not temps:
        return ""
    return " | ".join([f"{k}: {v:.1f}°C" for k, v in temps.items()])
