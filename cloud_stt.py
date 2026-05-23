import logging
import subprocess
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from config import LOCAL_FILES_TO_UPLOAD, settings
from log_config import setup_logging
from utils import filter_existing_files, get_mp3_files

app = typer.Typer(
    name=settings.app_name,
    help="☁️ 云端 STT 自动化工具 - 将本地音频发送到云服务器处理",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)

console = Console()
setup_logging()
logger = logging.getLogger(f"{settings.app_name}.cloud_stt")


def run_ssh_command_live(command: str) -> int:
    """
    实时透传远端终端输出，不捕获任何结果。
    适用于需要查看动态进度条、复杂终端动效的场景。
    """
    ssh_cmd = ["ssh", "-t", settings.ssh_alias, command]
    console.print(f"[dim]$ {' '.join(ssh_cmd)}[/dim]")

    proc = subprocess.run(
        ssh_cmd,
        stdout=None,
        stderr=None,
        text=True,
        stdin=None,
    )

    return proc.returncode


def run_ssh_command(command: str) -> tuple[int, str, str]:
    ssh_cmd = ["ssh", "-n", "-t", settings.ssh_alias, command]
    console.print(f"[dim]$ {' '.join(ssh_cmd)}[/dim]")

    proc = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        stdin=subprocess.DEVNULL,
    )

    stdout, _ = proc.communicate()

    if stdout:
        console.print(stdout)

    return proc.returncode, stdout, ""


def scp_file(local_path: Path, remote_path: str, upload: bool = True):
    if upload:
        scp_cmd = ["scp", str(local_path), f"{settings.ssh_alias}:{remote_path}"]
    else:
        scp_cmd = ["scp", f"{settings.ssh_alias}:{remote_path}", str(local_path)]
    console.print(f"[dim]$ {' '.join(scp_cmd)}[/dim]")
    result = subprocess.run(scp_cmd, capture_output=True, text=True, check=True)  # noqa: S603
    if result.stdout:
        console.print(result.stdout)
    if result.stderr:
        console.print(f"[yellow]{result.stderr}[/yellow]")


def setup_remote_server(no_uv: bool = False):
    console.print("[bold cyan]正在设置远程服务器...[/bold cyan]")

    commands = [
        f"mkdir -p {settings.remote_work_path} {settings.remote_input_path} {settings.remote_output_path}",
        f"ls -la {settings.remote_work_path}",
    ]

    for cmd in commands:
        code, stdout, stderr = run_ssh_command(cmd)
        if code != 0:
            console.print(f"[yellow]命令执行警告: {cmd}[/yellow]")
            if stderr:
                console.print(stderr)

    for f in LOCAL_FILES_TO_UPLOAD:
        if Path(f).exists():
            console.print(f"上传文件: {f}")
            scp_file(Path(f), str(settings.remote_work_path), upload=True)

    console.print("[cyan]上传远端 CUDA 11 配置文件...[/cyan]")
    cuda11_config = Path(__file__).parent / "pyproject.cuda11.toml"
    if cuda11_config.exists():
        scp_file(cuda11_config, str(settings.remote_work_path / "pyproject.toml"), upload=True)
        console.print("[green]✓ CUDA 11 配置文件已上传[/green]")
    else:
        console.print("[yellow]⚠ pyproject.cuda11.toml 未找到，跳过[/yellow]")

    console.print("[bold cyan]基础文件上传完成[/bold cyan]")

    console.print("[bold cyan]检查并安装 ffmpeg...[/bold cyan]")
    check_ffmpeg_cmd = 'which ffmpeg || echo "not_found"'
    code, stdout, stderr = run_ssh_command(check_ffmpeg_cmd)
    if "not_found" in stdout or code != 0:
        console.print("[yellow]ffmpeg 未找到，正在安装...[/yellow]")
        install_ffmpeg_cmd = "sudo apt-get update && sudo apt-get install -y ffmpeg"
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task("安装 ffmpeg 中...", total=None)
            code = run_ssh_command_live(install_ffmpeg_cmd)
            progress.update(task, completed=True)
        if code != 0:
            console.print(f"[red]ffmpeg 安装失败: {stderr}[/red]")
            raise typer.Exit(1)
        console.print("[green]ffmpeg 安装完成[/green]")

    if no_uv:
        console.print("[bold cyan]使用系统 pip 安装依赖（no-uv 模式）...[/bold cyan]")
        install_cmd = f"cd {settings.remote_work_path} && pip install -e ."
        console.print("[cyan]安装依赖中[/cyan]")
        code = run_ssh_command_live(install_cmd)
        if code != 0:
            console.print(f"[red]依赖安装失败[/red]")
            raise typer.Exit(1)
        console.print("[bold green]系统 pip 依赖安装完成![/bold green]")
    else:
        console.print("[bold cyan]检查并安装 uv...[/bold cyan]")
        check_uv_cmd = 'export PATH="$HOME/.local/bin:$PATH" && which uv || echo "not_found"'
        code, stdout, stderr = run_ssh_command(check_uv_cmd)
        if "not_found" in stdout or code != 0:
            console.print("[yellow]uv 未找到，正在安装...[/yellow]")
            install_uv_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
            ) as progress:
                task = progress.add_task("安装 uv 中...", total=None)
                code = run_ssh_command_live(install_uv_cmd)
                progress.update(task, completed=True)
            if code != 0:
                console.print(f"[red]uv 安装失败: {stderr}[/red]")
                raise typer.Exit(1)
            console.print("[green]uv 安装完成[/green]")

        console.print("[bold cyan]安装项目依赖...[/bold cyan]")
        install_cmd = (
            f'export PATH="$HOME/.local/bin:$PATH" && cd {settings.remote_work_path} && rm -rf uv.lock && uv sync'
        )
        console.print("[cyan]安装依赖中[/cyan]")
        code = run_ssh_command_live(install_cmd)

        if code != 0:
            console.print(f"[red]依赖安装失败[/red]")
            console.print(stderr)
            raise typer.Exit(1)

        console.print("[bold green]远程服务器设置完成![/bold green]")
        dep_list_cmd = f'export PATH="$HOME/.local/bin:$PATH" && cd {settings.remote_work_path} && uv pip list'
        run_ssh_command(dep_list_cmd)


def process_single_file(mp3_file: Path, output_dir: Path, model: str = "small", no_uv: bool = False):
    filename = mp3_file.name
    remote_mp3 = str(settings.remote_input_path / filename)
    remote_srt = str(settings.remote_output_path / f"{mp3_file.stem}.srt")

    python_cmd = "python" if no_uv else "uv run python"

    try:
        console.print(f"[yellow]上传: {filename}[/yellow]")
        scp_file(mp3_file, remote_mp3, upload=True)

        console.print(f"[cyan]处理中: {filename}[/cyan]")
        process_cmd = (
            f'export PATH="$HOME/.local/bin:$PATH" && export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH && '
            f"cd {settings.remote_work_path} && "
            f"{python_cmd} use_whisper.py transcribe "
            f'--model {model} --output "{settings.remote_output_path}" '
            f'"{remote_mp3}"'
        )

        start_time = time.time()
        code = run_ssh_command_live(process_cmd)
        process_time = time.time() - start_time

        if code != 0:
            console.print(f"[red]处理失败: {filename}[/red]")
            return False, process_time

        console.print(f"[green]下载结果: {filename}[/green]")
        scp_file(output_dir / f"{mp3_file.stem}.srt", remote_srt, upload=False)

        cleanup_cmd = f'rm -f "{remote_mp3}" "{remote_srt}"'
        run_ssh_command(cleanup_cmd)

        return True, process_time

    except Exception as e:
        console.print(f"[red]处理出错: {filename} - {e}[/red]")
        return False, 0


@app.command()
def transcribe(
    input_path: str = typer.Argument(..., help="🎧 输入音频文件或文件夹路径"),
    model: str = typer.Option(
        "small",
        "--model",
        "-m",
        help="🧠 Whisper 模型大小",
        case_sensitive=False,
    ),
    output: Path = typer.Option("output", "--output", "-o", help="📁 本地输出目录"),
    no_uv: bool = typer.Option(False, "--no-uv", help="不使用 uv 虚拟环境，直接用系统 pip"),
):
    """
    将音频发送到云服务器进行转录
    """
    input_p = Path(input_path)
    if not input_p.exists():
        console.print(f"[red]路径不存在: {input_p}[/red]")
        raise typer.Exit(1)

    mp3_files = get_mp3_files(input_p)
    if not mp3_files:
        console.print("[red]未找到 MP3 文件[/red]")
        raise typer.Exit(1)

    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True)

    files_to_process = filter_existing_files(mp3_files, output_dir)
    if not files_to_process:
        console.print("[green]所有文件都已处理完成！[/green]")
        return

    console.print(f"[bold]找到 {len(files_to_process)} 个文件待处理[/bold]")
    console.print(f"[dim]SSH主机: {settings.ssh_alias}[/dim]")
    console.print(f"[dim]远程工作目录: {settings.remote_work_path}[/dim]")
    console.print(f"[dim]运行模式: {'系统 pip' if no_uv else 'uv 虚拟环境'}[/dim]\n")

    total_files = len(files_to_process)
    total_process_time = 0.0
    success_count = 0

    env_python = "python" if no_uv else "uv run python"
    env_cmd = (
        f'export PATH="$HOME/.local/bin:$PATH" && export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH && '
        f"cd {settings.remote_work_path} && "
        f"{env_python} use_whisper.py env"
    )

    run_ssh_command(env_cmd)

    console.print("\n[bold]开始处理...[/bold]\n")

    for idx, mp3_file in enumerate(files_to_process, 1):
        console.print(f"[bold blue]进度: {idx}/{total_files}[/bold blue]")

        success, process_time = process_single_file(mp3_file, output_dir, model, no_uv)

        if success:
            success_count += 1
            total_process_time += process_time
            console.print(f"[green]✓ 完成: {mp3_file.name} (耗时 {process_time:.1f}s)[/green]\n")
        else:
            console.print(f"[red]✗ 失败: {mp3_file.name}[/red]\n")

    console.print("\n[bold green]处理总结[/bold green]")
    summary_table = Table("项目", "值")
    summary_table.add_row("总文件数", str(total_files))
    summary_table.add_row("成功", str(success_count))
    summary_table.add_row("失败", str(total_files - success_count))
    summary_table.add_row("总耗时", f"{total_process_time:.1f}s")
    if success_count > 0:
        summary_table.add_row("平均每文件", f"{total_process_time / success_count:.1f}s")
    console.print(summary_table)


@app.command()
def setup(no_uv: bool = typer.Option(False, "--no-uv", help="不使用 uv 虚拟环境，直接用系统 pip")):
    """
    仅设置远程服务器环境
    """
    setup_remote_server(no_uv=no_uv)


app()
