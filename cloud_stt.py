import logging
import re
import subprocess
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from config import LOCAL_FILES_TO_UPLOAD, settings
from utils import filter_existing_files, get_mp3_files

app = typer.Typer(
    name=settings.app_name,
    help="☁️ 云端 STT 自动化工具 - 将本地音频发送到云服务器处理",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(settings.app_name)


def run_ssh_command_live(command: str) -> int:
    """
    实时透传远端终端输出，不捕获任何结果。
    适用于需要查看动态进度条、复杂终端动效的场景。
    """
    # 使用 -t 强制分配伪终端，保证远端的 rich 进度条、颜色和清屏动效能正常渲染
    ssh_cmd = ["ssh", "-t", settings.ssh_alias, command]
    console.print(f"[dim]$ {' '.join(ssh_cmd)}[/dim]")

    # 核心修改：将 stdout 和 stderr 都设为 None
    # 这会让子进程（ssh）直接继承当前 Python 程序的终端输入输出
    proc = subprocess.run(
        ssh_cmd,
        stdout=None,  # 直接输出到本地终端，不经过管道
        stderr=None,  # 直接输出到本地终端，不经过管道
        text=True,
        stdin=None,  # 允许本地键盘输入直接传给远端（如果需要交互的话）
    )

    # 直接返回远端命令的退出码
    return proc.returncode


def run_ssh_command(command: str) -> tuple[int, str, str]:
    # ssh 命令添加 -n 参数：这个参数告诉 ssh 不要从标准输入读取任何内容。这是解决此类后台任务挂起问题的标准做法。
    # Python 代码中添加 stdin=subprocess.DEVNULL：这显式地将子进程的 stdin 重定向到一个空设备（类似 Unix/Linux 中的 /dev/null）。这确保了 ssh 进程的 stdin 是一个已知关闭的流，从而避免其因等待输入而挂起。
    # 使用 -t 分配伪终端，保证远程的 rich 进度条和颜色能正常显示
    ssh_cmd = ["ssh", "-n", "-t", settings.ssh_alias, command]
    console.print(f"[dim]$ {' '.join(ssh_cmd)}[/dim]")

    # 改用 Popen 启动进程
    proc = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # 关键修改：直接丢弃 stderr，不捕获也不打印
        text=True,
        stdin=subprocess.DEVNULL,  # 依然保持 stdin 关闭，防止挂起
    )

    # 使用 communicate() 安全地读取 stdout 并等待进程结束
    # 因为 stderr 被丢弃了，这里只需要接收 stdout
    stdout, _ = proc.communicate()

    if stdout:
        console.print(stdout)

    # 返回空字符串作为 stderr，保持函数接口不变
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


def setup_remote_server():
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
    install_cmd = f'export PATH="$HOME/.local/bin:$PATH" && cd {settings.remote_work_path} && rm -rf uv.lock && uv sync'
    console.print("[cyan]安装依赖中[/cyan]")
    code = run_ssh_command_live(install_cmd)

    if code != 0:
        console.print(f"[red]依赖安装失败[/red]")
        console.print(stderr)
        raise typer.Exit(1)

    console.print("[bold green]远程服务器设置完成![/bold green]")
    dep_list_cmd = f'export PATH="$HOME/.local/bin:$PATH" && cd {settings.remote_work_path} && uv pip list'
    run_ssh_command(dep_list_cmd)


def process_single_file(mp3_file: Path, output_dir: Path, model: str = "small"):
    filename = mp3_file.name
    remote_mp3 = str(settings.remote_input_path / filename)
    remote_srt = str(settings.remote_output_path / f"{mp3_file.stem}.srt")

    try:
        console.print(f"[yellow]上传: {filename}[/yellow]")
        scp_file(mp3_file, remote_mp3, upload=True)

        console.print(f"[cyan]处理中: {filename}[/cyan]")
        process_cmd = (
            f'export PATH="$HOME/.local/bin:$PATH" && export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH && '
            f"cd {settings.remote_work_path} && "
            f"uv run python use_whisper.py transcribe "
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
    console.print(f"[dim]远程工作目录: {settings.remote_work_path}[/dim]\n")

    total_files = len(files_to_process)
    total_process_time = 0.0
    success_count = 0

    env_cmd = (
        f'export PATH="$HOME/.local/bin:$PATH" && export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH && '
        f"cd {settings.remote_work_path} && "
        f"uv run python use_whisper.py env"
    )

    run_ssh_command(env_cmd)

    console.print("\n[bold]开始处理...[/bold]\n")

    for idx, mp3_file in enumerate(files_to_process, 1):
        console.print(f"[bold blue]进度: {idx}/{total_files}[/bold blue]")

        success, process_time = process_single_file(mp3_file, output_dir, model)

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
def setup():
    """
    仅设置远程服务器环境
    """
    setup_remote_server()


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
app()
