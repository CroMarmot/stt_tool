# TODO 完全移除，阿里云有 https://ecs.console.aliyun.com/keyPair 管理，所以这里就是配置 ~/.ssh/config 就完了，不用关心登录和复制，但是可以测试
import os
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt

from config import settings

app = typer.Typer(
    name="setup-ssh",
    help="🔑 SSH 配置助手 - 配置 SSH 密钥和主机别名",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)

console = Console()


def get_ssh_dir() -> Path:
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    return ssh_dir


def find_key(ssh_dir: Path) -> Path:
    """查找现有的 SSH 密钥，允许用户选择或指定路径"""
    key_types = ["id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"]
    found_keys = []

    for key_name in key_types:
        private_key = ssh_dir / key_name
        public_key = ssh_dir / f"{key_name}.pub"
        if private_key.exists() and public_key.exists():
            found_keys.append((key_name, private_key))

    if found_keys:
        console.print("[bold cyan]找到以下 SSH 密钥:[/bold cyan]")
        for i, (key_name, _) in enumerate(found_keys, 1):
            console.print(f"  {i}. {key_name}")
        console.print("  0. 指定其他密钥路径")

        choice = Prompt.ask("请选择密钥", default="1")

        if choice == "0":
            key_path = Prompt.ask("请输入私钥路径")
            private_key = Path(key_path).expanduser()
            public_key = private_key.parent / f"{private_key.name}.pub"
            if not private_key.exists():
                console.print(f"[red]私钥文件不存在: {private_key}[/red]")
                raise typer.Exit(1)
            if not public_key.exists():
                console.print(f"[red]公钥文件不存在: {public_key}[/red]")
                raise typer.Exit(1)
            console.print(f"[green]已选择密钥: {private_key.name}[/green]")
            return private_key
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(found_keys):
                    console.print(f"[green]已选择密钥: {found_keys[idx][0]}[/green]")
                    return found_keys[idx][1]
            except ValueError:
                pass
        console.print("[red]无效的选择[/red]")
        raise typer.Exit(1)
    else:
        console.print("[red]未找到 SSH 密钥![/red]")
        console.print("[yellow]请先生成一个 SSH 密钥，例如:[/yellow]")
        console.print(f'  ssh-keygen -t ed25519 -C "{os.getlogin()}@{os.uname().nodename}"')
        raise typer.Exit(1)


def copy_key_to_server(public_key: Path, host: str, user: str, port: int):
    """将公钥复制到服务器"""
    console.print(f"[cyan]正在将公钥复制到 {user}@{host}:{port}...[/cyan]")
    console.print("[dim]这一步需要输入服务器密码[/dim]")

    try:
        ssh_opts = [
            "-p",
            str(port),
        ]

        console.print("[dim]正在检查并修复远程服务器 SSH 目录权限...[/dim]")
        subprocess.run(  # noqa: S603
            ["ssh"] + ssh_opts + [f"{user}@{host}", "mkdir -p ~/.ssh && chmod 700 ~/.ssh"],
            check=True,
        )

        ssh_cmd = [
            "ssh-copy-id",
            "-p",
            str(port),
            "-i",
            str(public_key),
            f"{user}@{host}",
        ]
        subprocess.run(ssh_cmd, check=True)  # noqa: S603
        console.print("[green]公钥已成功添加到服务器![/green]")
    except subprocess.CalledProcessError:
        console.print("[red]复制公钥失败，请手动执行:[/red]")
        console.print(f"  ssh-copy-id -p {port} -i {public_key} {user}@{host}")
        raise typer.Exit(1)


def add_host_to_config(ssh_dir: Path, alias: str, host: str, user: str, port: int, identity_file: Path):  # noqa: PLR0913
    """添加主机配置到 ~/.ssh/config"""
    config_file = ssh_dir / "config"

    host_entry = f"""
Host {alias}
    HostName {host}
    User {user}
    Port {port}
    IdentityFile {identity_file}
"""

    if config_file.exists():
        content = config_file.read_text()
        if f"Host {alias}" in content:
            console.print(f"[yellow]配置中已存在 {alias}，正在更新...[/yellow]")
            lines = content.split("\n")
            new_lines = []
            skip = False
            for line in lines:
                if line.strip().startswith(f"Host {alias}"):
                    skip = True
                elif skip and line.strip().startswith("Host "):
                    skip = False
                    new_lines.append(line)
                elif not skip:
                    new_lines.append(line)
            content = "\n".join(new_lines)
        content += host_entry
    else:
        content = host_entry

    config_file.write_text(content)
    config_file.chmod(0o600)
    console.print(f"[green]已添加主机别名 '{alias}' 到 ~/.ssh/config[/green]")


def test_connection(alias: str):
    """测试 SSH 连接"""
    console.print(f"[cyan]正在测试连接到 {alias}...[/cyan]")
    try:
        subprocess.run(["ssh", alias, "echo '连接成功!'"], check=True)  # noqa: S603, S607
        console.print("[green]✓ SSH 连接测试成功![/green]")
    except subprocess.CalledProcessError:
        console.print("[red]✗ SSH 连接测试失败[/red]")
        raise typer.Exit(1)


@app.command()
def setup():
    """配置 SSH 密钥和主机别名"""
    console.print("[bold]=== SSH 配置助手 ===[/bold]\n")

    ssh_dir = get_ssh_dir()

    private_key = find_key(ssh_dir)
    public_key = private_key.parent / f"{private_key.name}.pub"

    host = Prompt.ask("服务器地址 (HostName)")
    user = Prompt.ask("用户名 (User)", default="root")
    port = int(Prompt.ask("端口 (Port)", default="22"))
    alias = settings.ssh_alias

    copy_key_to_server(public_key, host, user, port)
    add_host_to_config(ssh_dir, alias, host, user, port, private_key)
    test_connection(alias)

    console.print("\n[bold green]配置完成![/bold green]")
    console.print(f"现在你可以直接使用别名 '{alias}' 连接到服务器")
    console.print("\n接下来请:")
    console.print(f"   uv run python cloud_stt.py setup")
    console.print(f"   uv run python cloud_stt.py transcribe ./audio")


app()
