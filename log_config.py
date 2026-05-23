import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import settings


def setup_logging():
    """
    设置日志系统，支持stream和文件双输出，幂等调用
    """
    log_path = Path(settings.log_dir)
    log_path.mkdir(exist_ok=True)

    logger = logging.getLogger(settings.app_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 避免重复输出

    if logger.handlers:
        return  # 已经设置过，幂等退出

    # StreamHandler (控制台输出)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # FileHandler (文件输出)
    log_file = log_path / f"{settings.app_name}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
