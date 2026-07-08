import logging
import os

from logging.handlers import RotatingFileHandler

from src.entity.config_entity import LoggerConfig


def configure_logger():
    logger = logging.getLogger()

    if logger.handlers:
        return

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(LoggerConfig.log_dir_path, exist_ok=True)

    file_handler = RotatingFileHandler(
        LoggerConfig.log_file_path,
        maxBytes=LoggerConfig.max_log_file_size,
        backupCount=LoggerConfig.log_backup_count,
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def get_logger(name):
    return logging.getLogger(name)
