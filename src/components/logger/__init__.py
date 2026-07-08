import logging
import os

from logging.handlers import RotatingFileHandler

from entity.config_entity import LoggerConfig


def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname) - %(message)s",
    )

    # Make sure logs dir exist
    os.makedirs(LoggerConfig.log_dir_path, exist_ok=True)

    fileHander = RotatingFileHandler(
        LoggerConfig.log_file_path,
        maxBytes=LoggerConfig.max_log_file_size,
        backupCount=LoggerConfig.log_backup_count,
    )
    fileHander.setFormatter(formatter)
    fileHander.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(logging.DEBUG)

    logger.addHandler(fileHander)
    logger.addHandler(consoleHandler)


def get_logger(name):
    return logging.getLogger(name)
