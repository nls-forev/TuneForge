import os

from src.constants import (
    LOG_DIR,
    LOG_FILE,
    MAX_LOG_FILE_SIZE,
    BACKUP_COUNT,
)

from from_root import from_root

from dataclasses import dataclass


@dataclass
class LoggerConfig:
    log_dir_path: str = os.path.join(from_root(), LOG_DIR)
    log_file_path: str = os.path.join(log_dir_path, LOG_FILE)
    max_log_file_size: int = MAX_LOG_FILE_SIZE
    log_backup_count = BACKUP_COUNT
