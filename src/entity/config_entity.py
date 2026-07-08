import os

from src.constants import (
    LOG_DIR,
    LOG_FILE,
    MAX_LOG_FILE_SIZE,
    BACKUP_COUNT,
    ARTIFACT_DIR,
    HF_DATASET_ID,
    DATA_INGESTION_DIR,
    DATA_INGESTION_RAW_DIR,
    DATA_INGESTION_INGESTED_DIR,
    DATA_INGESTION_TESTVAL_SPLIT_RATIO,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    VAL_FILE_NAME,
    MEDMCQA_VAL_FILE_NAME,
)

from from_root import from_root

from dataclasses import dataclass


@dataclass
class LoggerConfig:
    log_dir_path: str = os.path.join(from_root(), LOG_DIR)
    log_file_path: str = os.path.join(log_dir_path, LOG_FILE)
    max_log_file_size: int = MAX_LOG_FILE_SIZE
    log_backup_count = BACKUP_COUNT


@dataclass
class DataIngestionConfig:
    data_ingestion_hf_dataset_id: str = HF_DATASET_ID
    data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR)
    data_ingestion_raw_dir: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_RAW_DIR
    )
    data_ingestion_ingested_dir: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_INGESTED_DIR
    )
    data_ingestion_testval_split_ratio: float = DATA_INGESTION_TESTVAL_SPLIT_RATIO
    data_ingestion_train_file_name: str = os.path.join(
        data_ingestion_ingested_dir, TRAIN_FILE_NAME
    )
    data_ingestion_test_file_name: str = os.path.join(
        data_ingestion_ingested_dir, TEST_FILE_NAME
    )
    data_ingestion_val_file_name: str = os.path.join(
        data_ingestion_ingested_dir, VAL_FILE_NAME
    )
    data_ingestion_medmcqa_val_file_name: str = os.path.join(
        data_ingestion_ingested_dir, MEDMCQA_VAL_FILE_NAME
    )
