from datetime import datetime

# global constants
RANDOM_STATE = 42

# Pipeline config
ARTIFACT_DIR = "artifact"

# File names
TRAIN_FILE_NAME = "train.parquet"
TEST_FILE_NAME = "test.parquet"
VAL_FILE_NAME = "val.parquet"
MEDMCQA_VAL_FILE_NAME = "medmcqa_val.parquet"

# Logger constants
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_FILE_SIZE = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 3

# Data Ingestion
HF_DATASET_ID = "lavita/AlpaCare-MedInstruct-52k"
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_RAW_DIR = "raw"
DATA_INGESTION_INGESTED_DIR = "data_ingested"
DATA_INGESTION_TESTVAL_SPLIT_RATIO: float = 0.2
