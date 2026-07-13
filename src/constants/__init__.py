from datetime import datetime

# global constants
RANDOM_STATE = 42
HF_DATASET_ID = "lavita/AlpaCare-MedInstruct-52k"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 2048

# Pipeline config
ARTIFACT_DIR = "artifact"

# File names
TRAIN_FILE_NAME = "train.parquet"
TEST_FILE_NAME = "test.parquet"
VAL_FILE_NAME = "val.parquet"
MEDMCQA_VAL_FILE_NAME = "medmcqa_val.parquet"

# Confif file name
HYPERPARAMS_FILE_NAME: str = "config/hyperparams.yaml"

TRAIN_SFT_FILE_NAME = "train_sft.parquet"
TEST_SFT_FILE_NAME = "test_sft.parquet"
VAL_SFT_FILE_NAME = "val_sft.parquet"
MEDMCQA_FILE_NAME = "medmcqa.parquet"

# Logger constants
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_FILE_SIZE = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 3

# Data Ingestion
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_RAW_DIR = "raw"
DATA_INGESTION_INGESTED_DIR = "data_ingested"
DATA_INGESTION_TESTVAL_SPLIT_RATIO: float = 0.2

# DATA TRANSFORMATION
DATA_TRANSFORMATION_DIR = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR = "transformed"

# MODEL TRAINER
MODEL_TRAINER_DIR = "model_trainer"
MODEL_TRAINER_ADAPTER_DIR = "adapter"
MODEL_TRAINER_CHECKPOINT_DIR = "checkpoints"

LOAD_IN_4BIT = True
USE_GRADIENT_CHECKPOINTING = "unsloth"
LORA_BIAS = "none"

# Experiment tracking
WANDB_PROJECT = "tuneforge-qlora-medical"
