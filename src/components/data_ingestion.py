from typing import Tuple

from datasets import load_dataset
from datasets import Dataset

from src.logger import get_logger
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants import RANDOM_STATE


logger = get_logger(__name__)


class DataIngestion:
    def __init__(
        self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        self.data_ingestion_config = data_ingestion_config

    def download_raw_data(self) -> Tuple[Dataset, Dataset]:
        try:
            dataset = load_dataset(
                self.data_ingestion_config.data_ingestion_hf_dataset_id,
                cache_dir=self.data_ingestion_config.data_ingestion_raw_dir,
                split="train",
            )

            logger.info(
                f"Successfully downloaded raw data under {self.data_ingestion_config.data_ingestion_raw_dir}"
            )

            bench = load_dataset(
                "openlifescienceai/medmcqa",
                cache_dir=self.data_ingestion_config.data_ingestion_raw_dir,
                split="validation",
            )

            logger.info(
                f"Successfully downloaded bench data under {self.data_ingestion_config.data_ingestion_raw_dir}"
            )

            return dataset, bench

        except Exception as e:
            logger.exception(f"Error: {e}")
            raise

    def split_dataset(self, dataset: Dataset, bench: Dataset) -> DataIngestionArtifact:
        try:
            train_test = dataset.train_test_split(
                test_size=self.data_ingestion_config.data_ingestion_testval_split_ratio,
                seed=RANDOM_STATE,
            )

            train_ds = train_test["train"]

            test_val = train_test["test"].train_test_split(
                test_size=0.5,
                seed=RANDOM_STATE,
            )

            test_ds = test_val["train"]
            val_ds = test_val["test"]

            logger.info("Split data into train, test and validation sets.")

            train_ds.to_parquet(
                self.data_ingestion_config.data_ingestion_train_file_name
            )
            test_ds.to_parquet(self.data_ingestion_config.data_ingestion_test_file_name)
            val_ds.to_parquet(self.data_ingestion_config.data_ingestion_val_file_name)
            bench.to_parquet(
                self.data_ingestion_config.data_ingestion_medmcqa_val_file_name
            )

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.data_ingestion_train_file_name,
                test_file_path=self.data_ingestion_config.data_ingestion_test_file_name,
                val_file_path=self.data_ingestion_config.data_ingestion_val_file_name,
                medmcqa_file_path=self.data_ingestion_config.data_ingestion_medmcqa_val_file_name,
            )

            logger.info("Saved all splits under: ")
            logger.info(data_ingestion_artifact)

            return data_ingestion_artifact

        except Exception as e:
            logger.exception(f"Error: {e}")
            raise

    def init_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Initiating Data ingestion...")

        logger.info("Downloading datasets...")
        dataset, bench = self.download_raw_data()

        logger.info("Splitting dataset...")
        return self.split_dataset(dataset, bench)
