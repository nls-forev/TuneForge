from datasets import Dataset, load_dataset

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.entity.experiment_config import ExperimentConfig
from src.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig(),
        experiment_config: ExperimentConfig | None = None,
    ):
        self.data_ingestion_config = data_ingestion_config
        self.experiment_config = experiment_config or ExperimentConfig.load()

    def download_raw_data(self) -> Dataset:
        try:
            dataset = load_dataset(
                self.experiment_config.dataset.id,
                revision=self.experiment_config.dataset.revision,
                cache_dir=self.data_ingestion_config.data_ingestion_raw_dir,
                split=self.experiment_config.dataset.split,
            )
            if "source_row_id" not in dataset.column_names:
                dataset = dataset.add_column(
                    "source_row_id", [str(i) for i in range(len(dataset))]
                )

            logger.info(
                "Successfully downloaded raw data under %s",
                self.data_ingestion_config.data_ingestion_raw_dir,
            )

            return dataset

        except Exception as e:
            logger.exception(f"Error: {e}")
            raise

    def split_dataset(self, dataset: Dataset) -> DataIngestionArtifact:
        try:
            train_test = dataset.train_test_split(
                test_size=self.experiment_config.dataset.testval_split_ratio,
                seed=self.experiment_config.seed,
            )

            train_ds = train_test["train"]

            test_val = train_test["test"].train_test_split(
                test_size=0.5,
                seed=self.experiment_config.seed,
            )

            test_ds = test_val["train"]
            val_ds = test_val["test"]

            logger.info("Split data into train, test and validation sets.")

            train_ds.to_parquet(
                self.data_ingestion_config.data_ingestion_train_file_name
            )
            test_ds.to_parquet(self.data_ingestion_config.data_ingestion_test_file_name)
            val_ds.to_parquet(self.data_ingestion_config.data_ingestion_val_file_name)

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.data_ingestion_train_file_name,
                test_file_path=self.data_ingestion_config.data_ingestion_test_file_name,
                val_file_path=self.data_ingestion_config.data_ingestion_val_file_name,
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
        dataset = self.download_raw_data()

        logger.info("Splitting dataset...")
        return self.split_dataset(dataset)
