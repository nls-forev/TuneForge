import argparse

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)

from src.logger import get_logger

logger = get_logger(__name__)


def ingestion_artifact() -> DataIngestionArtifact:
    config = DataIngestionConfig()
    return DataIngestionArtifact(
        config.data_ingestion_train_file_name,
        config.data_ingestion_test_file_name,
        config.data_ingestion_val_file_name,
        config.data_ingestion_medmcqa_val_file_name,
    )


def transformation_artifact() -> DataTransformationArtifact:
    config = DataTransformationConfig()
    return DataTransformationArtifact(
        config.data_transformation_train_file_name,
        config.data_transformation_test_file_name,
        config.data_transformation_val_file_name,
        config.data_transformation_medmcqa_file_name,
    )


def run_ingest():
    DataIngestion().init_data_ingestion()


def run_transform():
    DataTransformation(
        data_ingestion_artifact=ingestion_artifact(),
    ).init_data_transformation()


STAGES = {
    "ingest": run_ingest,
    "transform": run_transform,
}


def main():
    parser = argparse.ArgumentParser(description="Run a single pipeline stage.")
    parser.add_argument("stage", choices=STAGES.keys(), help="Pipeline stage to run")
    args = parser.parse_args()

    logger.info(f"Running pipeline stage: {args.stage}")
    STAGES[args.stage]()
    logger.info(f"Finished pipeline stage: {args.stage}")


if __name__ == "__main__":
    main()
