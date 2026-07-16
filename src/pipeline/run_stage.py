import argparse

from dotenv import load_dotenv

# Load local .env before any component import triggers a HF download. On AWS no
# .env exists and env comes from the container/SSM, so this is a harmless no-op
# (override=False keeps real env vars winning).
load_dotenv()

# Components are imported lazily inside each runner so a stage only needs its
# own dependency group (e.g. `judge` doesn't pull `datasets`/unsloth from the
# ingest/train components).

from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from src.logger import get_logger

logger = get_logger(__name__)


def ingestion_artifact() -> DataIngestionArtifact:
    config = DataIngestionConfig()
    return DataIngestionArtifact(
        config.data_ingestion_train_file_name,
        config.data_ingestion_test_file_name,
        config.data_ingestion_val_file_name,
    )


def transformation_artifact() -> DataTransformationArtifact:
    config = DataTransformationConfig()
    return DataTransformationArtifact(
        config.data_transformation_train_file_name,
        config.data_transformation_test_file_name,
        config.data_transformation_val_file_name,
    )


def run_ingest():
    from src.components.data_ingestion import DataIngestion

    DataIngestion().init_data_ingestion()


def run_transform():
    from src.components.data_transformation import DataTransformation

    DataTransformation(
        data_ingestion_artifact=ingestion_artifact(),
    ).init_data_transformation()


def run_trainer():
    from src.components.model_trainer import ModelTrainer

    ModelTrainer(
        data_transformation_artifact=transformation_artifact(),
    ).init_model_trainer()


def trainer_artifact() -> ModelTrainerArtifact:
    # Eval only reads the adapter; loss/runtime come from training and are
    # unused here, so they are left at 0.0.
    config = ModelTrainerConfig()
    return ModelTrainerArtifact(
        adapter_path=config.model_trainer_adapter_dir,
        train_loss=0.0,
        train_runtime=0.0,
    )


def run_generate():
    # Phase A of the LLM-as-judge eval (GPU): base + fine-tuned free-text
    # responses. Heavy deps (unsloth/torch) load inside the method.
    from src.components.evaluation.generate_responses import GenerateResponses

    GenerateResponses(
        model_trainer_artifact=trainer_artifact(),
    ).run()


def run_judge():
    # Phase B of the LLM-as-judge eval (local): DeepSeek 1-5 + win-rate +
    # ROUGE-L + BERTScore. Heavy deps (openai/rouge/bert) load inside the method.
    from src.components.evaluation.judge import Judge

    Judge().run()


STAGES = {
    "ingest": run_ingest,
    "transform": run_transform,
    "train": run_trainer,
    "generate": run_generate,
    "judge": run_judge,
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
