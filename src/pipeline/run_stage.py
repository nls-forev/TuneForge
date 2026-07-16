import argparse

from dotenv import load_dotenv

# Load local .env before any component import triggers a HF download. On AWS no
# .env exists and env comes from the container/SSM, so this is a harmless no-op
# (override=False keeps real env vars winning).
load_dotenv()

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

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


def run_trainer():
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


def run_evaluate():
    ModelEvaluation(
        data_transformation_artifact=transformation_artifact(),
        model_trainer_artifact=trainer_artifact(),
    ).init_evaluation()


def run_generate():
    # Phase A of the LLM-as-judge eval (GPU): generate free-text responses.
    # Imported lazily so the gpu group is only needed when this stage runs.
    from src.judge.generate_responses import main as generate_main

    rc = generate_main()
    if rc:
        raise SystemExit(rc)


def run_judge():
    # Phase B of the LLM-as-judge eval (local): DeepSeek + ROUGE-L + BERTScore.
    # Imported lazily so the judge group is only needed when this stage runs.
    from src.judge.judge_responses import main as judge_main

    rc = judge_main()
    if rc:
        raise SystemExit(rc)


STAGES = {
    "ingest": run_ingest,
    "transform": run_transform,
    "train": run_trainer,
    "evaluate": run_evaluate,
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
