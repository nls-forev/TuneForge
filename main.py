import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.components.model_evaluation import ModelEvaluation
from src.constants import WANDB_PROJECT
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.logger import configure_logger, get_logger

configure_logger()
logger = get_logger(__name__)

# Secrets come from the environment (.env locally, container env on AWS). The base
# 4bit model is public, so HF_TOKEN is optional. wandb reads WANDB_API_KEY; the
# .env stores it as WANDB, so mirror it.
if os.environ.get("WANDB") and not os.environ.get("WANDB_API_KEY"):
    os.environ["WANDB_API_KEY"] = os.environ["WANDB"]

METRICS_DIR = "artifact/model_evaluation"
METRICS_PATH = os.path.join(METRICS_DIR, "metrics.json")


def main() -> int:
    data_transformation_artifact = DataTransformationArtifact(
        train_sft_file_path="",
        test_sft_file_path="",
        val_sft_file_path="",
        medmcqa_file_path="artifact/data_transformation/transformed/medmcqa.parquet",
    )
    model_trainer_artifact = ModelTrainerArtifact(
        adapter_path="artifact/model_trainer/adapter",
        train_loss=0.0,
        train_runtime=0.0,
    )

    run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            run = wandb.init(project=WANDB_PROJECT, job_type="evaluation")
            logger.info("wandb run initialized: %s", run.name)
        except Exception:
            logger.exception("wandb init failed — continuing without it.")

    try:
        artifact = ModelEvaluation(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_artifact=model_trainer_artifact,
        ).init_evaluation()

        metrics = artifact.metrics
        os.makedirs(METRICS_DIR, exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote metrics to %s", METRICS_PATH)

        if run is not None:
            import wandb

            wandb.log(
                {
                    "medmcqa/accuracy": metrics["accuracy"],
                    "medmcqa/correct": metrics["correct"],
                    "medmcqa/total": metrics["total"],
                }
            )

        print(artifact)
        return 0

    except Exception:
        logger.exception("Evaluation failed.")
        return 1

    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    sys.exit(main())
