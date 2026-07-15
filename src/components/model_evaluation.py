import pandas as pd

from src.constants import (
    LOAD_IN_4BIT,
    MAX_SEQ_LENGTH,
    MODEL_ID_4BIT,
    MODEL_PATH,
)
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig = ModelEvaluationConfig(),
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_parquet(self.data_transformation_artifact.medmcqa_file_path)
            logger.info("Successfully loaded Medmcqa evaluation dataframe.")
            return df

        except FileNotFoundError:
            logger.exception(
                f"Medmcqa evaluation parquet doesn't exit at path: {self.data_transformation_artifact.medmcqa_file_path}"
            )
            raise

        except Exception as e:
            logger.exception(f"Unexpected error occured: {e}")
            raise

    def load_model(self):
        try:
            import os

            from huggingface_hub import snapshot_download
            from peft import PeftModel
            from unsloth import FastLanguageModel

            if os.path.isfile(os.path.join(MODEL_PATH, "config.json")):
                model_id = MODEL_PATH
            else:
                logger.info("Model not found locally — downloading...")
                os.makedirs(MODEL_PATH, exist_ok=True)
                snapshot_download(
                    repo_id=MODEL_ID_4BIT,
                    ignore_patterns=["*.pt", "*.bin"],
                    local_dir=MODEL_PATH,
                )
                model_id = MODEL_PATH

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=LOAD_IN_4BIT,
                device_map="sequential",
            )

            model = PeftModel.from_pretrained(
                model,
                self.model_trainer_artifact.adapter_path,
            )
            FastLanguageModel.for_inference(model)

            return model, tokenizer

        except Exception as e:
            logger.exception(f"Unexpected error occurred: {e}")
            raise

    def evaluate(self, df: pd.DataFrame, model, tokenizer):
        try:
            import torch
            from tqdm import tqdm

            # Score the first-token logits over the four answer-letter tokens.
            # The SFT model was trained on free-text answers only, so generating
            # and parsing the output emits answer *content* (e.g. "Amniotic...")
            # not the label — logit scoring picks A/B/C/D directly instead.
            letters = ["A", "B", "C", "D"]
            letter_ids = [
                tokenizer.encode(letter, add_special_tokens=False)[0]
                for letter in letters
            ]

            prompts = df["prompt"].tolist()
            ground_truth = df["answer"].tolist()

            correct = 0
            predictions = []

            model.eval()

            for i in tqdm(range(len(prompts)), desc="MedMCQA eval"):
                inputs = tokenizer(
                    prompts[i],
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                ).to(model.device)

                with torch.inference_mode():
                    logits = model(**inputs).logits[0, -1, :]

                letter_logits = logits[letter_ids]
                pred = letters[int(letter_logits.argmax())]

                if i < 5:
                    logger.info(
                        "sample %d — gt=%s pred=%s logits=%s",
                        i,
                        ground_truth[i],
                        pred,
                        letter_logits.tolist(),
                    )

                predictions.append(pred)

                if pred == ground_truth[i]:
                    correct += 1

            accuracy = correct / len(prompts)

            logger.info(
                "MedMCQA Accuracy: %.4f (%d/%d)",
                accuracy,
                correct,
                len(prompts),
            )

            return {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(prompts),
                "predictions": predictions,
            }

        except Exception:
            logger.exception("")
            raise

    def init_evaluation(self) -> ModelEvaluationArtifact:
        logger.info("Initializing model evaluation stage...")

        model, tokenizer = self.load_model()
        df = self.load_data()

        metrics = self.evaluate(df, model, tokenizer)

        logger.info("Completed model evaluation stage.")

        return ModelEvaluationArtifact(metrics=metrics)
