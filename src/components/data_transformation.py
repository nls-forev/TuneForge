import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from src.constants import MAX_SEQ_LENGTH, MODEL_ID
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.logger import get_logger

logger = get_logger(__name__)


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig = DataTransformationConfig(),
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config

    def _to_text(self, tokenizer, example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    def _transform_sft_split(self, src_file: str) -> pd.DataFrame:
        try:
            df = pd.read_parquet(src_file)
            logger.info("Read parquet file with head: ")
            logger.info(df.head())

            norm = (
                df["input"]
                .str.strip()
                .str.strip('"')
                .str.strip()
                .str.lower()
                .str.replace(" ", "", regex=False)
            )
            has_input = norm != "<noinput>"

            df["prompt"] = np.where(
                has_input,
                df["instruction"].str.strip() + "\n\n" + df["input"].str.strip(),
                df["instruction"].str.strip(),
            )

            logger.info("prompt feature was successfully created.")
            return df

        except Exception as e:
            logger.exception(f"Error: {e}")
            raise

    def to_text(self, df: pd.DataFrame, dest_path: str, tokenizer) -> None:
        try:
            ds = Dataset.from_pandas(df, preserve_index=False)
            ds = ds.map(lambda ex: self._to_text(tokenizer, ex))
            ds = ds.filter(
                lambda x: len(tokenizer(x["text"])["input_ids"]) <= MAX_SEQ_LENGTH
            )
            ds = ds.select_columns(["text"])

            os.makedirs(
                self.data_transformation_config.data_transformation_transformed_dir,
                exist_ok=True,
            )

            ds.to_parquet(dest_path)
            logger.info(f"Saved file to path: {dest_path}")

        except Exception as e:
            logger.exception(f"Error: {e}")
            raise

    def _transform_medmcqa(self, src_file, dest_path, tokenizer):
        df = pd.read_parquet(src_file)
        df = df[df["choice_type"] == "single"]
        letters = ["A", "B", "C", "D"]

        def build(ex):
            q = (
                f"{ex['question']}\n"
                f"A. {ex['opa']}\nB. {ex['opb']}\n"
                f"C. {ex['opc']}\nD. {ex['opd']}\nAnswer:"
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return {
                "prompt": prompt,
                "answer": letters[ex["cop"]],
            }

        ds = Dataset.from_pandas(df, preserve_index=False).map(build)
        ds.select_columns(["prompt", "answer"]).to_parquet(dest_path)

    def preprocess_text(self, tokenizer):
        try:
            for src_file, dest_path in [
                (
                    self.data_ingestion_artifact.train_file_path,
                    self.data_transformation_config.data_transformation_train_file_name,
                ),
                (
                    self.data_ingestion_artifact.test_file_path,
                    self.data_transformation_config.data_transformation_test_file_name,
                ),
                (
                    self.data_ingestion_artifact.val_file_path,
                    self.data_transformation_config.data_transformation_val_file_name,
                ),
            ]:
                df = self._transform_sft_split(src_file)
                self.to_text(df, dest_path, tokenizer)

            logger.info(
                f"All transformed splits are saved under: {self.data_transformation_config.data_transformation_transformed_dir}"
            )

        except Exception as e:
            logger.exception(f"Error: {e}")
            raise

    def init_data_transformation(self) -> DataTransformationArtifact:
        logger.info("Initializing data transformation stage...")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        logger.info("Started Preprocessing stage...")
        self.preprocess_text(tokenizer)
        logger.info("Completed text preprocessing stage.")

        logger.info("Started medmcqa preprocessing...")
        self._transform_medmcqa(
            self.data_ingestion_artifact.medmcqa_file_path,
            self.data_transformation_config.data_transformation_medmcqa_file_name,
            tokenizer=tokenizer,
        )
        logger.info("Completed medmcqa preprocessing stage.")

        logger.info("Completed data transformation stage.")
        return DataTransformationArtifact(
            self.data_transformation_config.data_transformation_train_file_name,
            self.data_transformation_config.data_transformation_test_file_name,
            self.data_transformation_config.data_transformation_val_file_name,
            medmcqa_file_path=self.data_transformation_config.data_transformation_medmcqa_file_name,
        )
