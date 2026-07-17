"""Eval phase A (GPU): generate base + fine-tuned responses -> responses.parquet."""

import os

import numpy as np
import pandas as pd

from src.constants import (
    JUDGE_MAX_NEW_TOKENS,
    JUDGE_N_SAMPLES,
    LOAD_IN_4BIT,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    MODEL_PATH,
    RANDOM_STATE,
)
from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.logger import get_logger

logger = get_logger(__name__)


class GenerateResponses:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig = ModelEvaluationConfig(),
    ):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config

    def load_base_model(self):
        # MODEL_ID matches training so LoRA deltas apply to the same frozen
        # weights; local MODEL_PATH snapshot preferred when present.
        try:
            from unsloth import FastLanguageModel

            model_id = (
                MODEL_PATH
                if os.path.isfile(os.path.join(MODEL_PATH, "config.json"))
                else MODEL_ID
            )
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=LOAD_IN_4BIT,
                device_map="sequential",
            )
            return model, tokenizer

        except Exception as e:
            logger.exception(f"Unexpected error occurred: {e}")
            raise

    def _build_prompts(self, df: pd.DataFrame) -> pd.DataFrame:
        # Same prompt construction as data_transformation._transform_sft_split.
        norm = (
            df["input"]
            .str.strip()
            .str.strip('"')
            .str.strip()
            .str.lower()
            .str.replace(" ", "", regex=False)
        )
        has_input = norm != "<noinput>"
        df = df.copy()
        df["user_prompt"] = np.where(
            has_input,
            df["instruction"].str.strip() + "\n\n" + df["input"].str.strip(),
            df["instruction"].str.strip(),
        )
        return df

    def _generate(self, model, tokenizer, prompts: list[str]) -> list[str]:
        import torch
        from tqdm import tqdm

        model.eval()
        out = []
        for i in tqdm(range(len(prompts)), desc="Generating"):
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompts[i]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                chat_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH - JUDGE_MAX_NEW_TOKENS,
                add_special_tokens=False,  # chat template already adds BOS
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            out.append(
                tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                ).strip()
            )
        return out

    def run(self) -> str:
        """Generate base + fine-tuned responses on the test split -> parquet."""
        from peft import PeftModel
        from unsloth import FastLanguageModel

        df = pd.read_parquet(
            self.model_evaluation_config.model_evaluation_test_file_name
        )
        df = self._build_prompts(df)
        df = df.sample(
            n=min(JUDGE_N_SAMPLES, len(df)), random_state=RANDOM_STATE
        ).reset_index(drop=True)
        prompts = df["user_prompt"].tolist()
        logger.info("Generating responses for %d held-out test prompts.", len(df))

        # Load base weights once. Base responses first, then add the adapter on
        # top of the same weights for the fine-tuned responses.
        model, tokenizer = self.load_base_model()
        FastLanguageModel.for_inference(model)

        logger.info("Generating BASE (no adapter) responses...")
        base_responses = self._generate(model, tokenizer, prompts)

        logger.info("Attaching adapter — generating FINE-TUNED responses...")
        model = PeftModel.from_pretrained(
            model, self.model_trainer_artifact.adapter_path
        )
        FastLanguageModel.for_inference(model)
        ft_responses = self._generate(model, tokenizer, prompts)

        out = pd.DataFrame(
            {
                "instruction": df["user_prompt"],
                "reference": df["output"].str.strip(),
                "response": ft_responses,
                "base_response": base_responses,
            }
        )
        os.makedirs(self.model_evaluation_config.model_evaluation_dir, exist_ok=True)
        path = self.model_evaluation_config.model_evaluation_responses_file_name
        out.to_parquet(path)
        logger.info("Wrote %d responses to %s", len(out), path)
        return path
