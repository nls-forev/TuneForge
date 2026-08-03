"""Eval phase A (GPU): generate base + fine-tuned responses -> responses.parquet."""

import os

import pandas as pd

from src.constants import (
    LOAD_IN_4BIT,
    MODEL_PATH,
)
from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.experiment_config import ExperimentConfig
from src.logger import get_logger
from src.utils.text_utils import has_real_input

logger = get_logger(__name__)


class GenerateResponses:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig = ModelEvaluationConfig(),
        experiment_config: ExperimentConfig | None = None,
    ):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config
        self.experiment_config = experiment_config or ExperimentConfig.load()

    def load_base_model(self):
        # MODEL_ID matches training so LoRA deltas apply to the same frozen
        # weights; local MODEL_PATH snapshot preferred when present.
        try:
            from unsloth import FastLanguageModel

            model_id = (
                MODEL_PATH
                if os.path.isfile(os.path.join(MODEL_PATH, "config.json"))
                else self.experiment_config.model.id
            )
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                revision=self.experiment_config.model.revision,
                max_seq_length=self.experiment_config.model.max_seq_length,
                load_in_4bit=LOAD_IN_4BIT,
                device_map="sequential",
            )
            return model, tokenizer

        except Exception as e:
            logger.exception(f"Unexpected error occurred: {e}")
            raise

    @staticmethod
    def _build_prompts(df: pd.DataFrame) -> pd.DataFrame:
        # Same prompt construction as data_transformation._transform_sft_split.
        has_input = has_real_input(df["input"])
        df = df.copy()
        instruction = df["instruction"].fillna("").astype(str).str.strip()
        input_text = df["input"].fillna("").astype(str).str.strip()
        df["user_prompt"] = instruction.where(
            ~has_input, instruction + "\n\n" + input_text
        )
        return df

    def _generate(self, model, tokenizer, prompts: list[str], seed: int) -> list[str]:
        import torch
        from tqdm import tqdm

        model.eval()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
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
                max_length=(
                    self.experiment_config.model.max_seq_length
                    - self.experiment_config.generation.max_new_tokens
                ),
                add_special_tokens=False,  # chat template already adds BOS
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    **self.experiment_config.generation.model_kwargs(),
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

    def _select_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select fixed IDs when configured, otherwise sample deterministically."""
        if "source_row_id" not in df.columns:
            # Backward compatibility for existing ingested artifacts.
            df = df.copy()
            df["source_row_id"] = df.index.astype(str)
        df["source_row_id"] = df["source_row_id"].astype(str)
        requested = self.experiment_config.evaluation.sample_ids
        if requested:
            indexed = df.set_index("source_row_id", drop=False)
            missing = sorted(set(requested) - set(indexed.index))
            if missing:
                raise ValueError(f"evaluation sample IDs not found: {missing[:10]}")
            return indexed.loc[list(requested)].reset_index(drop=True)
        return df.sample(
            n=min(self.experiment_config.evaluation.n_samples, len(df)),
            random_state=self.experiment_config.seed,
        ).reset_index(drop=True)

    def run(self) -> str:
        """Generate base + fine-tuned responses on the test split -> parquet."""
        from peft import PeftModel
        from unsloth import FastLanguageModel

        df = pd.read_parquet(
            self.model_evaluation_config.model_evaluation_test_file_name
        )
        df = self._build_prompts(df)
        df = self._select_examples(df)
        prompts = df["user_prompt"].tolist()
        logger.info("Generating responses for %d held-out test prompts.", len(df))

        # Load base weights once. Base responses first, then add the adapter on
        # top of the same weights for the fine-tuned responses.
        model, tokenizer = self.load_base_model()
        FastLanguageModel.for_inference(model)

        logger.info("Generating BASE (no adapter) responses...")
        seeds = self.experiment_config.generation.seeds
        base_by_seed = {
            seed: self._generate(model, tokenizer, prompts, seed) for seed in seeds
        }

        logger.info("Attaching adapter — generating FINE-TUNED responses...")
        model = PeftModel.from_pretrained(
            model, self.model_trainer_artifact.adapter_path
        )
        FastLanguageModel.for_inference(model)
        ft_by_seed = {
            seed: self._generate(model, tokenizer, prompts, seed) for seed in seeds
        }

        rows = []
        for seed in seeds:
            rows.append(
                pd.DataFrame(
                    {
                        "source_row_id": df["source_row_id"],
                        "generation_seed": seed,
                        "instruction": df["user_prompt"],
                        "reference": df["output"].str.strip(),
                        "response": ft_by_seed[seed],
                        "base_response": base_by_seed[seed],
                    }
                )
            )
        out = pd.concat(rows, ignore_index=True)
        os.makedirs(self.model_evaluation_config.model_evaluation_dir, exist_ok=True)
        path = self.model_evaluation_config.model_evaluation_responses_file_name
        out.to_parquet(path)
        resolved_config = self.experiment_config.with_sample_ids(
            df["source_row_id"].astype(str).tolist()
        )
        resolved_config.save_json(
            os.path.join(
                self.model_evaluation_config.model_evaluation_dir,
                "responses.experiment.json",
            )
        )
        logger.info("Wrote %d responses to %s", len(out), path)
        return path
