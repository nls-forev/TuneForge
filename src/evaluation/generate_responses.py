"""Generate run4 free-text responses on the held-out AlpaCare test split.

Phase A of the LLM-as-judge eval: runs on the GPU box inside the eval
container, writes artifact/model_evaluation/responses.parquet with columns
(instruction, prompt, reference, response) for phase B (local judging).
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.components.model_evaluation import ModelEvaluation  # noqa: E402
from src.constants import MAX_SEQ_LENGTH  # noqa: E402
from src.entity.artifact_entity import (  # noqa: E402
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.logger import configure_logger, get_logger  # noqa: E402

configure_logger()
logger = get_logger(__name__)

TEST_PARQUET = "artifact/data_ingestion/data_ingested/test.parquet"
OUT_DIR = "artifact/model_evaluation"
OUT_PATH = os.path.join(OUT_DIR, "responses.parquet")

N_SAMPLES = int(os.environ.get("JUDGE_N_SAMPLES", "200"))
SEED = 42
MAX_NEW_TOKENS = 512


def build_prompts(df: pd.DataFrame) -> pd.DataFrame:
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


def main() -> int:
    import torch
    from tqdm import tqdm

    df = pd.read_parquet(TEST_PARQUET)
    df = build_prompts(df)
    df = df.sample(n=min(N_SAMPLES, len(df)), random_state=SEED).reset_index(drop=True)
    logger.info("Generating responses for %d held-out test prompts.", len(df))

    model, tokenizer = ModelEvaluation(
        data_transformation_artifact=DataTransformationArtifact(
            train_sft_file_path="",
            test_sft_file_path="",
            val_sft_file_path="",
            medmcqa_file_path="",
        ),
        model_trainer_artifact=ModelTrainerArtifact(
            adapter_path="artifact/model_trainer/adapter",
            train_loss=0.0,
            train_runtime=0.0,
        ),
    ).load_model()
    model.eval()

    responses = []
    for i in tqdm(range(len(df)), desc="Generating"):
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": df["user_prompt"].iloc[i]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
            add_special_tokens=False,  # chat template already adds BOS
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        responses.append(response)

        if i < 3:
            logger.info("sample %d response: %s", i, response[:200])

    out = pd.DataFrame(
        {
            "instruction": df["user_prompt"],
            "reference": df["output"].str.strip(),
            "response": responses,
        }
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_parquet(OUT_PATH)
    logger.info("Wrote %d responses to %s", len(out), OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
