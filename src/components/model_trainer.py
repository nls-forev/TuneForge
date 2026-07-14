from __future__ import annotations

import os
from typing import Any, Tuple

import pandas as pd
from from_root import from_root

from src.constants import (
    HYPERPARAMS_FILE_NAME,
    LOAD_IN_4BIT,
    LORA_BIAS,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    RANDOM_STATE,
    USE_GRADIENT_CHECKPOINTING,
    WANDB_PROJECT,
)
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelTrainerConfig
from src.logger import get_logger
from src.utils.main_utils import load_yaml

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig | None = None,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config or ModelTrainerConfig()

        # Load hyperparameters once and reuse them.
        self.hyperparams: dict[str, Any] = load_yaml(
            os.path.join(from_root(), HYPERPARAMS_FILE_NAME)
        )

    def _load_model(self):
        try:
            from unsloth import FastLanguageModel

            lora_config = self.hyperparams["lora"]

            logger.info("Loaded model and LoRA hyperparameters.")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_ID,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=LOAD_IN_4BIT,
            )

            logger.info("Loaded %s and its tokenizer.", MODEL_ID)

            model = FastLanguageModel.get_peft_model(
                model=model,
                r=lora_config["lora_r"],
                lora_alpha=lora_config["lora_alpha"],
                lora_dropout=lora_config["lora_dropout"],
                target_modules=lora_config["target_modules"],
                use_rslora=lora_config["use_rslora"],
                use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
                random_state=RANDOM_STATE,
                bias=LORA_BIAS,
            )

            logger.info("Loaded PEFT model.")

            return model, tokenizer

        except Exception:
            logger.exception("Failed to load model.")
            raise

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            train_sft = pd.read_parquet(
                self.data_transformation_artifact.train_sft_file_path
            )
            test_sft = pd.read_parquet(
                self.data_transformation_artifact.test_sft_file_path
            )
            val_sft = pd.read_parquet(
                self.data_transformation_artifact.val_sft_file_path
            )

            return train_sft, test_sft, val_sft

        except FileNotFoundError:
            logger.exception("Training dataset files not found.")
            raise

        except Exception:
            logger.exception("Failed to load datasets.")
            raise

    def _train(self, model, tokenizer, train_df: pd.DataFrame, val_df: pd.DataFrame):
        try:
            from datasets import Dataset
            from trl import SFTConfig, SFTTrainer
            from unsloth import is_bfloat16_supported

            cfg = self.hyperparams["model"]

            os.environ["WANDB_PROJECT"] = WANDB_PROJECT

            if not os.environ.get("WANDB_API_KEY"):
                os.environ["WANDB_MODE"] = "offline"
                logger.warning("WANDB_API_KEY unset — running W&B in offline mode.")

            train_ds = Dataset.from_pandas(train_df, preserve_index=False)
            val_ds = Dataset.from_pandas(val_df, preserve_index=False)

            logger.info(
                "Built HF datasets — train: %d rows, val: %d rows.",
                len(train_ds),
                len(val_ds),
            )

            bf16 = is_bfloat16_supported()

            sft_kwargs = {
                "output_dir": self.model_trainer_config.model_trainer_checkpoint_dir,
                "per_device_train_batch_size": cfg["batch_size"],
                "gradient_accumulation_steps": cfg["grad_accum"],
                "learning_rate": float(cfg["lr"]),
                "num_train_epochs": cfg["epochs"],
                "warmup_ratio": cfg["warmup_ratio"],
                "weight_decay": cfg["weight_decay"],
                "lr_scheduler_type": cfg["lr_scheduler"],
                "logging_steps": cfg["logging_steps"],
                "eval_strategy": "steps",
                "eval_steps": cfg["eval_steps"],
                "save_steps": cfg["save_steps"],
                "save_total_limit": cfg["save_total_limit"],
                # safetensors drops unsloth adapter weights from checkpoints
                # (breaks both the saved model and resume) — use torch.save.
                "save_safetensors": False,
                "dataset_text_field": "text",
                "max_length": MAX_SEQ_LENGTH,
                "seed": RANDOM_STATE,
                "report_to": "wandb",
                "run_name": self.model_trainer_config.run_name,
                "optim": "adamw_8bit",
                "bf16": bf16,
                "fp16": not bf16,
            }

            sft_config = SFTConfig(**sft_kwargs)  # ty:ignore[invalid-argument-type]

            trainer = SFTTrainer(
                model=model,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                args=sft_config,
                processing_class=tokenizer,
            )

            # Auto-resume if checkpoint exists else fresh start.
            ckpt_dir = self.model_trainer_config.model_trainer_checkpoint_dir
            has_checkpoint = os.path.isdir(ckpt_dir) and any(
                name.startswith("checkpoint-") for name in os.listdir(ckpt_dir)
            )
            if has_checkpoint:
                logger.info("Found existing checkpoint in %s — resuming.", ckpt_dir)
            else:
                logger.info("No checkpoint found — starting fresh training.")

            logger.info("Starting SFT training...")
            train_stats = trainer.train(resume_from_checkpoint=has_checkpoint)
            logger.info("Completed training. Stats: %s", train_stats.metrics)

            return trainer, train_stats

        except Exception:
            logger.exception("Model training failed.")
            raise

    def _save(self, model, tokenizer) -> str:
        try:
            import glob

            adapter_path = self.model_trainer_config.model_trainer_adapter_dir
            os.makedirs(adapter_path, exist_ok=True)

            # safetensors silently drops the adapter weights for unsloth models
            # trained with gradient offloading (writes adapter_config.json but no
            # adapter_model.safetensors). torch.save tolerates the offloaded state.
            model.save_pretrained(adapter_path, safe_serialization=False)
            tokenizer.save_pretrained(adapter_path)

            # A config without weights is useless — fail loudly instead of
            # reporting a "successful" save that produced no model.
            weights = glob.glob(os.path.join(adapter_path, "adapter_model.*"))
            total = sum(os.path.getsize(f) for f in weights)
            if total < 1_000_000:
                raise RuntimeError(
                    f"Adapter weights missing/too small after save "
                    f"({total} bytes). Dir contents: {os.listdir(adapter_path)}"
                )

            logger.info(
                "Saved + verified QLoRA adapter (%.1f MB) and tokenizer to: %s",
                total / 1e6,
                adapter_path,
            )

            return adapter_path

        except Exception:
            logger.exception("Failed to save adapter.")
            raise

    def init_model_trainer(self) -> ModelTrainerArtifact:
        logger.info("Initializing model trainer stage...")

        model, tokenizer = self._load_model()
        train_df, _test_df, val_df = self._load_data()

        _trainer, train_stats = self._train(
            model,
            tokenizer,
            train_df,
            val_df,
        )

        adapter_path = self._save(model, tokenizer)

        metrics = train_stats.metrics

        logger.info("Completed model trainer stage.")

        return ModelTrainerArtifact(
            adapter_path=adapter_path,
            train_loss=metrics.get("train_loss", float("nan")),
            train_runtime=metrics.get("train_runtime", float("nan")),
        )
