"""Typed, serializable configuration for a reproducible experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from from_root import from_root

from src.utils.main_utils import load_yaml

EXPERIMENT_CONFIG_FILE = "config/experiment.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    id: str
    revision: str
    split: str
    testval_split_ratio: float


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str
    max_seq_length: int


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    seeds: tuple[int, ...]

    def model_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        return kwargs


@dataclass(frozen=True)
class EvaluationConfig:
    n_samples: int
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ExperimentConfig":
        source = Path(path or Path(from_root()) / EXPERIMENT_CONFIG_FILE)
        raw = load_yaml(str(source))
        generation = dict(raw["generation"])
        generation["seeds"] = tuple(generation["seeds"])
        evaluation = dict(raw["evaluation"])
        evaluation["sample_ids"] = tuple(str(x) for x in evaluation["sample_ids"])
        return cls(
            seed=int(raw["seed"]),
            dataset=DatasetConfig(**raw["dataset"]),
            model=ModelConfig(**raw["model"]),
            generation=GenerationConfig(**generation),
            evaluation=EvaluationConfig(**evaluation),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_sample_ids(self, sample_ids: list[str]) -> "ExperimentConfig":
        """Return a resolved snapshot containing the exact evaluated rows."""
        return replace(
            self,
            evaluation=replace(self.evaluation, sample_ids=tuple(sample_ids)),
        )

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
