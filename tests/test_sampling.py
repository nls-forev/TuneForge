import pandas as pd

from src.components.evaluation.generate_responses import GenerateResponses
from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.experiment_config import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
)


def _config(sample_ids=()):
    return ExperimentConfig(
        seed=7,
        dataset=DatasetConfig("dataset", "rev", "train", 0.2),
        model=ModelConfig("model", "rev", 128),
        generation=GenerationConfig(16, False, 1.0, 1.0, 0, 1.0, (7,)),
        evaluation=EvaluationConfig(3, sample_ids),
    )


def _generator(config):
    return GenerateResponses(
        ModelTrainerArtifact("adapter", 0.0, 0.0), experiment_config=config
    )


def test_sampling_is_deterministic_and_preserves_source_ids():
    frame = pd.DataFrame({"source_row_id": [str(i) for i in range(10)]})

    first = _generator(_config())._select_examples(frame)
    second = _generator(_config())._select_examples(frame)

    assert first["source_row_id"].tolist() == second["source_row_id"].tolist()
    assert len(first) == 3


def test_configured_sample_ids_keep_requested_order():
    frame = pd.DataFrame({"source_row_id": ["a", "b", "c"]})

    selected = _generator(_config(("c", "a")))._select_examples(frame)

    assert selected["source_row_id"].tolist() == ["c", "a"]
