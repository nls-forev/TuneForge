from pathlib import Path

from src.entity.experiment_config import ExperimentConfig


def test_experiment_config_round_trip(tmp_path: Path):
    config = ExperimentConfig.load()
    destination = tmp_path / "experiment_config.json"

    config.save_json(destination)

    assert destination.exists()
    assert '"revision"' in destination.read_text()
    assert config.generation.model_kwargs()["do_sample"] is True


def test_resolved_config_records_evaluation_ids():
    config = ExperimentConfig.load()

    resolved = config.with_sample_ids(["12", "3"])

    assert resolved.evaluation.sample_ids == ("12", "3")
    assert config.evaluation.sample_ids == ()
