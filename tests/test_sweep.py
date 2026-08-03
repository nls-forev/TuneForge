from src.pipeline.sweep import sweep_commands


def test_sweep_builds_cartesian_grid(tmp_path, monkeypatch):
    config = tmp_path / "sweeps.yaml"
    config.write_text(
        "parameters:\n"
        "  lora_rank: [8, 16]\n"
        "  learning_rate: [0.001]\n"
        "  epochs: [1, 2]\n"
        "  max_seq_length: [1024]\n"
    )
    monkeypatch.setattr("src.pipeline.sweep.from_root", lambda: tmp_path)

    commands = sweep_commands("sweeps.yaml")

    assert len(commands) == 4
    assert any("lora.lora_r=16" in part for part in commands[-1])
