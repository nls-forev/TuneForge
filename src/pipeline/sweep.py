"""Queue a reproducible DVC grid sweep from config/sweeps.yaml."""

from __future__ import annotations

import argparse
import subprocess
from itertools import product
from pathlib import Path

from from_root import from_root

from src.utils.main_utils import load_yaml


def sweep_commands(path: str | Path = "config/sweeps.yaml") -> list[list[str]]:
    raw = load_yaml(str(Path(from_root()) / path))["parameters"]
    keys = ("lora_rank", "learning_rate", "epochs", "max_seq_length")
    commands = []
    for rank, lr, epochs, max_length in product(*(raw[key] for key in keys)):
        commands.append(
            [
                "dvc",
                "exp",
                "run",
                "--queue",
                "-S",
                f"config/hyperparams.yaml:lora.lora_r={rank}",
                "-S",
                f"config/hyperparams.yaml:model.lr={lr}",
                "-S",
                f"config/hyperparams.yaml:model.epochs={epochs}",
                "-S",
                f"config/experiment.yaml:model.max_seq_length={max_length}",
            ]
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for command in sweep_commands():
        if args.dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
