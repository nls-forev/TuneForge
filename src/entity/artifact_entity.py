from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str
    val_file_path: str


@dataclass
class DataTransformationArtifact:
    train_sft_file_path: str
    test_sft_file_path: str
    val_sft_file_path: str


@dataclass
class ModelTrainerArtifact:
    adapter_path: str
    train_loss: float
    train_runtime: float


@dataclass
class ModelEvaluationArtifact:
    metrics: dict
