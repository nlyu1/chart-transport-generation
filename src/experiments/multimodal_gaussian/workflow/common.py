from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
from typing import Any

from src.config.base import BaseConfig
from src.experiments.multimodal_gaussian.serialization import (
    MultimodalSerializationConfig,
)


class MultimodalGaussianExperimentConfig(BaseConfig):
    training_config: object
    serialization_config: MultimodalSerializationConfig
    wandb_project: str
    run_name: str


def build_cli_parser(
    *,
    description: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Python config file exporting experiment_config or build_experiment_config().",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        required=True,
        help="Value assigned to CUDA_VISIBLE_DEVICES before torch imports.",
    )
    return parser


def configure_cuda_visible_devices(
    *,
    cuda_visible_devices: str,
) -> None:
    if "torch" in sys.modules:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be configured before torch is imported"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def load_experiment_config(
    *,
    config_path: Path,
) -> MultimodalGaussianExperimentConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Experiment config must be a file: {config_path}")

    config_namespace = runpy.run_path(str(config_path))
    raw_config: Any
    if "build_experiment_config" in config_namespace:
        raw_config = config_namespace["build_experiment_config"]()
    elif "experiment_config" in config_namespace:
        raw_config = config_namespace["experiment_config"]
    else:
        raise KeyError(
            f"{config_path} must define `experiment_config` or `build_experiment_config`"
        )

    experiment_config = MultimodalGaussianExperimentConfig.model_validate(raw_config)

    from src.experiments.multimodal_gaussian.config import (
        MultimodalGaussianTrainingConfig,
    )

    if not isinstance(
        experiment_config.training_config,
        MultimodalGaussianTrainingConfig,
    ):
        raise TypeError(
            "experiment_config.training_config must be a "
            "MultimodalGaussianTrainingConfig instance"
        )

    return experiment_config


def initialize_runtime(
    *,
    experiment_config: MultimodalGaussianExperimentConfig,
):
    from src.experiments.multimodal_gaussian.state import MultimodalTrainingRuntime

    return MultimodalTrainingRuntime.initialize(
        tc=experiment_config.training_config,
        cuda_device=0,
        wandb_project=experiment_config.wandb_project,
        run_name=experiment_config.run_name,
    )


def run_pretrain_with_runtime(
    *,
    runtime: Any,
    serialization_config: MultimodalSerializationConfig,
) -> dict[str, dict[str, float]]:
    chart_metrics = runtime.chart_pretrain()
    if serialization_config.save_after_chart_pretrain:
        chart_model_path = runtime.save_chart_pretrain_model()
        runtime.fabric.print(f"Saved chart-pretrained model to {chart_model_path}")

    critic_metrics = runtime.critic_pretrain()
    if serialization_config.save_after_critic_pretrain:
        critic_model_path = runtime.save_critic_pretrain_model()
        runtime.fabric.print(f"Saved critic-pretrained model to {critic_model_path}")

    return {
        "chart_pretrain": chart_metrics,
        "critic_pretrain": critic_metrics,
    }


def run_pretrain_experiment(
    *,
    experiment_config: MultimodalGaussianExperimentConfig,
) -> dict[str, dict[str, float]]:
    runtime = initialize_runtime(experiment_config=experiment_config)
    try:
        return run_pretrain_with_runtime(
            runtime=runtime,
            serialization_config=experiment_config.serialization_config,
        )
    finally:
        runtime.finish()


def run_integrated_experiment(
    *,
    experiment_config: MultimodalGaussianExperimentConfig,
) -> dict[str, float]:
    runtime = initialize_runtime(experiment_config=experiment_config)
    try:
        model_path = runtime.load_latest_pretrain_model()
        runtime.fabric.print(f"Loaded pretrained model from {model_path}")
        return runtime.integrated()
    finally:
        runtime.finish()


def run_one_shot_experiment(
    *,
    experiment_config: MultimodalGaussianExperimentConfig,
) -> dict[str, float]:
    runtime = initialize_runtime(experiment_config=experiment_config)
    try:
        run_pretrain_with_runtime(
            runtime=runtime,
            serialization_config=experiment_config.serialization_config,
        )
        return runtime.integrated()
    finally:
        runtime.finish()


__all__ = [
    "MultimodalGaussianExperimentConfig",
    "build_cli_parser",
    "configure_cuda_visible_devices",
    "initialize_runtime",
    "load_experiment_config",
    "run_integrated_experiment",
    "run_one_shot_experiment",
    "run_pretrain_experiment",
    "run_pretrain_with_runtime",
]
