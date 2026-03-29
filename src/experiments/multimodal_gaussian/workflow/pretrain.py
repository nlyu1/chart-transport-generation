from __future__ import annotations

from src.experiments.multimodal_gaussian.workflow.common import (
    build_cli_parser,
    configure_cuda_visible_devices,
    load_experiment_config,
    run_pretrain_experiment,
)


def main() -> None:
    parser = build_cli_parser(
        description="Run multimodal Gaussian chart and critic pretraining.",
    )
    args = parser.parse_args()
    configure_cuda_visible_devices(
        cuda_visible_devices=args.cuda_visible_devices,
    )
    experiment_config = load_experiment_config(
        config_path=args.config,
    )
    run_pretrain_experiment(
        experiment_config=experiment_config,
    )


if __name__ == "__main__":
    main()
