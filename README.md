## Multimodal Gaussian CLI workflows

The notebook flow in [`chart-transport-new.ipynb`](/home/nlyu/Code/diffusive-latent-generation/chart-transport-new.ipynb) is now exposed through three CLI entrypoints:

- `uv run python -m src.experiments.multimodal_gaussian.workflow.pretrain`
- `uv run python -m src.experiments.multimodal_gaussian.workflow.integrated`
- `uv run python -m src.experiments.multimodal_gaussian.workflow.one_shot`

Each command requires:

- `--config /path/to/experiment_config.py`
- `--cuda-visible-devices <gpu-list>`

Example:

```bash
uv run python -m src.experiments.multimodal_gaussian.workflow.pretrain \
  --config /tmp/multimodal_workflow_config.py \
  --cuda-visible-devices 0
```

The config file should export either `experiment_config` or `build_experiment_config()`. A minimal pattern is:

```python
from pathlib import Path

from src.data.gaussian_mixture.data import MultimodalGaussianDataConfig
from src.experiments.multimodal_gaussian.canonical import (
    get_canonical_chart_transport_configs,
    get_canonical_chart_transport_monitor_configs,
)
from src.experiments.multimodal_gaussian.config import MultimodalGaussianTrainingConfig
from src.experiments.multimodal_gaussian.serialization import MultimodalSerializationConfig
from src.experiments.multimodal_gaussian.workflow.common import (
    MultimodalGaussianExperimentConfig,
)


def build_experiment_config() -> MultimodalGaussianExperimentConfig:
    data_config = MultimodalGaussianDataConfig.initialize(
        num_modes=8,
        mode_std=0.1,
        offset=0.0,
        ambient_dimension=8,
    )
    training_config = MultimodalGaussianTrainingConfig.initialize(
        seed=0,
        train_batch_size=4096,
        eval_batch_size=4096,
        integrated_n_steps=1_000_000,
        chart_transport_config=get_canonical_chart_transport_configs(
            data_config=data_config,
            latent_dimension=8,
        ),
        monitor_config=get_canonical_chart_transport_monitor_configs(),
        folder=Path("artifacts/multimodal_gaussian/example"),
        raise_on_existing_folder=False,
    )
    return MultimodalGaussianExperimentConfig(
        training_config=training_config,
        serialization_config=MultimodalSerializationConfig(
            save_after_chart_pretrain=True,
            save_after_critic_pretrain=True,
        ),
        wandb_project="diffusive-latent-generation",
        run_name="example",
    )
```

Saved model artifacts live under `<training_config.folder>/runtime/`:

- `chart_pretrain_model.pt`
- `critic_pretrain_model.pt`

`integrated` restores the newest available pretrain artifact, preferring `critic_pretrain_model.pt` and falling back to `chart_pretrain_model.pt`.
