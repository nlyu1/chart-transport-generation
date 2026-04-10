from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import threading

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from tqdm.autonotebook import tqdm

from src.config.base import BaseConfig
from src.deterministic_chart_transport.study import (
    DeterministicChartTransportStudyConfig,
    DeterministicChartTransportStudyState,
)
from src.monitoring.utils import save_go, scatterplot
from src.stochastic_chart_transport.study import (
    StochasticChartTransportStudyConfig,
    StochasticChartTransportStudyState,
)


def resolve_device(*, device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def get_device_context(*, device: torch.device):
    if device.type == "cuda":
        return torch.cuda.device(device)
    return nullcontext()


def get_autocast_context(*, device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _start_torch_save_thread(*, obj, path: Path) -> threading.Thread:
    path.parent.mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(
        target=torch.save,
        args=(obj, path),
        daemon=False,
    )
    thread.start()
    return thread


class AsyncArtifactWriter:
    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []

    def save_snapshot(self, *, snapshot: dict, path: Path) -> None:
        self._threads.append(_start_torch_save_thread(obj=snapshot, path=path))

    def save_figure(self, *, figure: go.Figure, path_root: Path) -> None:
        path_root.parent.mkdir(parents=True, exist_ok=True)
        thread = threading.Thread(
            target=save_go,
            args=(figure, path_root.parent, path_root.name),
            daemon=False,
        )
        thread.start()
        self._threads.append(thread)

    def join(self) -> None:
        for thread in self._threads:
            thread.join()
        self._threads.clear()


class ChartTransportTrainingPhaseConfig(BaseConfig):
    num_steps: int


class IntegratedTrainingPhaseConfig(ChartTransportTrainingPhaseConfig):
    checkpoint_every_n_steps: int
    transport_chart_every_n_steps: int


class CheckpointSelectionConfig(BaseConfig):
    start_step: int
    end_step: int
    step_stride: int

    def steps(self) -> list[int]:
        return list(range(self.start_step, self.end_step + 1, self.step_stride))


class ChartTransportTwoPanelFigureConfig(BaseConfig):
    marker_size: float
    marker_opacity: float
    width: int
    height: int
    combined_width: int
    combined_height: int
    camera_eye: tuple[float, float, float]


class ChartTransportResourceConfig(BaseConfig):
    visualize_batch_size_per_mode: int
    model_latent_batch_size: int
    checkpoint_selection: CheckpointSelectionConfig
    figure: ChartTransportTwoPanelFigureConfig


class StochasticChartTransportWriteupConfig(BaseConfig):
    artifact_root: Path
    seed: int
    device_name: str
    batch_size: int
    study: StochasticChartTransportStudyConfig
    pretrain: ChartTransportTrainingPhaseConfig
    critic: ChartTransportTrainingPhaseConfig
    integrated: IntegratedTrainingPhaseConfig
    resources: ChartTransportResourceConfig


class DeterministicChartTransportWriteupConfig(BaseConfig):
    artifact_root: Path
    seed: int
    device_name: str
    batch_size: int
    study: DeterministicChartTransportStudyConfig
    pretrain: ChartTransportTrainingPhaseConfig
    critic: ChartTransportTrainingPhaseConfig
    integrated: IntegratedTrainingPhaseConfig
    resources: ChartTransportResourceConfig


@dataclass
class ChartTransportTrainingResult:
    device: torch.device
    state: object
    history: dict[str, list[float]]
    checkpoint_paths: list[Path]


@dataclass
class ChartTransportRenderResult:
    selected_steps: list[int]
    individual_paths: list[Path]
    combined_path: Path
    combined_figure: go.Figure


def _checkpoint_dir(*, artifact_root: Path) -> Path:
    return artifact_root / "checkpoints"


def _resource_dir(*, artifact_root: Path) -> Path:
    return artifact_root / "writeup_resources"


def _snapshot_dir(*, artifact_root: Path) -> Path:
    return _resource_dir(artifact_root=artifact_root) / "snapshots"


def _individual_figure_dir(*, artifact_root: Path) -> Path:
    return _resource_dir(artifact_root=artifact_root) / "individual"


def _combined_figure_root(*, artifact_root: Path, name: str) -> Path:
    return _resource_dir(artifact_root=artifact_root) / name


def _monitor_bundle_path(*, artifact_root: Path) -> Path:
    return artifact_root / "monitor_bundle.pt"


def _training_history_path(*, artifact_root: Path) -> Path:
    return artifact_root / "training_history.pt"


def _writeup_config_path(*, artifact_root: Path) -> Path:
    return artifact_root / "writeup_config.pt"


def _step_checkpoint_path(*, artifact_root: Path, step: int) -> Path:
    return _checkpoint_dir(artifact_root=artifact_root) / f"step_{step:06d}.pt"


def _configure_scene(
    fig: go.Figure,
    *,
    scene_name: str,
    config: ChartTransportTwoPanelFigureConfig,
) -> None:
    fig.update_layout(
        {
            scene_name: dict(
                xaxis=dict(title="x"),
                yaxis=dict(title="y"),
                zaxis=dict(title="z"),
                aspectmode="data",
                camera=dict(
                    eye=dict(
                        x=config.camera_eye[0],
                        y=config.camera_eye[1],
                        z=config.camera_eye[2],
                    )
                ),
            )
        }
    )


def _configure_all_scenes(
    fig: go.Figure,
    *,
    num_scenes: int,
    config: ChartTransportTwoPanelFigureConfig,
) -> None:
    for index in range(num_scenes):
        scene_name = "scene" if index == 0 else f"scene{index + 1}"
        _configure_scene(fig, scene_name=scene_name, config=config)


def _restyle_scatter_figure(
    fig: go.Figure,
    *,
    marker_size: float,
    marker_opacity: float,
) -> go.Figure:
    for trace in fig.data:
        trace.update(marker=dict(size=marker_size), opacity=marker_opacity)
    return fig


def _make_two_panel_figure(
    *,
    latent_clouds: dict[str, torch.Tensor],
    sample_clouds: dict[str, torch.Tensor],
    step: int,
    config: ChartTransportTwoPanelFigureConfig,
) -> go.Figure:
    latent_figure = _restyle_scatter_figure(
        scatterplot(latent_clouds),
        marker_size=config.marker_size,
        marker_opacity=config.marker_opacity,
    )
    sample_figure = _restyle_scatter_figure(
        scatterplot(sample_clouds),
        marker_size=config.marker_size,
        marker_opacity=config.marker_opacity,
    )
    combo = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"Step {step}: latent clouds",
            f"Step {step}: sample clouds",
        ),
        horizontal_spacing=0.03,
    )
    for trace in latent_figure.data:
        combo.add_trace(trace, row=1, col=1)
    for trace in sample_figure.data:
        combo.add_trace(trace, row=1, col=2)
    combo.update_layout(
        width=config.width,
        height=config.height,
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    _configure_all_scenes(combo, num_scenes=2, config=config)
    return combo


def _make_progress_figure(
    *,
    snapshots: list[dict],
    config: ChartTransportTwoPanelFigureConfig,
) -> go.Figure:
    num_steps = len(snapshots)
    subplot_titles = []
    for snapshot in snapshots:
        subplot_titles.extend(
            [
                f"Step {snapshot['step']}: latent",
                f"Step {snapshot['step']}: sample",
            ]
        )
    combo = make_subplots(
        rows=2,
        cols=num_steps,
        specs=[[{"type": "scene"}] * num_steps, [{"type": "scene"}] * num_steps],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )
    for column, snapshot in enumerate(snapshots, start=1):
        latent_figure = _restyle_scatter_figure(
            scatterplot(snapshot["latent_clouds"]),
            marker_size=config.marker_size,
            marker_opacity=config.marker_opacity,
        )
        sample_figure = _restyle_scatter_figure(
            scatterplot(snapshot["sample_clouds"]),
            marker_size=config.marker_size,
            marker_opacity=config.marker_opacity,
        )
        for trace in latent_figure.data:
            combo.add_trace(trace, row=1, col=column)
        for trace in sample_figure.data:
            combo.add_trace(trace, row=2, col=column)
    combo.update_layout(
        width=config.combined_width,
        height=config.combined_height,
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    _configure_all_scenes(combo, num_scenes=2 * num_steps, config=config)
    return combo


def _save_checkpoint(
    *,
    state,
    artifact_root: Path,
    step: int,
    checkpoint_paths: list[Path],
) -> None:
    checkpoint_path = _step_checkpoint_path(artifact_root=artifact_root, step=step)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    checkpoint_paths.append(checkpoint_path)


def _sample_monitor_data(
    *,
    num_modes: int,
    sample_class,
    batch_size_per_mode: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        f"mode_{mode_id}": sample_class(
            mode_id=mode_id,
            batch_size=batch_size_per_mode,
        ).to(device)
        for mode_id in range(num_modes)
    }


def _save_monitor_bundle(
    *,
    monitor_bundle: dict,
    artifact_root: Path,
) -> None:
    path = _monitor_bundle_path(artifact_root=artifact_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_move_nested_tensors_to_cpu(monitor_bundle), path)


def _load_monitor_bundle(*, artifact_root: Path) -> dict:
    return torch.load(
        _monitor_bundle_path(artifact_root=artifact_root),
        map_location="cpu",
        weights_only=False,
    )


def _load_checkpoint(*, artifact_root: Path, step: int, device: torch.device):
    return torch.load(
        _step_checkpoint_path(artifact_root=artifact_root, step=step),
        map_location=device,
        weights_only=False,
    )


def _move_nested_tensors_to_cpu(value):
    if isinstance(value, dict):
        return {
            key: _move_nested_tensors_to_cpu(inner_value)
            for key, inner_value in value.items()
        }
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _move_nested_tensors_to_device(value, *, device: torch.device):
    if isinstance(value, dict):
        return {
            key: _move_nested_tensors_to_device(inner_value, device=device)
            for key, inner_value in value.items()
        }
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _monitor_bundle_for_stochastic(
    *,
    config: StochasticChartTransportWriteupConfig,
    state: StochasticChartTransportStudyState,
    device: torch.device,
) -> dict:
    batch_size_per_mode = config.resources.visualize_batch_size_per_mode
    model_batch_size = config.resources.model_latent_batch_size
    return {
        "mode_samples": _sample_monitor_data(
            num_modes=config.study.data.num_modes,
            sample_class=config.study.data.sample_class,
            batch_size_per_mode=batch_size_per_mode,
            device=device,
        ),
        "data_fiber": state.get_fiber(batch_size=batch_size_per_mode).to(device),
        "prior_latent": state.prior_config.sample(batch_size=model_batch_size).to(device),
        "model_fiber": state.get_fiber(batch_size=model_batch_size).to(device),
    }


def _monitor_bundle_for_deterministic(
    *,
    config: DeterministicChartTransportWriteupConfig,
    state: DeterministicChartTransportStudyState,
    device: torch.device,
) -> dict:
    batch_size_per_mode = config.resources.visualize_batch_size_per_mode
    model_batch_size = config.resources.model_latent_batch_size
    return {
        "mode_samples": _sample_monitor_data(
            num_modes=config.study.data.num_modes,
            sample_class=config.study.data.sample_class,
            batch_size_per_mode=batch_size_per_mode,
            device=device,
        ),
        "prior_latent": state.prior_config.sample(batch_size=model_batch_size).to(device),
    }


def _snapshot_from_stochastic_state(
    *,
    state: StochasticChartTransportStudyState,
    monitor_bundle: dict,
    step: int,
) -> dict:
    with torch.no_grad():
        model_sample, _ = state.decode(monitor_bundle["prior_latent"])
        model_latent = state.encode(
            data=model_sample,
            fiber=monitor_bundle["model_fiber"],
        )
        data_latent = {
            mode_name: state.encode(
                data=mode_samples,
                fiber=monitor_bundle["data_fiber"],
            )
            for mode_name, mode_samples in monitor_bundle["mode_samples"].items()
        }
    latent_clouds = {
        key: value.detach().cpu()
        for key, value in (data_latent | {"model": model_latent}).items()
    }
    sample_clouds = {
        key: value.detach().cpu()
        for key, value in (
            monitor_bundle["mode_samples"] | {"model": model_sample}
        ).items()
    }
    return {
        "step": step,
        "latent_clouds": latent_clouds,
        "sample_clouds": sample_clouds,
    }


def _snapshot_from_deterministic_state(
    *,
    state: DeterministicChartTransportStudyState,
    monitor_bundle: dict,
    step: int,
) -> dict:
    with torch.no_grad():
        model_sample = state.decode(monitor_bundle["prior_latent"])
        model_latent = state.encode(data=model_sample)
        data_latent = {
            mode_name: state.encode(data=mode_samples)
            for mode_name, mode_samples in monitor_bundle["mode_samples"].items()
        }
    latent_clouds = {
        key: value.detach().cpu()
        for key, value in (data_latent | {"model": model_latent}).items()
    }
    sample_clouds = {
        key: value.detach().cpu()
        for key, value in (
            monitor_bundle["mode_samples"] | {"model": model_sample}
        ).items()
    }
    return {
        "step": step,
        "latent_clouds": latent_clouds,
        "sample_clouds": sample_clouds,
    }


def run_stochastic_writeup_training(
    *,
    config: StochasticChartTransportWriteupConfig,
) -> ChartTransportTrainingResult:
    torch.manual_seed(config.seed)
    artifact_root = config.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    torch.save(config, _writeup_config_path(artifact_root=artifact_root))
    device = resolve_device(device_name=config.device_name)
    state = StochasticChartTransportStudyState.initialize(
        config=config.study,
        device=device,
    )
    monitor_bundle = _monitor_bundle_for_stochastic(
        config=config,
        state=state,
        device=device,
    )
    _save_monitor_bundle(monitor_bundle=monitor_bundle, artifact_root=artifact_root)

    history = {"pretrain": [], "critic": [], "integrated": []}
    checkpoint_paths: list[Path] = []
    _save_checkpoint(
        state=state,
        artifact_root=artifact_root,
        step=0,
        checkpoint_paths=checkpoint_paths,
    )

    for _ in tqdm(range(config.pretrain.num_steps), desc="stochastic pretrain"):
        data = config.study.data.sample_unconditional(batch_size=config.batch_size).to(device)
        with get_device_context(device=device), get_autocast_context(device=device):
            loss = state.compute_chart_pretrain_loss(data=data)
            total_loss = loss.sum()
        total_loss.backward()
        state.step_and_zero_grad()
        history["pretrain"].append(float(total_loss.detach().cpu()))

    for _ in tqdm(range(config.critic.num_steps), desc="stochastic critic"):
        data = config.study.data.sample_unconditional(batch_size=config.batch_size).to(device)
        with get_device_context(device=device), get_autocast_context(device=device):
            loss = state.compute_critic_only_loss(data=data)
            total_loss = loss.sum()
        total_loss.backward()
        state.step_and_zero_grad()
        history["critic"].append(float(total_loss.detach().cpu()))

    for step in tqdm(range(1, config.integrated.num_steps + 1), desc="stochastic train"):
        data = config.study.data.sample_unconditional(batch_size=config.batch_size).to(device)
        with get_device_context(device=device), get_autocast_context(device=device):
            losses = state.compute_integrated_loss(
                data=data,
                compute_transport_loss=(
                    step % config.integrated.transport_chart_every_n_steps == 0
                ),
            )
            total_loss = losses.sum()
        total_loss.backward()
        state.step_and_zero_grad()
        history["integrated"].append(float(total_loss.detach().cpu()))
        if step % config.integrated.checkpoint_every_n_steps == 0:
            _save_checkpoint(
                state=state,
                artifact_root=artifact_root,
                step=step,
                checkpoint_paths=checkpoint_paths,
            )

    torch.save(history, _training_history_path(artifact_root=artifact_root))
    return ChartTransportTrainingResult(
        device=device,
        state=state,
        history=history,
        checkpoint_paths=checkpoint_paths,
    )


def run_deterministic_writeup_training(
    *,
    config: DeterministicChartTransportWriteupConfig,
) -> ChartTransportTrainingResult:
    torch.manual_seed(config.seed)
    artifact_root = config.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    torch.save(config, _writeup_config_path(artifact_root=artifact_root))
    device = resolve_device(device_name=config.device_name)
    state = DeterministicChartTransportStudyState.initialize(
        config=config.study,
        device=device,
    )
    monitor_bundle = _monitor_bundle_for_deterministic(
        config=config,
        state=state,
        device=device,
    )
    _save_monitor_bundle(monitor_bundle=monitor_bundle, artifact_root=artifact_root)

    history = {"pretrain": [], "critic": [], "integrated": []}
    checkpoint_paths: list[Path] = []
    _save_checkpoint(
        state=state,
        artifact_root=artifact_root,
        step=0,
        checkpoint_paths=checkpoint_paths,
    )

    for _ in tqdm(range(config.pretrain.num_steps), desc="deterministic pretrain"):
        data = config.study.data.sample_unconditional(batch_size=config.batch_size).to(device)
        with get_device_context(device=device), get_autocast_context(device=device):
            loss = state.compute_chart_pretrain_loss(data=data)
            total_loss = loss.sum()
        total_loss.backward()
        state.step_and_zero_grad()
        history["pretrain"].append(float(total_loss.detach().cpu()))

    for _ in tqdm(range(config.critic.num_steps), desc="deterministic critic"):
        data = config.study.data.sample_unconditional(batch_size=config.batch_size).to(device)
        with get_device_context(device=device), get_autocast_context(device=device):
            loss = state.compute_critic_only_loss(data=data)
            total_loss = loss.sum()
        total_loss.backward()
        state.step_and_zero_grad()
        history["critic"].append(float(total_loss.detach().cpu()))

    for step in tqdm(range(1, config.integrated.num_steps + 1), desc="deterministic train"):
        data = config.study.data.sample_unconditional(batch_size=config.batch_size).to(device)
        with get_device_context(device=device), get_autocast_context(device=device):
            losses = state.compute_integrated_loss(
                data=data,
                compute_transport_loss=(
                    step % config.integrated.transport_chart_every_n_steps == 0
                ),
            )
            total_loss = losses.sum()
        total_loss.backward()
        state.step_and_zero_grad()
        history["integrated"].append(float(total_loss.detach().cpu()))
        if step % config.integrated.checkpoint_every_n_steps == 0:
            _save_checkpoint(
                state=state,
                artifact_root=artifact_root,
                step=step,
                checkpoint_paths=checkpoint_paths,
            )

    torch.save(history, _training_history_path(artifact_root=artifact_root))
    return ChartTransportTrainingResult(
        device=device,
        state=state,
        history=history,
        checkpoint_paths=checkpoint_paths,
    )


def render_stochastic_writeup_resources(
    *,
    config: StochasticChartTransportWriteupConfig,
) -> ChartTransportRenderResult:
    artifact_root = config.artifact_root
    device = resolve_device(device_name=config.device_name)
    monitor_bundle = _move_nested_tensors_to_device(
        _load_monitor_bundle(artifact_root=artifact_root),
        device=device,
    )
    writer = AsyncArtifactWriter()
    snapshots = []
    individual_paths = []
    for step in tqdm(config.resources.checkpoint_selection.steps(), desc="stochastic render"):
        state = _load_checkpoint(artifact_root=artifact_root, step=step, device=device)
        snapshot = _snapshot_from_stochastic_state(
            state=state,
            monitor_bundle=monitor_bundle,
            step=step,
        )
        figure = _make_two_panel_figure(
            latent_clouds=snapshot["latent_clouds"],
            sample_clouds=snapshot["sample_clouds"],
            step=step,
            config=config.resources.figure,
        )
        snapshot_path = _snapshot_dir(artifact_root=artifact_root) / f"step_{step:06d}.pt"
        figure_root = _individual_figure_dir(artifact_root=artifact_root) / f"step_{step:06d}"
        writer.save_snapshot(snapshot=snapshot, path=snapshot_path)
        writer.save_figure(figure=figure, path_root=figure_root)
        snapshots.append(snapshot)
        individual_paths.append(figure_root)
    combined_figure = _make_progress_figure(
        snapshots=snapshots,
        config=config.resources.figure,
    )
    combined_path = _combined_figure_root(
        artifact_root=artifact_root,
        name="stochastic_progression",
    )
    writer.save_figure(figure=combined_figure, path_root=combined_path)
    writer.join()
    return ChartTransportRenderResult(
        selected_steps=config.resources.checkpoint_selection.steps(),
        individual_paths=individual_paths,
        combined_path=combined_path,
        combined_figure=combined_figure,
    )


def render_deterministic_writeup_resources(
    *,
    config: DeterministicChartTransportWriteupConfig,
) -> ChartTransportRenderResult:
    artifact_root = config.artifact_root
    device = resolve_device(device_name=config.device_name)
    monitor_bundle = _move_nested_tensors_to_device(
        _load_monitor_bundle(artifact_root=artifact_root),
        device=device,
    )
    writer = AsyncArtifactWriter()
    snapshots = []
    individual_paths = []
    for step in tqdm(config.resources.checkpoint_selection.steps(), desc="deterministic render"):
        state = _load_checkpoint(artifact_root=artifact_root, step=step, device=device)
        snapshot = _snapshot_from_deterministic_state(
            state=state,
            monitor_bundle=monitor_bundle,
            step=step,
        )
        figure = _make_two_panel_figure(
            latent_clouds=snapshot["latent_clouds"],
            sample_clouds=snapshot["sample_clouds"],
            step=step,
            config=config.resources.figure,
        )
        snapshot_path = _snapshot_dir(artifact_root=artifact_root) / f"step_{step:06d}.pt"
        figure_root = _individual_figure_dir(artifact_root=artifact_root) / f"step_{step:06d}"
        writer.save_snapshot(snapshot=snapshot, path=snapshot_path)
        writer.save_figure(figure=figure, path_root=figure_root)
        snapshots.append(snapshot)
        individual_paths.append(figure_root)
    combined_figure = _make_progress_figure(
        snapshots=snapshots,
        config=config.resources.figure,
    )
    combined_path = _combined_figure_root(
        artifact_root=artifact_root,
        name="deterministic_progression",
    )
    writer.save_figure(figure=combined_figure, path_root=combined_path)
    writer.join()
    return ChartTransportRenderResult(
        selected_steps=config.resources.checkpoint_selection.steps(),
        individual_paths=individual_paths,
        combined_path=combined_path,
        combined_figure=combined_figure,
    )


def build_stochastic_progress_figure(
    *,
    config: StochasticChartTransportWriteupConfig,
) -> go.Figure:
    snapshot_dir = _snapshot_dir(artifact_root=config.artifact_root)
    snapshots = [
        torch.load(
            snapshot_dir / f"step_{step:06d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for step in config.resources.checkpoint_selection.steps()
    ]
    return _make_progress_figure(
        snapshots=snapshots,
        config=config.resources.figure,
    )


def build_deterministic_progress_figure(
    *,
    config: DeterministicChartTransportWriteupConfig,
) -> go.Figure:
    snapshot_dir = _snapshot_dir(artifact_root=config.artifact_root)
    snapshots = [
        torch.load(
            snapshot_dir / f"step_{step:06d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for step in config.resources.checkpoint_selection.steps()
    ]
    return _make_progress_figure(
        snapshots=snapshots,
        config=config.resources.figure,
    )
