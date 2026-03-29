from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import torch
from jaxtyping import Float, Int
from pydantic import model_validator
from torch import Tensor
from torchvision.datasets import MNIST as TorchvisionMNIST

from src.data.base import BaseDataConfig


DataSplit = Literal["train", "val"]

NUM_MNIST_CLASSES = 10
MNIST_SIDE_LENGTH = 28
MNIST_NUMEL = MNIST_SIDE_LENGTH * MNIST_SIDE_LENGTH


class MNISTDataConfig(BaseDataConfig):
    split: DataSplit
    flatten: bool
    samples: Float[Tensor, "num_samples data_dim"]
    labels: Int[Tensor, "num_samples"]
    class_indices: tuple[Int[Tensor, "num_class_samples"], ...]

    @classmethod
    def initialize(
        cls,
        *,
        root: Path,
        split: DataSplit,
        flatten: bool,
        download: bool,
    ) -> Self:
        if not flatten:
            raise NotImplementedError("MNIST with flatten=False is not implemented yet")
        dataset = TorchvisionMNIST(
            root=root,
            train=split == "train",
            download=download,
        )
        samples = dataset.data.to(dtype=torch.float32) / 255.0
        samples = samples.reshape(samples.shape[0], MNIST_NUMEL)
        labels = dataset.targets.to(dtype=torch.long)
        class_indices = tuple(
            torch.nonzero(labels == mode_id, as_tuple=False).squeeze(-1)
            for mode_id in range(NUM_MNIST_CLASSES)
        )
        return cls(
            num_classes=NUM_MNIST_CLASSES,
            data_shape=[MNIST_NUMEL],
            split=split,
            flatten=flatten,
            samples=samples,
            labels=labels,
            class_indices=class_indices,
        )

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.num_classes != NUM_MNIST_CLASSES:
            raise ValueError("MNIST must have exactly 10 classes")
        if self.data_shape != [MNIST_NUMEL]:
            raise ValueError("flattened MNIST must report data_shape=[784]")
        if not self.flatten:
            raise NotImplementedError("MNIST with flatten=False is not implemented yet")
        if self.samples.ndim != 2:
            raise ValueError("samples must have shape [num_samples, data_dim]")
        if self.samples.shape[1] != MNIST_NUMEL:
            raise ValueError("samples must have shape [num_samples, 784]")
        if self.labels.ndim != 1:
            raise ValueError("labels must have shape [num_samples]")
        if self.samples.shape[0] != self.labels.shape[0]:
            raise ValueError("samples and labels must contain the same number of items")
        if len(self.class_indices) != self.num_classes:
            raise ValueError("class_indices must contain one tensor per class")
        for mode_id, mode_indices in enumerate(self.class_indices):
            if mode_indices.ndim != 1:
                raise ValueError("every class_indices entry must be one-dimensional")
            if mode_indices.numel() == 0:
                raise ValueError(f"class {mode_id} contains no samples")
        return self

    def _sample_from_indices(
        self,
        *,
        indices: Int[Tensor, "num_candidates"],
        batch_size: int,
    ) -> Float[Tensor, "batch data_dim"]:
        sampled_offsets = torch.randint(
            indices.shape[0],
            size=(batch_size,),
            device=indices.device,
        )
        return self.samples[indices[sampled_offsets]]

    def _slice_from_indices(
        self,
        *,
        indices: Int[Tensor, "num_candidates"],
        start_index: int,
        batch_size: int,
    ) -> Float[Tensor, "batch data_dim"]:
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        stop_index = start_index + batch_size
        if stop_index > indices.shape[0]:
            raise ValueError(
                f"requested [{start_index}, {stop_index}) exceeds class size {indices.shape[0]}"
            )
        return self.samples[indices[start_index:stop_index]]

    def sample_class(
        self,
        *,
        mode_id: int,
        batch_size: int,
    ) -> Float[Tensor, "batch data_dim"]:
        if mode_id < 0 or mode_id >= self.num_classes:
            raise ValueError(f"mode_id must be in [0, {self.num_classes})")
        return self._sample_from_indices(
            indices=self.class_indices[mode_id],
            batch_size=batch_size,
        )

    def class_batch(
        self,
        *,
        mode_id: int,
        batch_size: int,
        start_index: int,
    ) -> Float[Tensor, "batch data_dim"]:
        if mode_id < 0 or mode_id >= self.num_classes:
            raise ValueError(f"mode_id must be in [0, {self.num_classes})")
        return self._slice_from_indices(
            indices=self.class_indices[mode_id],
            start_index=start_index,
            batch_size=batch_size,
        )

    def stratified_batch(
        self,
        *,
        batch_size_per_class: int,
    ) -> tuple[Float[Tensor, "batch data_dim"], Int[Tensor, "batch"]]:
        samples = []
        labels = []
        for mode_id in range(self.num_classes):
            samples.append(
                self.sample_class(
                    mode_id=mode_id,
                    batch_size=batch_size_per_class,
                )
            )
            labels.append(
                torch.full(
                    (batch_size_per_class,),
                    fill_value=mode_id,
                    device=self.labels.device,
                    dtype=torch.long,
                )
            )
        return torch.cat(samples, dim=0), torch.cat(labels, dim=0)

    def stratified_class_batch(
        self,
        *,
        batch_size_per_class: int,
        start_index: int,
    ) -> tuple[Float[Tensor, "batch data_dim"], Int[Tensor, "batch"]]:
        samples = []
        labels = []
        for mode_id in range(self.num_classes):
            samples.append(
                self.class_batch(
                    mode_id=mode_id,
                    batch_size=batch_size_per_class,
                    start_index=start_index,
                )
            )
            labels.append(
                torch.full(
                    (batch_size_per_class,),
                    fill_value=mode_id,
                    device=self.labels.device,
                    dtype=torch.long,
                )
            )
        return torch.cat(samples, dim=0), torch.cat(labels, dim=0)

    def sample_unconditional(
        self,
        *,
        batch_size: int,
    ) -> Float[Tensor, "batch data_dim"]:
        sampled_indices = torch.randint(
            self.samples.shape[0],
            size=(batch_size,),
            device=self.samples.device,
        )
        return self.samples[sampled_indices]

    def as_images(
        self,
        samples: Float[Tensor, "batch data_dim"],
    ) -> Float[Tensor, "batch height width"]:
        if samples.ndim != 2:
            raise ValueError("samples must have shape [batch, data_dim]")
        if samples.shape[-1] != MNIST_NUMEL:
            raise ValueError("samples must have shape [batch, 784]")
        return samples.reshape(samples.shape[0], MNIST_SIDE_LENGTH, MNIST_SIDE_LENGTH)

    def log_likelihood(
        self,
        samples: Float[Tensor, "batch data_dim"],
    ) -> Float[Tensor, "batch"]:
        raise NotImplementedError(
            "MNISTDataConfig.log_likelihood is not implemented for the empirical data distribution"
        )


__all__ = ["MNISTDataConfig"]
