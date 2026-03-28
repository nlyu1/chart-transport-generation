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

    def log_likelihood(
        self,
        samples: Float[Tensor, "batch data_dim"],
    ) -> Float[Tensor, "batch"]:
        raise NotImplementedError(
            "MNISTDataConfig.log_likelihood is not implemented for the empirical data distribution"
        )


__all__ = ["MNISTDataConfig"]
