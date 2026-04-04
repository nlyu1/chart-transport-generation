from __future__ import annotations

import torch

from src.utils import clip_norm


def test_clip_norm_preserves_vectors_below_cap() -> None:
    tensor = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    clipped = clip_norm(tensor, max_norm=5.0)
    torch.testing.assert_close(clipped, tensor)


def test_clip_norm_caps_each_sample_norm() -> None:
    tensor = torch.tensor([[6.0, 8.0], [0.0, 10.0]])
    clipped = clip_norm(tensor, max_norm=5.0)
    expected = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    torch.testing.assert_close(clipped, expected)


def test_clip_norm_handles_zero_vectors() -> None:
    tensor = torch.zeros(2, 3)
    clipped = clip_norm(tensor, max_norm=5.0)
    torch.testing.assert_close(clipped, tensor)
