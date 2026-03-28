from __future__ import annotations

import plotly.graph_objects as go
import torch
from jaxtyping import Float
from torch import Tensor

PRIOR_ARROW_HEAD_SCALE = 0.2


def plot_prior_with_scores(
    *,
    samples: Float[Tensor, "batch 2"],
    anchor_samples: Float[Tensor, "subset 2"],
    scores: Float[Tensor, "subset 2"],
    title: str,
    sample_trace_name: str,
    arrow_trace_name: str,
    sample_color: str,
    alpha: float,
    arrow_color: str,
    arrow_scale: float,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=samples[:, 0].tolist(),
            y=samples[:, 1].tolist(),
            mode="markers",
            marker=dict(size=7, color=sample_color, opacity=alpha),
            name=sample_trace_name,
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )

    score_norms = torch.linalg.vector_norm(scores, dim=-1)
    nonzero_mask = score_norms > 0.0
    if bool(nonzero_mask.any()):
        arrow_tails = anchor_samples[nonzero_mask].to(dtype=torch.float32, device="cpu")
        arrow_scores = scores[nonzero_mask].to(dtype=torch.float32, device="cpu")
        arrow_norms = score_norms[nonzero_mask].to(dtype=torch.float32, device="cpu")
        arrow_heads = arrow_tails + arrow_scale * arrow_scores

        arrow_x: list[float | None] = []
        arrow_y: list[float | None] = []
        for tail, head in zip(arrow_tails, arrow_heads, strict=True):
            tail_x = float(tail[0].item())
            tail_y = float(tail[1].item())
            head_x = float(head[0].item())
            head_y = float(head[1].item())
            delta_x = head_x - tail_x
            delta_y = head_y - tail_y
            left_x = head_x - PRIOR_ARROW_HEAD_SCALE * (delta_x + delta_y)
            left_y = head_y - PRIOR_ARROW_HEAD_SCALE * (delta_y - delta_x)
            right_x = head_x - PRIOR_ARROW_HEAD_SCALE * (delta_x - delta_y)
            right_y = head_y - PRIOR_ARROW_HEAD_SCALE * (delta_y + delta_x)
            arrow_x.extend([tail_x, head_x, None, left_x, head_x, right_x, None])
            arrow_y.extend([tail_y, head_y, None, left_y, head_y, right_y, None])

        figure.add_trace(
            go.Scatter(
                x=arrow_x,
                y=arrow_y,
                mode="lines",
                line=dict(width=1.5, color=arrow_color),
                name=arrow_trace_name,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=arrow_tails[:, 0].tolist(),
                y=arrow_tails[:, 1].tolist(),
                mode="markers",
                name=arrow_trace_name,
                showlegend=False,
                customdata=arrow_norms.unsqueeze(-1).tolist(),
                hovertemplate=(
                    "anchor=(%{x:.3f}, %{y:.3f})<br>"
                    "|analytic_score|=%{customdata[0]:.3f}<extra></extra>"
                ),
                marker=dict(size=8, opacity=0.0, color=arrow_color),
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title="y[0]",
        yaxis_title="y[1]",
        template="plotly_white",
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1.0)
    return figure


__all__ = ["PRIOR_ARROW_HEAD_SCALE", "plot_prior_with_scores"]
