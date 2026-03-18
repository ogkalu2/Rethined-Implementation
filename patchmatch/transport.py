from __future__ import annotations

import torch


def solve_capacity_transport(
    logits: torch.Tensor,
    *,
    epsilon: float = 0.05,
    num_iters: int = 12,
    capacity_scale: float = 1.25,
) -> torch.Tensor:
    """Approximate a row-stochastic transport plan with soft column capacities.

    Rows are normalized to sum to one. Columns are clipped toward a soft capacity
    budget, which discourages hub patches from absorbing most of the mass while
    still allowing reuse when needed.
    """
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits [queries, keys], got {tuple(logits.shape)}")
    if logits.numel() == 0:
        return logits.new_zeros(logits.shape)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive.")
    if capacity_scale <= 0:
        raise ValueError("capacity_scale must be positive.")

    num_queries, num_keys = logits.shape
    scaled_logits = logits.float() / epsilon
    scaled_logits = scaled_logits - scaled_logits.amax(dim=-1, keepdim=True)
    plan = torch.exp(scaled_logits).clamp_min(1e-8)
    plan = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    avg_capacity = capacity_scale * float(num_queries) / float(max(num_keys, 1))
    capacities = plan.new_full((num_keys,), avg_capacity)

    for _ in range(num_iters):
        plan = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        col_mass = plan.sum(dim=0)
        col_scale = torch.minimum(
            torch.ones_like(col_mass),
            capacities / col_mass.clamp_min(1e-8),
        )
        plan = plan * col_scale.unsqueeze(0)

    return plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def harden_transport_plan(plan: torch.Tensor) -> torch.Tensor:
    """Convert a soft row-stochastic plan into a one-hot per-row assignment."""
    if plan.dim() != 2:
        raise ValueError(f"Expected 2D plan [queries, keys], got {tuple(plan.shape)}")
    if plan.numel() == 0:
        return torch.zeros_like(plan)
    indices = plan.argmax(dim=-1, keepdim=True)
    return torch.zeros_like(plan).scatter(-1, indices, 1.0)
