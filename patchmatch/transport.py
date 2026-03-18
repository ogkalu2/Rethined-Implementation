from __future__ import annotations

import math

import torch


def solve_capacity_transport(
    logits: torch.Tensor,
    *,
    epsilon: float = 1.0,
    num_iters: int = 24,
    capacity_scale: float = 1.25,
    mass_penalty: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve a row-balanced, column-unbalanced transport problem in log space.

    Rows are treated as balanced because each masked query ultimately needs one
    donor. Columns are treated as unbalanced with a soft capacity budget scaled
    by ``capacity_scale * num_queries / num_keys`` so transport can discourage
    hub keys without forbidding valid reuse.

    Returns transport scores, log-probabilities, and probabilities. The scores
    are row-wise comparable and can be used for ranking and hardening, while the
    probabilities remain differentiable for supervision.
    """
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits [queries, keys], got {tuple(logits.shape)}")
    if logits.numel() == 0:
        empty = logits.new_zeros(logits.shape, dtype=torch.float32)
        return empty, empty, empty
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive.")
    if capacity_scale <= 0:
        raise ValueError("capacity_scale must be positive.")
    if mass_penalty <= 0:
        raise ValueError("mass_penalty must be positive.")

    num_queries, num_keys = logits.shape
    avg_capacity = max(capacity_scale * float(num_queries) / float(max(num_keys, 1)), 1e-6)

    logits_float = logits.float()
    finite_mask = torch.isfinite(logits_float)
    if not finite_mask.any():
        empty = logits.new_zeros(logits.shape, dtype=torch.float32)
        return empty, empty, empty

    centered_logits = logits_float - logits_float[finite_mask].max()
    neg_inf = torch.finfo(centered_logits.dtype).min
    log_kernel = torch.where(
        finite_mask,
        (centered_logits / epsilon).clamp(min=-60.0, max=0.0),
        torch.full_like(centered_logits, neg_inf),
    )

    log_a = logits.new_zeros((num_queries,), dtype=torch.float32)
    log_b = logits.new_full((num_keys,), math.log(avg_capacity), dtype=torch.float32)
    log_v = torch.zeros_like(log_b)
    tau_col = mass_penalty / (mass_penalty + epsilon)

    for _ in range(num_iters):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=-1)
        log_v = tau_col * (
            log_b
            - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
        )

    transport_scores = log_kernel + log_v.unsqueeze(0)
    transport_log_probs = transport_scores - torch.logsumexp(
        transport_scores,
        dim=-1,
        keepdim=True,
    )
    transport_probs = torch.exp(transport_log_probs)
    return transport_scores, transport_log_probs, transport_probs


def harden_transport_plan(plan: torch.Tensor) -> torch.Tensor:
    """Convert a soft row-stochastic plan into a one-hot per-row assignment."""
    if plan.dim() != 2:
        raise ValueError(f"Expected 2D plan [queries, keys], got {tuple(plan.shape)}")
    if plan.numel() == 0:
        return torch.zeros_like(plan)
    indices = plan.argmax(dim=-1, keepdim=True)
    return torch.zeros_like(plan).scatter(-1, indices, 1.0)
