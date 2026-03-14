from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchmatchHelpersMixin:
    def _apply_attention_schedule(self):
        if not self.training:
            self.multihead_attention.attention_selection = self.base_attention_selection
            self.multihead_attention.attention_gumbel_hard = True
            return

        if (
            self.attention_warmup_selection is not None
            and self.current_training_step < self.attention_warmup_steps
        ):
            self.multihead_attention.attention_selection = self.attention_warmup_selection
            self.multihead_attention.attention_gumbel_hard = False
            return

        self.multihead_attention.attention_selection = self.base_attention_selection
        if self.base_attention_selection == "gumbel":
            self.multihead_attention.attention_gumbel_hard = (
                self.current_training_step >= self.attention_gumbel_hard_start_step
            )
        else:
            self.multihead_attention.attention_gumbel_hard = True

    def _apply_branch_dropout(self, branch: torch.Tensor, drop_prob: float) -> torch.Tensor:
        if (not self.training) or drop_prob <= 0:
            return branch
        keep_prob = 1.0 - drop_prob
        mask = branch.new_empty((branch.shape[0], 1, 1, 1)).bernoulli_(keep_prob)
        mask = mask / max(keep_prob, 1e-8)
        return branch * mask

    def _prepare_matching_branch(
        self,
        branch: torch.Tensor,
        *,
        drop_prob: float,
    ) -> torch.Tensor:
        return self._apply_branch_dropout(branch, drop_prob)

    def _build_context_encoder(self, in_channels: int, out_channels: int) -> nn.Sequential:
        hidden_channels = max(out_channels, 32)
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.GELU(),
        )

    def _build_projection_head(self, input_dim: int, output_dim: int) -> nn.Sequential:
        hidden_dim = output_dim
        return nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1, stride=1, bias=False),
        )

    def _build_matching_descriptor_head(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
    ) -> nn.Sequential:
        hidden_dim = max(input_dim, output_dim) if hidden_dim is None else int(hidden_dim)
        return nn.Sequential(
            nn.Conv2d(
                input_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1, stride=1, bias=False),
        )

    def direct_patch_mix_masked_queries(
        self,
        query_tokens_full: torch.Tensor,
        key_tokens_full: torch.Tensor,
        patch_values: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        default_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_patches, _ = query_tokens_full.shape
        mixed = patch_values.clone() if default_tokens is None else default_tokens.clone()
        eye = torch.eye(num_patches, device=patch_values.device, dtype=patch_values.dtype)
        dense_attn = eye.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        masked_queries = query_mask_flat > 0.5
        valid_keys = key_valid_flat > 0.5

        for batch_idx in range(batch_size):
            query_indices = masked_queries[batch_idx].nonzero(as_tuple=False).flatten()
            if query_indices.numel() == 0:
                continue

            key_indices = valid_keys[batch_idx].nonzero(as_tuple=False).flatten()
            replacement_rows = patch_values.new_zeros((query_indices.numel(), num_patches))
            if key_indices.numel() > 0:
                query_tokens = query_tokens_full[batch_idx : batch_idx + 1, query_indices]
                key_tokens = key_tokens_full[batch_idx : batch_idx + 1, key_indices]
                value_tokens = patch_values[batch_idx : batch_idx + 1, key_indices]
                mixed_queries, masked_attention = self.multihead_attention(
                    query_tokens,
                    key_tokens,
                    value_tokens,
                    direct_patch_mixing=True,
                )
                mixed_queries = mixed_queries.squeeze(0).to(dtype=mixed.dtype)
                masked_attention = masked_attention.squeeze(0).squeeze(0).to(dtype=replacement_rows.dtype)
                replacement_rows[:, key_indices] = masked_attention
                mixed[batch_idx].index_copy_(0, query_indices, mixed_queries)

            dense_attn[batch_idx, 0].index_copy_(0, query_indices, replacement_rows)

        return mixed, dense_attn

    def build_paper_attention_mask(
        self,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, num_patches = query_mask_flat.shape
        is_masked_q = query_mask_flat.unsqueeze(-1)
        if key_valid_flat is None:
            key_valid_flat = 1.0 - query_mask_flat
        is_valid_k = key_valid_flat.unsqueeze(1)
        eye = torch.eye(num_patches, device=query_mask_flat.device, dtype=query_mask_flat.dtype).unsqueeze(0)
        allowed = (1.0 - is_masked_q) * eye + is_masked_q * is_valid_k
        return allowed.unsqueeze(1)

    def summarize_attention(
        self,
        attn_map: torch.Tensor,
        query_mask_flat: torch.Tensor,
    ) -> dict[str, float]:
        probs = attn_map.detach()
        if probs.dim() == 4:
            probs = probs.mean(dim=1)
        if probs.dim() != 3:
            raise ValueError(f"Expected attention map with 3 or 4 dims, got {tuple(attn_map.shape)}")

        masked_queries = query_mask_flat > 0.5
        masked_query_ratio = masked_queries.float().mean().item()
        if not masked_queries.any():
            return {
                "attention_top1": 1.0,
                "attention_top4": 1.0,
                "attention_entropy": 0.0,
                "attention_masked_ratio": masked_query_ratio,
            }

        masked_probs = probs[masked_queries]
        top1 = masked_probs.max(dim=-1).values.mean().item()
        top4_k = min(4, masked_probs.shape[-1])
        top4 = masked_probs.topk(k=top4_k, dim=-1).values.sum(dim=-1).mean().item()
        entropy = (
            -(masked_probs.clamp_min(1e-8) * masked_probs.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
        )
        return {
            "attention_top1": top1,
            "attention_top4": top4,
            "attention_entropy": entropy,
            "attention_masked_ratio": masked_query_ratio,
        }

    def get_positional_encoding(self) -> torch.Tensor | None:
        if self.positionalencoding is None:
            return None
        if self.positionalencoding.shape[-2:] != (self.token_grid_size, self.token_grid_size):
            pos = F.interpolate(
                self.positionalencoding,
                size=(self.token_grid_size, self.token_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            pos = self.positionalencoding
        return pos.flatten(start_dim=2).transpose(1, 2)

    def reparameterize(self):
        return self
