from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .transport import harden_transport_plan, solve_capacity_transport


class LightweightContextBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            padding_mode="reflect",
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.pointwise(x)
        x = self.activation(x)
        return residual + x


class LightweightContextEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden_channels = max(out_channels, 32)
        self.stem = nn.Sequential(
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
        )
        self.blocks = nn.Sequential(
            LightweightContextBlock(hidden_channels, dilation=1),
            LightweightContextBlock(hidden_channels, dilation=2),
            LightweightContextBlock(hidden_channels, dilation=4),
        )
        self.proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.proj(x)


class PatchmatchHelpersMixin:
    def _apply_attention_mode(self):
        self.multihead_attention.attention_selection = (
            self.attention_selection if self.training else self.attention_eval_selection
        )
        self.multihead_attention.attention_top_k = self.attention_top_k

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
        return LightweightContextEncoder(in_channels, out_channels)

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
            LightweightContextBlock(hidden_dim, dilation=2),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1, stride=1, bias=False),
        )

    def build_attention_supervision_entries(
        self,
        query_tokens: torch.Tensor,
        key_tokens: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> list[dict[str, torch.Tensor | tuple[int, int]]]:
        supervision_entries: list[dict[str, torch.Tensor | tuple[int, int]]] = []
        valid_key_mask = key_valid_flat > 0.5
        masked_query_mask = query_mask_flat > 0.5

        for batch_idx in range(query_tokens.shape[0]):
            query_indices = masked_query_mask[batch_idx].nonzero(as_tuple=False).flatten()
            key_indices = valid_key_mask[batch_idx].nonzero(as_tuple=False).flatten()
            entry: dict[str, torch.Tensor | tuple[int, int]] = {
                "query_indices": query_indices,
                "key_indices": key_indices,
                "token_hw": token_hw,
            }
            if query_indices.numel() == 0 or key_indices.numel() == 0:
                entry["raw_logits"] = query_tokens.new_empty((0, 0), dtype=torch.float32)
                supervision_entries.append(entry)
                continue

            raw_logits, _ = self.multihead_attention.compute_attention_logits(
                query_tokens[batch_idx : batch_idx + 1, query_indices],
                key_tokens[batch_idx : batch_idx + 1, key_indices],
            )
            entry["raw_logits"] = raw_logits.mean(dim=1).squeeze(0)
            supervision_entries.append(entry)

        return supervision_entries

    def _normalized_token_coords(
        self,
        indices: torch.Tensor,
        token_hw: tuple[int, int],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = token_hw
        ys = torch.div(indices, width, rounding_mode="floor").to(dtype=dtype)
        xs = (indices % width).to(dtype=dtype)
        if width > 1:
            xs = xs / float(width - 1)
        if height > 1:
            ys = ys / float(height - 1)
        return torch.stack([xs, ys], dim=-1)

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
        mixed_default = patch_values if default_tokens is None else default_tokens
        mixed_default = mixed_default.clone()
        eye = torch.eye(num_patches, device=patch_values.device, dtype=patch_values.dtype)
        dense_attn = eye.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        masked_queries = query_mask_flat > 0.5
        valid_keys = key_valid_flat > 0.5
        has_valid_keys = valid_keys.any(dim=1, keepdim=True)
        safe_valid_keys = torch.where(has_valid_keys, valid_keys, torch.ones_like(valid_keys))
        key_mask = safe_valid_keys.to(dtype=query_tokens_full.dtype).unsqueeze(1).unsqueeze(1)

        mixed_all, probs_all = self.multihead_attention(
            query_tokens_full,
            key_tokens_full,
            patch_values,
            post_softmax_mask=key_mask,
            direct_patch_mixing=True,
            query_mask_flat=query_mask_flat,
        )
        mixed_all = mixed_all.to(dtype=mixed_default.dtype)
        probs_all = probs_all.to(dtype=dense_attn.dtype)

        active_queries = masked_queries & has_valid_keys
        mixed = torch.where(active_queries.unsqueeze(-1), mixed_all, mixed_default)

        dense_attn = torch.where(active_queries.unsqueeze(1).unsqueeze(-1), probs_all, dense_attn)
        empty_key_queries = masked_queries & (~has_valid_keys)
        dense_attn = torch.where(
            empty_key_queries.unsqueeze(1).unsqueeze(-1),
            torch.zeros_like(dense_attn),
            dense_attn,
        )
        return mixed, dense_attn

    def transport_patch_mix_masked_queries(
        self,
        query_tokens_full: torch.Tensor,
        key_tokens_full: torch.Tensor,
        patch_values: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        *,
        token_hw: tuple[int, int] | None = None,
        default_tokens: torch.Tensor | None = None,
        return_aux_entries: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor | tuple[int, int]]]]:
        batch_size, num_patches, _ = query_tokens_full.shape
        mixed_default = patch_values if default_tokens is None else default_tokens
        mixed_default = mixed_default.clone()
        eye = torch.eye(num_patches, device=patch_values.device, dtype=patch_values.dtype)
        dense_attn = eye.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        masked_queries = query_mask_flat > 0.5
        valid_keys = key_valid_flat > 0.5
        entries: list[dict[str, torch.Tensor | tuple[int, int]]] = []

        for batch_idx in range(batch_size):
            query_indices = masked_queries[batch_idx].nonzero(as_tuple=False).flatten()
            key_indices = valid_keys[batch_idx].nonzero(as_tuple=False).flatten()
            if query_indices.numel() == 0 or key_indices.numel() == 0:
                if return_aux_entries:
                    entries.append(
                        {
                            "query_indices": query_indices,
                            "key_indices": key_indices,
                            "token_hw": token_hw,
                            "raw_logits": query_tokens_full.new_empty((0, 0), dtype=torch.float32),
                            "ranking_scores": query_tokens_full.new_empty((0, 0), dtype=torch.float32),
                            "pred_probs": query_tokens_full.new_empty((0, 0), dtype=torch.float32),
                            "pred_log_probs": query_tokens_full.new_empty((0, 0), dtype=torch.float32),
                        }
                    )
                continue

            raw_logits, masked_logits = self.multihead_attention.compute_attention_logits(
                query_tokens_full[batch_idx : batch_idx + 1, query_indices],
                key_tokens_full[batch_idx : batch_idx + 1, key_indices],
                query_mask_flat=torch.ones(
                    (1, query_indices.numel()),
                    device=query_tokens_full.device,
                    dtype=query_mask_flat.dtype,
                ),
            )
            raw_logits = raw_logits.mean(dim=1).squeeze(0).float()
            masked_logits = masked_logits.mean(dim=1).squeeze(0).float()
            transport_plan = solve_capacity_transport(
                masked_logits,
                epsilon=self.transport_epsilon,
                num_iters=self.transport_iters,
                capacity_scale=self.transport_capacity_scale,
            ).to(dtype=patch_values.dtype)

            if self.training and self.transport_train_hard:
                hard_plan = harden_transport_plan(transport_plan)
                mix_plan = hard_plan - transport_plan.detach() + transport_plan
                ranking_plan = hard_plan
            elif (not self.training) and self.transport_eval_hard:
                mix_plan = harden_transport_plan(transport_plan)
                ranking_plan = mix_plan
            else:
                mix_plan = transport_plan
                ranking_plan = transport_plan

            mixed_subset = mix_plan @ patch_values[batch_idx, key_indices]
            mixed_subset = mixed_subset.to(dtype=mixed_default.dtype)
            mixed_default[batch_idx, query_indices] = mixed_subset

            dense_attn_batch = dense_attn[batch_idx, 0]
            dense_attn_batch[query_indices] = 0
            dense_attn_batch[query_indices.unsqueeze(1), key_indices.unsqueeze(0)] = mix_plan.to(
                dtype=dense_attn_batch.dtype
            )

            if return_aux_entries:
                entries.append(
                    {
                        "query_indices": query_indices,
                        "key_indices": key_indices,
                        "token_hw": token_hw,
                        "raw_logits": raw_logits,
                        "ranking_scores": ranking_plan.clamp_min(1e-8).log().float(),
                        "pred_probs": transport_plan.float(),
                        "pred_log_probs": transport_plan.clamp_min(1e-8).log().float(),
                    }
                )

        if return_aux_entries:
            return mixed_default, dense_attn, entries
        return mixed_default, dense_attn

    def build_attention_mask(
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
