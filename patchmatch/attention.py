from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        d_v: int,
        n_head: int,
        dropout: float,
        d_qk: int,
        attention_temperature: float = 1.0,
        attention_top_k: int | None = None,
        attention_selection: str = "softmax",
        attention_gumbel_tau: float = 1.0,
        attention_gumbel_hard: bool = True,
        attention_softmax_straight_through: bool = True,
    ):
        super().__init__()
        self.d_v = int(d_v)
        self.n_head = int(n_head)
        self.d_k = int(d_qk)
        self.dropout = nn.Dropout(float(dropout))
        self.attention_temperature = float(attention_temperature)
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive.")
        self.attention_top_k = None if attention_top_k is None else int(attention_top_k)
        if self.attention_top_k is not None and self.attention_top_k <= 0:
            self.attention_top_k = None
        self.attention_selection = str(attention_selection).lower()
        if self.attention_selection not in {"softmax", "gumbel", "argmax"}:
            raise ValueError(
                "attention_selection must be one of {'softmax', 'gumbel', 'argmax'}."
            )
        self.attention_gumbel_tau = float(attention_gumbel_tau)
        if self.attention_gumbel_tau <= 0:
            raise ValueError("attention_gumbel_tau must be positive.")
        self.attention_gumbel_hard = bool(attention_gumbel_hard)
        self.attention_softmax_straight_through = bool(attention_softmax_straight_through)
        self.w_qs = nn.Linear(embed_dim, self.n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(embed_dim, self.n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(self.d_v, self.n_head * self.d_v, bias=False)
        self.fc = nn.Linear(self.n_head * self.d_v, self.d_v, bias=False)

    def _build_masked_query_selector(
        self,
        attn_logits: torch.Tensor,
        query_mask_flat: torch.Tensor | None,
    ) -> torch.Tensor:
        if query_mask_flat is None:
            return torch.ones(
                (attn_logits.shape[0], 1, attn_logits.shape[2], 1),
                dtype=torch.bool,
                device=attn_logits.device,
            )
        return (query_mask_flat > 0.5).unsqueeze(1).unsqueeze(-1)

    def _restrict_attention_logits(
        self,
        attn_logits: torch.Tensor,
        query_mask_flat: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.attention_top_k is None or self.attention_top_k >= attn_logits.shape[-1]:
            return attn_logits

        top_k = min(self.attention_top_k, attn_logits.shape[-1])
        _, topk_indices = torch.topk(attn_logits, k=top_k, dim=-1)
        keep_mask = torch.zeros_like(attn_logits, dtype=torch.bool)
        keep_mask = keep_mask.scatter(
            -1,
            topk_indices,
            torch.ones_like(topk_indices, dtype=torch.bool),
        )

        masked_queries = self._build_masked_query_selector(attn_logits, query_mask_flat)
        keep_mask = torch.where(masked_queries, keep_mask, torch.ones_like(keep_mask))
        return attn_logits.masked_fill(~keep_mask, torch.finfo(attn_logits.dtype).min)

    def _hard_attention_from_logits(self, attn_logits: torch.Tensor) -> torch.Tensor:
        attn = torch.zeros_like(attn_logits)
        top_indices = attn_logits.argmax(dim=-1, keepdim=True)
        attn = attn.scatter(-1, top_indices, torch.ones_like(top_indices, dtype=attn.dtype))
        return attn

    def _normalize_attention_logits(self, attn_logits: torch.Tensor) -> torch.Tensor:
        if self.attention_selection == "softmax":
            return F.softmax(attn_logits, dim=-1)
        if self.attention_selection == "gumbel":
            if self.training:
                return F.gumbel_softmax(
                    attn_logits,
                    tau=self.attention_gumbel_tau,
                    hard=self.attention_gumbel_hard,
                    dim=-1,
                )
            return self._hard_attention_from_logits(attn_logits)
        return self._hard_attention_from_logits(attn_logits)

    def compute_attention_logits(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        post_softmax_mask: torch.Tensor | None = None,
        query_mask_flat: torch.Tensor | None = None,
        logit_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, len_q, len_k = q.size(0), q.size(1), k.size(1)
        q_proj = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k).transpose(1, 2)
        k_proj = self.w_ks(k).view(batch_size, len_k, self.n_head, self.d_k).transpose(1, 2)
        
        q_proj = F.normalize(q_proj.float(), p=2, dim=-1).to(q_proj.dtype)
        k_proj = F.normalize(k_proj.float(), p=2, dim=-1).to(k_proj.dtype)
        
        attn_logits_raw = torch.matmul(q_proj, k_proj.transpose(2, 3)).float()
        if self.attention_temperature != 1.0:
            attn_logits_raw = attn_logits_raw / self.attention_temperature
        if logit_bias is not None:
            if logit_bias.dim() == 3:
                logit_bias = logit_bias.unsqueeze(1)
            attn_logits_raw = attn_logits_raw + logit_bias.to(dtype=attn_logits_raw.dtype)
        if post_softmax_mask is not None:
            attn_logits_raw = attn_logits_raw.masked_fill(
                post_softmax_mask == 0,
                torch.finfo(attn_logits_raw.dtype).min,
            )
        attn_logits = self._restrict_attention_logits(attn_logits_raw, query_mask_flat)
        return attn_logits_raw, attn_logits

    def attention_from_logits(
        self,
        attn_logits: torch.Tensor,
        *,
        value_dtype: torch.dtype,
        direct_patch_mixing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_probs = self._normalize_attention_logits(attn_logits).to(value_dtype)
        attn = attn_probs
        if direct_patch_mixing and self.training:
            needs_straight_through_hardening = (
                (self.attention_selection == "softmax" and self.attention_softmax_straight_through)
                or (self.attention_selection == "gumbel" and not self.attention_gumbel_hard)
            )
            if needs_straight_through_hardening:
                hard_attn = self._hard_attention_from_logits(attn_logits).to(value_dtype)
                attn = hard_attn - attn_probs.detach() + attn_probs
        if not direct_patch_mixing:
            attn = self.dropout(attn)
        return attn, attn_probs

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        post_softmax_mask: torch.Tensor | None = None,
        direct_patch_mixing: bool = False,
        query_mask_flat: torch.Tensor | None = None,
        logit_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        if direct_patch_mixing:
            if self.n_head != 1:
                raise ValueError("direct patch mixing expects a single attention head.")
            v_proj = v.unsqueeze(1)
        else:
            v_proj = self.w_vs(v).view(batch_size, len_v, self.n_head, self.d_v).transpose(1, 2)

        _, attn_logits = self.compute_attention_logits(
            q,
            k,
            post_softmax_mask=post_softmax_mask,
            query_mask_flat=query_mask_flat,
            logit_bias=logit_bias,
        )
        attn, attn_probs = self.attention_from_logits(
            attn_logits,
            value_dtype=v.dtype,
            direct_patch_mixing=direct_patch_mixing,
        )

        mixed = torch.matmul(attn, v_proj)
        output = mixed.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        if not direct_patch_mixing:
            output = self.fc(output)
        # For direct patch mixing, downstream consumers (e.g. HR upscaling) must reuse
        # the exact weights used to form `mixed`, not a softer probability variant.
        returned_attention = attn if direct_patch_mixing else attn_probs
        return output, returned_attention
