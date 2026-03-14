"""Paper-aligned NeuralPatchMatch refinement modules."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from blocks import NativeGaussianBlur2d


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
        keep_mask.scatter_(-1, topk_indices, True)

        masked_queries = self._build_masked_query_selector(attn_logits, query_mask_flat)
        keep_mask = torch.where(masked_queries, keep_mask, torch.ones_like(keep_mask))
        return attn_logits.masked_fill(~keep_mask, torch.finfo(attn_logits.dtype).min)

    def _hard_attention_from_logits(self, attn_logits: torch.Tensor) -> torch.Tensor:
        attn = torch.zeros_like(attn_logits)
        top_indices = attn_logits.argmax(dim=-1, keepdim=True)
        attn.scatter_(-1, top_indices, 1.0)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, len_q, len_k = q.size(0), q.size(1), k.size(1)
        q_proj = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k).transpose(1, 2)
        k_proj = self.w_ks(k).view(batch_size, len_k, self.n_head, self.d_k).transpose(1, 2)
        attn_logits_raw = torch.matmul(q_proj / (self.d_k ** 0.5), k_proj.transpose(2, 3)).float()
        if self.attention_temperature != 1.0:
            attn_logits_raw = attn_logits_raw / self.attention_temperature
        if post_softmax_mask is not None:
            attn_logits_raw = attn_logits_raw.masked_fill(
                post_softmax_mask == 0,
                torch.finfo(attn_logits_raw.dtype).min,
            )
        attn_logits = self._restrict_attention_logits(attn_logits_raw, query_mask_flat)
        return attn_logits_raw, attn_logits

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        post_softmax_mask: torch.Tensor | None = None,
        direct_patch_mixing: bool = False,
        query_mask_flat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        if direct_patch_mixing:
            if self.n_head != 1:
                raise ValueError("Paper-style direct patch mixing expects a single attention head.")
            v_proj = v.unsqueeze(1)
        else:
            v_proj = self.w_vs(v).view(batch_size, len_v, self.n_head, self.d_v).transpose(1, 2)

        _, attn_logits = self.compute_attention_logits(
            q,
            k,
            post_softmax_mask=post_softmax_mask,
            query_mask_flat=query_mask_flat,
        )
        attn_probs = self._normalize_attention_logits(attn_logits).to(v.dtype)
        attn = attn_probs
        if direct_patch_mixing and self.training:
            # For direct RGB patch mixing, soft attention averages unrelated patches and
            # changes the task into patch blending. Use hard routing in the forward pass
            # while preserving soft gradients during warmup.
            needs_straight_through_hardening = (
                self.attention_selection == "softmax"
                or (self.attention_selection == "gumbel" and not self.attention_gumbel_hard)
            )
            if needs_straight_through_hardening:
                hard_attn = self._hard_attention_from_logits(attn_logits).to(v.dtype)
                attn = hard_attn - attn_probs.detach() + attn_probs
        # Direct patch mixing behaves like weighted image reconstruction, so dropping
        # sparse attention weights introduces random dark artifacts instead of useful regularization.
        if not direct_patch_mixing:
            attn = self.dropout(attn)

        mixed = torch.matmul(attn, v_proj)
        output = mixed.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        if not direct_patch_mixing:
            output = self.fc(output)
        return output, attn_probs


class PatchInpainting(nn.Module):
    def __init__(
        self,
        *,
        kernel_size: int,
        value_patch_size: int | None = None,
        attention_temperature: float = 1.0,
        attention_top_k: int | None = None,
        attention_selection: str = "softmax",
        attention_gumbel_tau: float = 1.0,
        attention_gumbel_hard: bool = True,
        attention_warmup_selection: str | None = None,
        attention_warmup_steps: int = 0,
        attention_gumbel_hard_start_step: int = 0,
        matching_descriptor_dim: int | None = None,
        match_coarse_rgb: bool = True,
        detach_coarse_rgb: bool = False,
        normalize_matching_branches: bool = False,
        learnable_matching_branch_scales: bool = False,
        coarse_rgb_branch_dropout: float = 0.0,
        candidate_reranker: bool = False,
        candidate_reranker_hidden_dim: int = 128,
        candidate_rerank_context_channels: int = 32,
        query_image_context_matching: bool = False,
        separate_query_key_matching: bool = False,
        shared_query_key_descriptor: bool = False,
        query_context_channels: int = 32,
        key_context_channels: int = 32,
        query_context_residual_init: float = 0.0,
        key_coarse_rgb_residual_init: float = 0.0,
        key_feature_residual_init: float = 0.0,
        value_source: str = "rgb",
        fusion_mode: str = "replace",
        fusion_hidden_channels: int = 32,
        nheads: int,
        stem_out_stride: int = 1,
        stem_out_channels: int = 3,
        use_positional_encoding: bool = True,
        image_size: int = 512,
        embed_dim: int = 2048,
        dropout: float = 0.1,
        feature_i: int = 2,
        feature_dim: int = 128,
        concat_features: bool = True,
        attention_masking: bool = True,
        final_conv: bool = False,
        positional_grid_size: int = 32,
        use_conv_unfold: bool = False,
        model,
    ):
        super().__init__()
        self.use_conv_unfold = use_conv_unfold
        self.kernel_size = int(kernel_size)
        self.value_patch_size = self.kernel_size if value_patch_size is None else int(value_patch_size)
        self.nheads = int(nheads)
        self.stem_out_stride = int(stem_out_stride)
        self.stem_out_channels = int(stem_out_channels)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.feature_i = int(feature_i)
        self.feature_dim = int(feature_dim)
        self.matching_descriptor_dim = (
            None if matching_descriptor_dim is None else int(matching_descriptor_dim)
        )
        self.match_coarse_rgb = bool(match_coarse_rgb)
        self.detach_coarse_rgb = bool(detach_coarse_rgb)
        self.normalize_matching_branches = bool(normalize_matching_branches)
        self.learnable_matching_branch_scales = bool(learnable_matching_branch_scales)
        self.coarse_rgb_branch_dropout = float(coarse_rgb_branch_dropout)
        self.candidate_reranker = bool(candidate_reranker)
        self.candidate_reranker_hidden_dim = int(candidate_reranker_hidden_dim)
        self.candidate_rerank_context_channels = int(candidate_rerank_context_channels)
        self.query_image_context_matching = bool(query_image_context_matching)
        self.separate_query_key_matching = bool(separate_query_key_matching)
        self.shared_query_key_descriptor = bool(shared_query_key_descriptor)
        self.query_context_channels = int(query_context_channels)
        self.key_context_channels = int(key_context_channels)
        self.query_context_residual_init = float(query_context_residual_init)
        self.key_coarse_rgb_residual_init = float(key_coarse_rgb_residual_init)
        self.key_feature_residual_init = float(key_feature_residual_init)
        if not 0.0 <= self.coarse_rgb_branch_dropout < 1.0:
            raise ValueError("coarse_rgb_branch_dropout must be in [0, 1).")
        if self.candidate_reranker_hidden_dim <= 0:
            raise ValueError("candidate_reranker_hidden_dim must be positive.")
        if self.candidate_rerank_context_channels <= 0:
            raise ValueError("candidate_rerank_context_channels must be positive.")
        if self.query_image_context_matching and self.separate_query_key_matching:
            raise ValueError(
                "query_image_context_matching and separate_query_key_matching cannot both be enabled."
            )
        if self.separate_query_key_matching:
            if self.query_context_channels <= 0:
                raise ValueError("query_context_channels must be positive when separate_query_key_matching=True.")
            if self.key_context_channels <= 0:
                raise ValueError("key_context_channels must be positive when separate_query_key_matching=True.")
            if self.shared_query_key_descriptor and self.query_context_channels != self.key_context_channels:
                raise ValueError(
                    "query_context_channels and key_context_channels must match when shared_query_key_descriptor=True."
                )
        elif self.query_image_context_matching and self.query_context_channels <= 0:
            raise ValueError(
                "query_context_channels must be positive when query_image_context_matching=True."
            )
        self.value_source = str(value_source).lower()
        if self.value_source not in {"rgb", "high_freq_residual"}:
            raise ValueError("value_source must be either 'rgb' or 'high_freq_residual'.")
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in {"replace", "add", "gate"}:
            raise ValueError("fusion_mode must be one of {'replace', 'add', 'gate'}.")
        self.fusion_hidden_channels = int(fusion_hidden_channels)
        if self.fusion_hidden_channels <= 0:
            raise ValueError("fusion_hidden_channels must be positive.")
        self.concat_features = bool(concat_features)
        self.attention_masking = bool(attention_masking)
        self.final_conv = bool(final_conv)
        self.image_size = int(image_size)
        if self.value_patch_size < self.kernel_size:
            raise ValueError("value_patch_size must be greater than or equal to kernel_size.")
        if (self.value_patch_size - self.kernel_size) % 2 != 0:
            raise ValueError("value_patch_size - kernel_size must be even for centered overlap-add padding.")
        self.value_patch_padding = (self.value_patch_size - self.kernel_size) // 2
        self.token_grid_size = self.image_size // self.stem_out_stride // self.kernel_size
        self.query_patch_dim = self.stem_out_channels * self.kernel_size * self.kernel_size
        self.value_patch_dim = self.stem_out_channels * self.value_patch_size * self.value_patch_size
        self.matching_input_dim = 0
        if self.match_coarse_rgb:
            self.matching_input_dim += self.query_patch_dim
        if self.concat_features:
            self.matching_input_dim += self.feature_dim
        if not self.separate_query_key_matching and self.matching_input_dim == 0:
            raise ValueError("Matching must use coarse RGB patches, coarse features, or both.")
        self.query_matching_input_dim = self.matching_input_dim
        self.key_matching_input_dim = self.matching_input_dim
        if self.separate_query_key_matching:
            self.query_matching_input_dim = self.query_patch_dim + self.query_context_channels
            if self.match_coarse_rgb:
                self.query_matching_input_dim += self.query_patch_dim
            if self.concat_features:
                self.query_matching_input_dim += self.feature_dim
            self.key_matching_input_dim = self.query_patch_dim + self.key_context_channels
            if self.match_coarse_rgb:
                self.key_matching_input_dim += self.query_patch_dim
            if self.concat_features:
                self.key_matching_input_dim += self.feature_dim
        self.patch_token_dim = (
            max(self.query_matching_input_dim, self.key_matching_input_dim)
            if self.matching_descriptor_dim is None
            else self.matching_descriptor_dim
        )
        self.positional_grid_size = max(1, min(int(positional_grid_size), self.token_grid_size))
        self.base_attention_selection = str(attention_selection).lower()
        self.attention_warmup_selection = (
            None if attention_warmup_selection is None else str(attention_warmup_selection).lower()
        )
        if self.attention_warmup_selection not in {None, "softmax", "gumbel", "argmax"}:
            raise ValueError(
                "attention_warmup_selection must be one of {None, 'softmax', 'gumbel', 'argmax'}."
            )
        self.attention_warmup_steps = max(0, int(attention_warmup_steps))
        self.attention_gumbel_hard_start_step = max(0, int(attention_gumbel_hard_start_step))
        self.current_training_step = 0

        self.encoder_decoder = model
        self.final_gaussian_blur = NativeGaussianBlur2d((7, 7), sigma=(2.01, 2.01))
        self.query_context_encoder = None
        self.key_context_encoder = None
        self.query_context_descriptor_head = None
        self.query_context_scale = None
        self.key_coarse_rgb_scale = None
        self.key_feature_scale = None
        self.shared_query_key_descriptor_head = None
        self.candidate_reranker_head = None
        self.candidate_query_context_encoder = None
        self.candidate_key_context_encoder = None
        self.coarse_rgb_branch_norm = None
        if self.match_coarse_rgb and self.normalize_matching_branches:
            self.coarse_rgb_branch_norm = nn.GroupNorm(1, self.query_patch_dim)
        self.feature_branch_norm = None
        if self.concat_features and self.normalize_matching_branches:
            self.feature_branch_norm = nn.GroupNorm(1, self.feature_dim)
        self.coarse_rgb_branch_scale = None
        if self.match_coarse_rgb and self.learnable_matching_branch_scales:
            self.coarse_rgb_branch_scale = nn.Parameter(torch.tensor(1.0))
        self.feature_branch_scale = None
        if self.concat_features and self.learnable_matching_branch_scales:
            self.feature_branch_scale = nn.Parameter(torch.tensor(1.0))
        self.query_descriptor_head = None
        self.key_descriptor_head = None
        self.matching_descriptor_head = None
        if self.separate_query_key_matching:
            self.query_context_encoder = self._build_context_encoder(4, self.query_context_channels)
            self.key_context_encoder = self._build_context_encoder(3, self.key_context_channels)
            if self.match_coarse_rgb:
                self.key_coarse_rgb_scale = nn.Parameter(torch.tensor(self.key_coarse_rgb_residual_init))
            if self.concat_features:
                self.key_feature_scale = nn.Parameter(torch.tensor(self.key_feature_residual_init))
            if self.shared_query_key_descriptor:
                self.shared_query_key_descriptor_head = self._build_matching_descriptor_head(
                    self.query_matching_input_dim,
                    self.patch_token_dim,
                    hidden_dim=max(self.matching_input_dim, self.patch_token_dim),
                )
            else:
                self.query_descriptor_head = self._build_matching_descriptor_head(
                    self.query_matching_input_dim,
                    self.patch_token_dim,
                )
                self.key_descriptor_head = self._build_matching_descriptor_head(
                    self.key_matching_input_dim,
                    self.patch_token_dim,
                )
        elif self.matching_descriptor_dim is not None:
            self.matching_descriptor_head = self._build_matching_descriptor_head(
                self.matching_input_dim,
                self.matching_descriptor_dim,
            )
        if self.query_image_context_matching:
            self.query_context_encoder = self._build_context_encoder(4, self.query_context_channels)
            self.query_context_descriptor_head = self._build_projection_head(
                self.query_context_channels + self.query_patch_dim,
                self.patch_token_dim,
            )
            self.query_context_scale = nn.Parameter(torch.tensor(self.query_context_residual_init))
        self.multihead_attention = MultiHeadAttention(
            embed_dim=self.patch_token_dim,
            d_v=self.value_patch_dim,
            n_head=self.nheads,
            dropout=float(dropout),
            d_qk=int(embed_dim),
            attention_temperature=float(attention_temperature),
            attention_top_k=attention_top_k,
            attention_selection=attention_selection,
            attention_gumbel_tau=float(attention_gumbel_tau),
            attention_gumbel_hard=attention_gumbel_hard,
        )
        if self.candidate_reranker:
            if self.multihead_attention.attention_top_k is None:
                raise ValueError("candidate_reranker requires attention_top_k to be set.")
            if not self.attention_masking:
                raise ValueError("candidate_reranker currently requires attention_masking=True.")
            self.candidate_query_context_encoder = self._build_context_encoder(
                7,
                self.candidate_rerank_context_channels,
            )
            self.candidate_key_context_encoder = self._build_context_encoder(
                3,
                self.candidate_rerank_context_channels,
            )
            rerank_input_dim = (self.patch_token_dim * 4) + (self.candidate_rerank_context_channels * 4)
            self.candidate_reranker_head = nn.Sequential(
                nn.Linear(rerank_input_dim, self.candidate_reranker_hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(self.candidate_reranker_hidden_dim, 1),
            )
            nn.init.zeros_(self.candidate_reranker_head[-1].weight)
            nn.init.zeros_(self.candidate_reranker_head[-1].bias)
        self.pre_attention_norm = nn.LayerNorm(self.patch_token_dim)
        self.positionalencoding = (
            nn.Parameter(
                torch.zeros(
                    1,
                    self.patch_token_dim,
                    self.positional_grid_size,
                    self.positional_grid_size,
                )
            )
            if self.use_positional_encoding
            else None
        )
        self.fusion_gate = None
        if self.fusion_mode == "gate":
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(
                    10,
                    self.fusion_hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="reflect",
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.fusion_hidden_channels,
                    1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="reflect",
                ),
            )
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.constant_(self.fusion_gate[-1].bias, -2.0)
        self.coherence_layer = (
            nn.Conv2d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
            if self.final_conv
            else None
        )
        if self.coherence_layer is not None:
            nn.init.zeros_(self.coherence_layer.weight)
            nn.init.zeros_(self.coherence_layer.bias)

        self.register_buffer(
            "unfolding_weights",
            self._compute_unfolding_weights(self.kernel_size, self.stem_out_channels),
            persistent=False,
        )
        self.register_buffer(
            "unfolding_weights_image",
            self._compute_unfolding_weights(self.kernel_size, 3),
            persistent=False,
        )
        self.register_buffer(
            "unfolding_weights_mask",
            self._compute_unfolding_weights(self.kernel_size, 1),
            persistent=False,
        )
        self._apply_attention_schedule()

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_attention_schedule()
        return self

    def set_training_step(self, step: int):
        self.current_training_step = max(0, int(step))
        self._apply_attention_schedule()

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
        norm: nn.Module | None,
        scale: torch.Tensor | None,
        drop_prob: float,
    ) -> torch.Tensor:
        if norm is not None:
            branch = norm(branch)
        if scale is not None:
            branch = branch * scale.to(dtype=branch.dtype, device=branch.device)
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

    def _pool_to_token_grid(self, feature_map: torch.Tensor, token_hw: tuple[int, int]) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feature_map, token_hw)

    def _compute_unfolding_weights(self, kernel_size: int, channels: int) -> torch.Tensor:
        weights = torch.eye(kernel_size * kernel_size, dtype=torch.float32)
        weights = weights.view(kernel_size * kernel_size, 1, kernel_size, kernel_size)
        return weights.repeat(channels, 1, 1, 1)

    def _get_unfolding_weights(
        self,
        kernel_size: int,
        channels: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if kernel_size == self.kernel_size:
            if channels == self.stem_out_channels:
                weights = self.unfolding_weights
            elif channels == 3:
                weights = self.unfolding_weights_image
            elif channels == 1:
                weights = self.unfolding_weights_mask
            else:
                weights = self._compute_unfolding_weights(kernel_size, channels)
        else:
            weights = self._compute_unfolding_weights(kernel_size, channels)
        return weights.to(device=device, dtype=dtype)

    def unfold_native(self, feature_map: torch.Tensor, kernel_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
        return self.extract_patches(feature_map, kernel_size, stride=kernel_size, padding=0)

    def extract_patches(
        self,
        feature_map: torch.Tensor,
        patch_size: int,
        *,
        stride: int,
        padding: int = 0,
        pad_mode: str = "reflect",
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        batch_size, channels, height, width = feature_map.shape
        patch_size = int(patch_size)
        stride = int(stride)
        padding = int(padding)
        if height % stride != 0 or width % stride != 0:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by stride {stride} for patch extraction."
            )

        n_h = height // stride
        n_w = width // stride

        if patch_size == stride and padding == 0:
            if self.use_conv_unfold:
                weights = self._get_unfolding_weights(
                    patch_size, channels, dtype=feature_map.dtype, device=feature_map.device
                )
                patches = F.conv2d(feature_map, weights, stride=patch_size, groups=channels)
                patches = patches.view(batch_size, channels * patch_size * patch_size, n_h, n_w)
            else:
                x = feature_map.view(batch_size, channels, n_h, patch_size, n_w, patch_size)
                x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
                patches = x.view(batch_size, channels * patch_size * patch_size, n_h, n_w)
            return patches, (height, width)

        if padding > 0:
            feature_map = F.pad(feature_map, (padding, padding, padding, padding), mode=pad_mode)

        patches = F.unfold(feature_map, kernel_size=patch_size, stride=stride)
        patches = patches.view(batch_size, channels * patch_size * patch_size, n_h, n_w)

        return patches, (height, width)

    def fold_native(
        self,
        patches: torch.Tensor,
        output_size: tuple[int, int],
        *,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        use_window: bool = False,
    ) -> torch.Tensor:
        kernel_size = int(kernel_size)
        stride = kernel_size if stride is None else int(stride)
        padding = int(padding)
        if patches.dim() == 4:
            batch_size, patch_dim, n_h, n_w = patches.shape
            cols = patches.view(batch_size, patch_dim, n_h * n_w)
        elif patches.dim() == 3:
            batch_size, num_patches, patch_dim = patches.shape
            cols = patches.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported patch tensor shape: {tuple(patches.shape)}")

        channels = patch_dim // (kernel_size * kernel_size)
        if channels * kernel_size * kernel_size != patch_dim:
            raise ValueError(
                f"Patch dimension {patch_dim} is not divisible by kernel footprint {kernel_size * kernel_size}."
            )

        padded_output = (output_size[0] + 2 * padding, output_size[1] + 2 * padding)

        if use_window and (stride != kernel_size or padding > 0):
            window = torch.hann_window(kernel_size, periodic=False, device=cols.device, dtype=cols.dtype)
            window_2d = torch.outer(window, window).clamp_min(1e-3)
            window_2d = window_2d / window_2d.max().clamp_min(1e-8)
            weight_cols = window_2d.reshape(1, 1, -1).repeat(1, channels, 1).reshape(1, -1, 1)
            weighted_cols = cols * weight_cols
            output = F.fold(weighted_cols, output_size=padded_output, kernel_size=kernel_size, stride=stride)
            norm = F.fold(
                cols.new_ones((cols.shape[0], 1, cols.shape[-1])) * weight_cols,
                output_size=padded_output,
                kernel_size=kernel_size,
                stride=stride,
            )
            output = output / norm.clamp_min(1e-6)
        else:
            output = F.fold(cols, output_size=padded_output, kernel_size=kernel_size, stride=stride)

        if padding > 0:
            output = output[..., padding:-padding, padding:-padding]
        return output

    def direct_patch_mix_masked_queries(
        self,
        query_tokens_full: torch.Tensor,
        key_tokens_full: torch.Tensor,
        query_context_tokens_full: torch.Tensor | None,
        key_context_tokens_full: torch.Tensor | None,
        patch_values: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        default_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        batch_size, num_patches, _ = query_tokens_full.shape
        mixed = patch_values.clone() if default_tokens is None else default_tokens.clone()
        eye = torch.eye(num_patches, device=patch_values.device, dtype=patch_values.dtype)
        dense_attn = eye.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        masked_queries = query_mask_flat > 0.5
        valid_keys = key_valid_flat > 0.5
        rerank_aux = None
        shortlist_k = self.multihead_attention.attention_top_k
        if self.candidate_reranker and shortlist_k is not None:
            rerank_aux = {
                "candidate_indices": torch.full(
                    (batch_size, num_patches, shortlist_k),
                    -1,
                    dtype=torch.long,
                    device=patch_values.device,
                ),
                "candidate_base_logits": torch.zeros(
                    (batch_size, num_patches, shortlist_k),
                    dtype=query_tokens_full.dtype,
                    device=query_tokens_full.device,
                ),
                "candidate_logits": torch.zeros(
                    (batch_size, num_patches, shortlist_k),
                    dtype=query_tokens_full.dtype,
                    device=query_tokens_full.device,
                ),
                "candidate_valid_mask": torch.zeros(
                    (batch_size, num_patches, shortlist_k),
                    dtype=torch.bool,
                    device=query_tokens_full.device,
                ),
            }

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
                if self.candidate_reranker:
                    _, base_logits = self.multihead_attention.compute_attention_logits(query_tokens, key_tokens)
                    base_logits = base_logits.squeeze(0).squeeze(0)
                    candidate_count = min(base_logits.shape[-1], self.multihead_attention.attention_top_k)
                    base_candidate_logits, candidate_local_indices = torch.topk(
                        base_logits,
                        k=candidate_count,
                        dim=-1,
                    )
                    query_tokens_flat = query_tokens.squeeze(0)
                    key_tokens_flat = key_tokens.squeeze(0)
                    query_expanded = query_tokens_flat.unsqueeze(1).expand(-1, candidate_count, -1)
                    candidate_key_tokens = key_tokens_flat[candidate_local_indices]
                    if query_context_tokens_full is None or key_context_tokens_full is None:
                        raise ValueError("candidate_reranker requires query/key context tokens.")
                    query_context_tokens = query_context_tokens_full[batch_idx, query_indices]
                    key_context_tokens = key_context_tokens_full[batch_idx, key_indices]
                    query_context_expanded = query_context_tokens.unsqueeze(1).expand(-1, candidate_count, -1)
                    candidate_key_context_tokens = key_context_tokens[candidate_local_indices]
                    rerank_features = torch.cat(
                        [
                            query_expanded,
                            candidate_key_tokens,
                            query_expanded - candidate_key_tokens,
                            query_expanded * candidate_key_tokens,
                            query_context_expanded,
                            candidate_key_context_tokens,
                            query_context_expanded - candidate_key_context_tokens,
                            query_context_expanded * candidate_key_context_tokens,
                        ],
                        dim=-1,
                    )
                    rerank_delta = self.candidate_reranker_head(rerank_features).squeeze(-1)
                    rerank_logits = base_candidate_logits + rerank_delta
                    mixed_queries, masked_attention = self.multihead_attention(
                        query_tokens,
                        key_tokens,
                        value_tokens,
                        direct_patch_mixing=True,
                    )
                    mixed_queries = mixed_queries.squeeze(0).to(dtype=mixed.dtype)
                    masked_attention = masked_attention.squeeze(0).squeeze(0).to(dtype=replacement_rows.dtype)
                    candidate_absolute_indices = key_indices[candidate_local_indices]
                    replacement_rows[:, key_indices] = masked_attention
                    rerank_aux["candidate_indices"][batch_idx, query_indices, :candidate_count] = candidate_absolute_indices
                    rerank_aux["candidate_base_logits"][batch_idx, query_indices, :candidate_count] = base_candidate_logits
                    rerank_aux["candidate_logits"][batch_idx, query_indices, :candidate_count] = rerank_logits
                    rerank_aux["candidate_valid_mask"][batch_idx, query_indices, :candidate_count] = True
                else:
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

        return mixed, dense_attn, rerank_aux

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

    def flatten_query_mask(self, mask: torch.Tensor) -> torch.Tensor:
        query_mask_patch_map, _ = self.unfold_native(mask, self.kernel_size)
        return (query_mask_patch_map.amax(dim=1) > 0).to(dtype=mask.dtype).flatten(start_dim=1)

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

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        value_image: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        coarse_raw, features = self.encoder_decoder(image)
        known_image = image if value_image is None else value_image
        coarse_composite = coarse_raw * mask + known_image * (1 - mask)

        patch_map, output_size = self.unfold_native(coarse_raw, self.kernel_size)
        default_patch_values = None
        if self.value_source == "high_freq_residual":
            value_base = known_image - self.final_gaussian_blur(known_image)
        else:
            value_base = known_image

        source_patch_map, _ = self.extract_patches(
            value_base,
            self.value_patch_size,
            stride=self.kernel_size,
            padding=self.value_patch_padding,
        )
        query_mask_flat = self.flatten_query_mask(mask)
        key_mask_patch_map, _ = self.extract_patches(
            mask,
            self.value_patch_size,
            stride=self.kernel_size,
            padding=self.value_patch_padding,
        )
        key_valid_flat = (key_mask_patch_map.amax(dim=1) == 0).to(dtype=patch_map.dtype).flatten(start_dim=1)

        query_matching_tokens = None
        key_matching_tokens = None
        candidate_query_context_tokens = None
        candidate_key_context_tokens = None
        if self.candidate_reranker:
            candidate_query_context_map = self._pool_to_token_grid(
                self.candidate_query_context_encoder(torch.cat([image, coarse_composite, mask], dim=1)),
                patch_map.shape[-2:],
            )
            candidate_key_context_map = self._pool_to_token_grid(
                self.candidate_key_context_encoder(known_image),
                patch_map.shape[-2:],
            )
            candidate_query_context_tokens = candidate_query_context_map.flatten(start_dim=2).transpose(1, 2)
            candidate_key_context_tokens = candidate_key_context_map.flatten(start_dim=2).transpose(1, 2)
        if self.separate_query_key_matching:
            visible_patch_map, _ = self.unfold_native(image, self.kernel_size)
            query_context_map = self._pool_to_token_grid(
                self.query_context_encoder(torch.cat([image, mask], dim=1)),
                patch_map.shape[-2:],
            )
            key_context_map = self._pool_to_token_grid(
                self.key_context_encoder(image),
                patch_map.shape[-2:],
            )

            query_token_inputs = [query_context_map, visible_patch_map]
            if self.match_coarse_rgb:
                coarse_match_map = patch_map.detach() if self.detach_coarse_rgb else patch_map
                coarse_match_map = self._prepare_matching_branch(
                    coarse_match_map,
                    norm=self.coarse_rgb_branch_norm,
                    scale=self.coarse_rgb_branch_scale,
                    drop_prob=self.coarse_rgb_branch_dropout,
                )
                query_token_inputs.append(coarse_match_map)
            if self.concat_features:
                coarse_features = F.interpolate(
                    features[self.feature_i],
                    size=patch_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                coarse_features = self._prepare_matching_branch(
                    coarse_features,
                    norm=self.feature_branch_norm,
                    scale=self.feature_branch_scale,
                    drop_prob=0.0,
                )
                query_token_inputs.append(coarse_features)

            key_token_inputs = [key_context_map, visible_patch_map]
            if self.match_coarse_rgb:
                key_token_inputs.append(self.key_coarse_rgb_scale * coarse_match_map)
            if self.concat_features:
                key_token_inputs.append(self.key_feature_scale * coarse_features)
            query_token_map = torch.cat(query_token_inputs, dim=1)
            key_token_map = torch.cat(key_token_inputs, dim=1)
            if self.shared_query_key_descriptor:
                query_token_map = self.shared_query_key_descriptor_head(query_token_map)
                key_token_map = self.shared_query_key_descriptor_head(key_token_map)
            else:
                query_token_map = self.query_descriptor_head(query_token_map)
                key_token_map = self.key_descriptor_head(key_token_map)
            query_matching_tokens = query_token_map.flatten(start_dim=2).transpose(1, 2)
            key_matching_tokens = key_token_map.flatten(start_dim=2).transpose(1, 2)
            query_tokens = query_matching_tokens
            key_tokens = key_matching_tokens
        else:
            token_inputs = []
            if self.match_coarse_rgb:
                coarse_match_map = patch_map.detach() if self.detach_coarse_rgb else patch_map
                coarse_match_map = self._prepare_matching_branch(
                    coarse_match_map,
                    norm=self.coarse_rgb_branch_norm,
                    scale=self.coarse_rgb_branch_scale,
                    drop_prob=self.coarse_rgb_branch_dropout,
                )
                token_inputs.append(coarse_match_map)
            if self.concat_features:
                coarse_features = F.interpolate(
                    features[self.feature_i],
                    size=patch_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                coarse_features = self._prepare_matching_branch(
                    coarse_features,
                    norm=self.feature_branch_norm,
                    scale=self.feature_branch_scale,
                    drop_prob=0.0,
                )
                token_inputs.append(coarse_features)

            token_map = token_inputs[0] if len(token_inputs) == 1 else torch.cat(token_inputs, dim=1)
            if self.matching_descriptor_head is not None:
                token_map = self.matching_descriptor_head(token_map)

            query_matching_tokens = token_map.flatten(start_dim=2).transpose(1, 2)
            key_matching_tokens = query_matching_tokens
            if self.query_image_context_matching:
                visible_patch_map, _ = self.unfold_native(image, self.kernel_size)
                query_context_map = self._pool_to_token_grid(
                    self.query_context_encoder(torch.cat([image, mask], dim=1)),
                    patch_map.shape[-2:],
                )
                query_context_map = torch.cat([query_context_map, visible_patch_map], dim=1)
                query_context_map = self.query_context_descriptor_head(query_context_map)
                query_context_tokens = query_context_map.flatten(start_dim=2).transpose(1, 2)
                query_matching_tokens = query_matching_tokens + self.query_context_scale * query_context_tokens

            query_tokens = query_matching_tokens
            key_tokens = key_matching_tokens

        positional_encoding = self.get_positional_encoding()
        if positional_encoding is not None:
            query_tokens = query_tokens + positional_encoding
            key_tokens = key_tokens + positional_encoding
        query_tokens = self.pre_attention_norm(query_tokens)
        key_tokens = self.pre_attention_norm(key_tokens)

        # The paper mixes source LR patches with attention learned from coarse tokens.
        patch_values = source_patch_map.flatten(start_dim=2).transpose(1, 2)
        if self.value_source == "high_freq_residual":
            default_patch_values = torch.zeros_like(patch_values)
        rerank_aux = None
        if self.attention_masking:
            mixed_patches_flat, masked_attention, rerank_aux = self.direct_patch_mix_masked_queries(
                query_tokens,
                key_tokens,
                candidate_query_context_tokens,
                candidate_key_context_tokens,
                patch_values,
                query_mask_flat,
                key_valid_flat,
                default_tokens=default_patch_values,
            )
        else:
            mixed_patches_flat, masked_attention = self.multihead_attention(
                query_tokens,
                key_tokens,
                patch_values,
                direct_patch_mixing=True,
                query_mask_flat=query_mask_flat,
            )

        mixed_image = self.fold_native(
            mixed_patches_flat,
            output_size,
            kernel_size=self.value_patch_size,
            stride=self.kernel_size,
            padding=self.value_patch_padding,
            use_window=self.value_patch_padding > 0,
        )
        if self.fusion_mode == "add" or self.value_source == "high_freq_residual":
            refined = coarse_composite + mixed_image
        elif self.fusion_mode == "gate":
            gate_input = torch.cat(
                [coarse_composite, mixed_image, (coarse_composite - mixed_image).abs(), mask],
                dim=1,
            )
            gate = torch.sigmoid(self.fusion_gate(gate_input))
            refined = gate * mixed_image + (1.0 - gate) * coarse_composite
        else:
            refined = mixed_image

        if self.coherence_layer is not None:
            refined = refined + self.coherence_layer(refined)

        refined = refined * mask + known_image * (1 - mask)

        if return_aux:
            aux = {
                "query_mask_flat": query_mask_flat,
                "key_valid_flat": key_valid_flat,
                "kernel_size": self.kernel_size,
                "value_patch_size": self.value_patch_size,
                "value_patch_padding": self.value_patch_padding,
                "matching_tokens": query_matching_tokens,
                "query_matching_tokens": query_matching_tokens,
                "key_matching_tokens": key_matching_tokens,
            }
            if rerank_aux is not None:
                aux.update(rerank_aux)
            return refined, masked_attention, coarse_raw, aux
        return refined, masked_attention, coarse_raw
