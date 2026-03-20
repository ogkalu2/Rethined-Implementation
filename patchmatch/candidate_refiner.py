from __future__ import annotations

import torch
from torch import nn


class CandidateRefinementBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, dropout: float, ff_mult: float = 2.0):
        super().__init__()
        ff_dim = max(token_dim, int(round(token_dim * float(ff_mult))))
        self.self_norm = nn.LayerNorm(token_dim)
        self.ff_norm = nn.LayerNorm(token_dim)
        self.self_attn = nn.MultiheadAttention(
            token_dim,
            num_heads=max(1, int(num_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(token_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, token_dim),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        norm_tokens = self.self_norm(tokens)
        self_out, _ = self.self_attn(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
        tokens = residual + self.dropout(self_out)

        residual = tokens
        ff_out = self.ff(self.ff_norm(tokens))
        tokens = residual + self.dropout(ff_out)
        return tokens


class CandidateJointRefiner(nn.Module):
    def __init__(
        self,
        token_dim: int,
        *,
        steps: int,
        num_heads: int,
        dropout: float,
        ff_mult: float = 2.0,
    ):
        super().__init__()
        self.steps = max(0, int(steps))
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.query_initializer = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.blocks = nn.ModuleList(
            [
                CandidateRefinementBlock(
                    token_dim,
                    num_heads=max(1, int(num_heads)),
                    dropout=float(dropout),
                    ff_mult=float(ff_mult),
                )
                for _ in range(self.steps)
            ]
        )
        self.delta_head = nn.Sequential(
            nn.Linear(token_dim * 2 + 1, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )

    def _component_local_groups(
        self,
        query_indices: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> list[torch.Tensor]:
        if query_indices.numel() == 0:
            return []
        height, width = int(token_hw[0]), int(token_hw[1])
        order = torch.full((height * width,), -1, dtype=torch.long, device=query_indices.device)
        local_order = torch.arange(query_indices.numel(), device=query_indices.device, dtype=torch.long)
        order[query_indices] = local_order
        visited = torch.zeros((query_indices.numel(),), dtype=torch.bool, device=query_indices.device)
        components: list[torch.Tensor] = []

        for start_local in range(query_indices.numel()):
            if bool(visited[start_local]):
                continue
            visited[start_local] = True
            stack = [int(start_local)]
            current: list[int] = []

            while stack:
                local_idx = stack.pop()
                current.append(local_idx)
                token_idx = int(query_indices[local_idx].item())
                y = token_idx // width
                x = token_idx % width
                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                        continue
                    neighbor_token = (next_y * width) + next_x
                    neighbor_local = int(order[neighbor_token].item())
                    if neighbor_local < 0 or bool(visited[neighbor_local]):
                        continue
                    visited[neighbor_local] = True
                    stack.append(neighbor_local)

            components.append(torch.tensor(current, dtype=torch.long, device=query_indices.device))

        return components

    def forward(
        self,
        query_tokens: torch.Tensor,
        query_indices: torch.Tensor,
        candidate_key_tokens: torch.Tensor,
        candidate_coords: torch.Tensor,
        candidate_logits: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> torch.Tensor:
        if query_tokens.numel() == 0 or candidate_logits.numel() == 0:
            return candidate_logits

        coord_inputs = torch.cat([candidate_coords, candidate_logits.unsqueeze(-1)], dim=-1)
        coord_features = self.coord_encoder(coord_inputs)
        candidate_features = self.candidate_encoder(torch.cat([candidate_key_tokens, coord_features], dim=-1))
        candidate_probs = torch.softmax(candidate_logits, dim=-1)
        candidate_summary = (candidate_probs.unsqueeze(-1) * candidate_features).sum(dim=1)
        query_state = self.query_initializer(torch.cat([query_tokens, candidate_summary], dim=-1))

        if self.steps > 0:
            for component in self._component_local_groups(query_indices, token_hw):
                if component.numel() <= 1:
                    continue
                component_state = query_state[component].unsqueeze(0)
                for block in self.blocks:
                    component_state = block(component_state)
                query_state[component] = component_state.squeeze(0)

        expanded_query = query_state.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        delta_inputs = torch.cat([expanded_query, candidate_features, candidate_logits.unsqueeze(-1)], dim=-1)
        delta_logits = self.delta_head(delta_inputs).squeeze(-1)
        return candidate_logits + delta_logits
