"""Policy/value network for AlphaZero-style MCTS."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn


class PolicyValueNet(nn.Module):
    """
    CNN trunk over map channels + optional aux vector (loc + heading).

    Forward returns policy logits and scalar value in [-1, 1].
    """

    def __init__(
        self,
        in_channels: int,
        num_actions: int = 3,
        aux_dim: int = 10,
        trunk_channels: Sequence[int] = (32, 64, 128),
        latent_dim: int = 256,
        head_dim: int = 128,
        value_head_dim: int = 128,
    ) -> None:
        super().__init__()
        self.aux_dim = int(aux_dim)

        layers: list[nn.Module] = []
        prev = int(in_channels)
        for ch in trunk_channels:
            layers.append(nn.Conv2d(prev, int(ch), kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            prev = int(ch)
        self.trunk = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.trunk_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev, latent_dim),
            nn.ReLU(inplace=True),
        )

        fused_in = latent_dim + self.aux_dim if self.aux_dim > 0 else latent_dim
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, head_dim),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Linear(head_dim, num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(head_dim, value_head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(value_head_dim, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def forward(
        self,
        obs: Any,
        aux: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        map_t, aux_t = self._unpack_obs(obs, aux)
        feats = self.trunk(map_t)
        feats = self.pool(feats)
        feats = self.trunk_fc(feats)

        if self.aux_dim > 0:
            if aux_t is None:
                raise ValueError("aux input is required when aux_dim > 0.")
            if aux_t.shape[1] != self.aux_dim:
                raise ValueError(f"aux_dim mismatch: expected {self.aux_dim}, got {aux_t.shape[1]}")
            feats = torch.cat([feats, aux_t], dim=1)

        fused = self.fuse(feats)
        policy_logits = self.policy_head(fused)
        value = self.value_head(fused)
        return policy_logits, value

    def _unpack_obs(
        self,
        obs: Any,
        aux: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(obs, dict):
            map_t = obs.get("map")
            aux_t = self._build_aux_from_obs(obs, map_t)
        elif isinstance(obs, (tuple, list)) and len(obs) == 2:
            map_t, aux_t = obs
        else:
            map_t, aux_t = obs, aux

        if not isinstance(map_t, torch.Tensor):
            map_t = torch.as_tensor(map_t)

        if map_t.dim() == 3:
            map_t = map_t.unsqueeze(0)
        map_t = map_t.to(dtype=torch.float32)

        if aux_t is not None and not isinstance(aux_t, torch.Tensor):
            aux_t = torch.as_tensor(aux_t)

        if aux_t is not None:
            if aux_t.dim() == 1:
                aux_t = aux_t.unsqueeze(0)
            if aux_t.shape[0] == 1 and map_t.shape[0] > 1:
                aux_t = aux_t.expand(map_t.shape[0], -1)
            aux_t = aux_t.to(device=map_t.device, dtype=torch.float32)

        map_t = map_t.to(device=map_t.device)
        return map_t, aux_t

    @staticmethod
    def _build_aux_from_obs(
        obs: Dict[str, Any],
        map_t: Any,
    ) -> Optional[torch.Tensor]:
        if "loc" not in obs or "hdg" not in obs:
            return None

        loc = obs["loc"]
        hdg = obs["hdg"]
        if not isinstance(loc, torch.Tensor):
            loc = torch.as_tensor(loc)
        if not isinstance(hdg, torch.Tensor):
            hdg = torch.as_tensor(hdg)

        if loc.dim() == 1:
            loc = loc.unsqueeze(0)
        if hdg.dim() == 1:
            hdg = hdg.unsqueeze(0)

        if isinstance(map_t, torch.Tensor) and map_t.dim() == 4:
            batch = map_t.shape[0]
            if loc.shape[0] == 1 and batch > 1:
                loc = loc.expand(batch, -1)
            if hdg.shape[0] == 1 and batch > 1:
                hdg = hdg.expand(batch, -1)

        return torch.cat([loc, hdg], dim=1)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def compute_policy_value_loss(
    model: nn.Module,
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    l2_weight: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """
    Compute policy loss, value loss, and L2 regularization for AlphaZero-style training.
    """
    if target_policy.dim() == 1:
        target_policy = target_policy.unsqueeze(0)
    if policy_logits.dim() == 1:
        policy_logits = policy_logits.unsqueeze(0)

    log_probs = torch.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

    value_pred = value_pred.view(-1)
    target_value = target_value.view(-1).to(value_pred.device, dtype=value_pred.dtype)
    value_loss = torch.mean((value_pred - target_value) ** 2)

    l2_loss = torch.zeros((), device=value_pred.device, dtype=value_pred.dtype)
    if l2_weight > 0:
        l2_loss = sum((param ** 2).sum() for param in model.parameters()) * l2_weight

    total_loss = policy_loss + value_loss + l2_loss
    return {
        "total": total_loss,
        "policy": policy_loss,
        "value": value_loss,
        "l2": l2_loss,
    }
