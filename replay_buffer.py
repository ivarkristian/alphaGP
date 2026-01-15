"""Replay buffer for self-play trajectories."""
from __future__ import annotations

from typing import Any, Iterable, Mapping
import random

import torch
from pathlib import Path


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000, device: str | torch.device | None = None) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device) if device is not None else None
        self._data: list[dict[str, Any]] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._data)

    def add_episode(self, episode: Iterable[Any]) -> None:
        for item in episode:
            if isinstance(item, Mapping):
                obs = item.get("obs")
                policy = item.get("policy")
                reward = item.get("reward")
                done = item.get("done")
                value = item.get("value")
            else:
                if len(item) != 5:
                    raise ValueError("Episode items must be (obs, policy, reward, done, value).")
                obs, policy, reward, done, value = item
            self.add_transition(obs, policy, reward, done, value)

    def add_transition(self, obs, policy, reward, done, value) -> None:
        entry = {
            "obs": obs,
            "policy": policy,
            "reward": reward,
            "done": done,
            "value": value,
        }
        if len(self._data) < self.capacity:
            self._data.append(entry)
        else:
            self._data[self._pos] = entry
            self._pos = (self._pos + 1) % self.capacity

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "pos": self._pos,
            "data": self._to_cpu(self._data),
        }

    def load_state_dict(self, state: Mapping[str, Any], device: str | torch.device | None = None) -> None:
        self.capacity = int(state["capacity"])
        self._pos = int(state["pos"])
        self._data = self._to_device(state["data"], device=device or self.device)

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "ReplayBuffer":
        state = torch.load(path, map_location=device or "cpu")
        buffer = cls(capacity=int(state["capacity"]), device=device)
        buffer.load_state_dict(state, device=device)
        return buffer

    def sample_batch(self, batch_size: int) -> dict[str, Any]:
        if batch_size > len(self._data):
            raise ValueError("Batch size exceeds number of stored transitions.")
        indices = random.sample(range(len(self._data)), batch_size)
        batch = [self._data[i] for i in indices]

        obs_batch = self._collate_obs([item["obs"] for item in batch])
        policy_batch = self._collate_tensor([item["policy"] for item in batch], dtype=torch.float32)
        reward_batch = self._collate_tensor([item["reward"] for item in batch], dtype=torch.float32).view(-1)
        done_batch = self._collate_tensor([item["done"] for item in batch], dtype=torch.bool).view(-1)
        value_batch = self._collate_tensor([item["value"] for item in batch], dtype=torch.float32).view(-1)

        return {
            "obs": obs_batch,
            "policy": policy_batch,
            "reward": reward_batch,
            "done": done_batch,
            "value": value_batch,
        }

    def _collate_obs(self, obs_list: list[Any]):
        if not obs_list:
            raise ValueError("No observations provided.")
        first = obs_list[0]
        if isinstance(first, Mapping):
            keys = first.keys()
            return {key: self._collate_tensor([obs[key] for obs in obs_list]) for key in keys}
        return self._collate_tensor(obs_list)

    def _collate_tensor(self, items: list[Any], dtype: torch.dtype | None = None) -> torch.Tensor:
        tensors = [self._to_tensor(item, dtype=dtype) for item in items]
        return torch.stack(tensors, dim=0)

    def _to_tensor(self, item: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
        if torch.is_tensor(item):
            tensor = item
        else:
            tensor = torch.as_tensor(item)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor

    def _to_cpu(self, obj: Any):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_cpu(v) for v in obj)
        return obj

    def _to_device(self, obj: Any, device: str | torch.device | None):
        if torch.is_tensor(obj):
            return obj.to(device) if device is not None else obj
        if isinstance(obj, dict):
            return {k: self._to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_device(v, device) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_device(v, device) for v in obj)
        return obj
