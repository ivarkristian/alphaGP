"""AlphaZero-style MCTS tailored to EnvAdapter."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import EnvAdapterClass

class Node:
    def __init__(self, prior: torch.Tensor, device: torch.device) -> None:
        self.prior = prior if torch.is_tensor(prior) else torch.as_tensor(prior, device=device)
        self.visit_count = torch.zeros((), device=device, dtype=torch.float32)
        self.value_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.children: Dict[int, "Node"] = {}
        self.device = device

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> torch.Tensor:
        zero = torch.zeros((), device=self.device, dtype=self.value_sum.dtype)
        return torch.where(self.visit_count > 0, self.value_sum / self.visit_count, zero)

    def expand(self, policy_probs: torch.Tensor, legal_actions: Iterable[int]) -> None:
        if torch.is_tensor(legal_actions):
            legal_actions = legal_actions.tolist()
        for action in legal_actions:
            if action in self.children:
                continue
            self.children[action] = Node(prior=policy_probs[action], device=self.device)


class MCTS:
    def __init__(
        self,
        net,
        num_simulations: int = 50,
        cpuct: float = 1.5,
        discount: float = 1.0,
        temperature: float = 1.0,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_epsilon: float = 0.0,
        tree_depth_max: Optional[int] = None,
        rollout_mode: str = "belief_surrogate",
        rollout_reward_weights: Sequence[float] = (0.34, 0.33, 0.33),
        rollout_var_reduction_scale: float = 0.5,
        rollout_lengthscale: float | None = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.net = net
        self.num_simulations = int(num_simulations)
        self.cpuct = float(cpuct)
        self.discount = float(discount)
        self.temperature = float(temperature)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.tree_depth_max = tree_depth_max
        self.rollout_mode = rollout_mode
        self.rollout_reward_weights = tuple(rollout_reward_weights)
        self.rollout_var_reduction_scale = float(rollout_var_reduction_scale)
        self.rollout_lengthscale = rollout_lengthscale
        self.device = device or next(net.parameters()).device

    def run_search(self, root_env, return_root: bool = False):
        """
        Run MCTS simulations from the current EnvAdapter state.

        Returns (policy_probs, root_value) or (policy_probs, root_value, root_node)
        when return_root is True.
        """
        if not getattr(root_env.env, "return_torch", False):
            raise ValueError("MCTS requires env.return_torch=True to avoid CPU/GPU copies.")
        root_obs = self._current_obs(root_env)
        root_policy, root_value = self._evaluate_batch([root_obs])

        root_node = Node(prior=torch.tensor(1.0, device=self.device), device=self.device)
        root_legal = root_env.legal_actions()
        policy_probs = self._mask_and_normalize(root_policy[0], root_legal)

        if self.dirichlet_alpha and self.dirichlet_epsilon > 0.0:
            policy_probs = self._add_dirichlet_noise(policy_probs, root_legal)

        root_node.expand(policy_probs, root_legal)

        for _ in range(self.num_simulations):
            node = root_node
            env = root_env.clone()
            search_path: List[Tuple[Node, torch.Tensor]] = []

            done = False
            obs = None
            depth = 0
            depth_limit_hit = False
            while node.expanded():
                if self.tree_depth_max is not None and depth >= self.tree_depth_max:
                    depth_limit_hit = True
                    break
                action, action_t = self._select_action(node)
                obs, reward, terminated, truncated, _info = self._env_rollout_step(env, action_t)
                reward_t = reward if torch.is_tensor(reward) else torch.as_tensor(reward, device=self.device, dtype=torch.float32)
                search_path.append((node, reward_t))
                node = node.children[action]
                done = bool(terminated or truncated)
                if done:
                    break
                depth += 1

            if done:
                leaf_value = torch.zeros((), device=self.device, dtype=torch.float32)
            else:
                if obs is None:
                    obs = self._current_obs(env)
                policy_logits, value = self._evaluate_batch([obs])
                leaf_legal = env.legal_actions()
                policy_probs = self._mask_and_normalize(policy_logits[0], leaf_legal)
                if not depth_limit_hit:
                    node.expand(policy_probs, leaf_legal)
                leaf_value = value[0]

            self._backpropagate(search_path, leaf_value)

        policy = self._visits_to_policy(root_node, self.temperature, num_actions=int(root_policy.shape[-1]))
        if return_root:
            return policy, root_node.value(), root_node
        return policy, root_node.value()

    def _current_obs(self, env_adapter):
        obs, _truncated, _info = env_adapter.env._get_obs_truncated_info()
        return obs

    def _select_action(self, node: Node) -> Tuple[int, torch.Tensor]:
        actions = list(node.children.keys())
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        priors = torch.stack([node.children[a].prior for a in actions])
        visits = torch.stack([node.children[a].visit_count for a in actions])
        values = torch.stack([node.children[a].value() for a in actions])

        total_visits = visits.sum()
        sqrt_total = torch.sqrt(total_visits + 1.0)
        u = self.cpuct * priors * sqrt_total / (1.0 + visits)
        scores = values + u
        best_idx_t = torch.argmax(scores)
        action_t = actions_t[best_idx_t]
        return int(action_t.item()), action_t

    def _backpropagate(self, path: List[Tuple[Node, torch.Tensor]], leaf_value: torch.Tensor) -> None:
        value = leaf_value
        for node, reward in reversed(path):
            node.visit_count += 1.0
            node.value_sum += value
            value = reward + self.discount * value

    def _visits_to_policy(self, root: Node, temperature: float, num_actions: int) -> torch.Tensor:
        visit_counts = {a: child.visit_count for a, child in root.children.items()}
        actions = sorted(visit_counts.keys())
        if not actions:
            return torch.zeros(num_actions, device=self.device, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        counts = torch.stack([visit_counts[a] for a in actions])

        if temperature <= 0:
            best = torch.argmax(counts).item()
            probs = torch.zeros_like(counts)
            probs[best] = 1.0
        else:
            counts = counts.pow(1.0 / temperature)
            probs = counts / counts.sum()

        policy = torch.zeros(num_actions, device=self.device, dtype=torch.float32)
        policy.index_copy_(0, actions_t, probs)
        return policy

    def _prepare_obs(self, obs):
        if isinstance(obs, dict):
            map_t = obs.get("map")
            if not torch.is_tensor(map_t):
                map_t = torch.as_tensor(map_t, device=self.device)
            map_t = map_t.to(device=self.device, dtype=torch.float32)
            max_val = map_t.max()
            scale = torch.where(
                max_val > 1.5,
                torch.tensor(1.0 / 255.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            )
            map_t = map_t * scale

            loc = obs.get("loc")
            hdg = obs.get("hdg")
            if loc is not None and not torch.is_tensor(loc):
                loc = torch.as_tensor(loc, device=self.device, dtype=torch.float32)
            if hdg is not None and not torch.is_tensor(hdg):
                hdg = torch.as_tensor(hdg, device=self.device, dtype=torch.float32)
            if torch.is_tensor(loc):
                loc = loc.to(device=self.device, dtype=torch.float32)
            if torch.is_tensor(hdg):
                hdg = hdg.to(device=self.device, dtype=torch.float32)

            return map_t, loc, hdg

        map_t = obs if torch.is_tensor(obs) else torch.as_tensor(obs, device=self.device)
        map_t = map_t.to(device=self.device, dtype=torch.float32)
        max_val = map_t.max()
        scale = torch.where(
            max_val > 1.5,
            torch.tensor(1.0 / 255.0, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        map_t = map_t * scale
        return map_t, None, None

    def _evaluate_batch(self, obs_list: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate a batch of observations with the policy/value net.

        Structured for easy batching later; currently batches the list into a
        single forward pass.
        """
        maps = []
        aux_list = []
        has_aux = False

        for obs in obs_list:
            map_t, loc, hdg = self._prepare_obs(obs)
            if map_t.dim() == 4:
                map_t = map_t[0]
            maps.append(map_t)
            if loc is not None and hdg is not None:
                aux = torch.cat([loc.view(-1), hdg.view(-1)], dim=0)
                aux_list.append(aux)
                has_aux = True

        map_batch = torch.stack(maps, dim=0).to(self.device)
        aux_batch = None
        if has_aux:
            aux_batch = torch.stack(aux_list, dim=0).to(self.device)

        with torch.no_grad():
            policy_logits, values = self.net(map_batch, aux_batch)

        return policy_logits, values.view(-1)

    def _mask_and_normalize(self, logits: torch.Tensor, legal_actions: Iterable[int]) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        legal_actions_t = torch.as_tensor(list(legal_actions), device=self.device, dtype=torch.long)
        if legal_actions_t.numel() == 0:
            return probs
        mask = torch.zeros_like(probs)
        mask.index_fill_(0, legal_actions_t, 1.0)
        masked = probs * mask
        masked_sum = masked.sum()
        fallback = mask / mask.sum()
        return torch.where(masked_sum > 0, masked / masked_sum, fallback)

    def _add_dirichlet_noise(self, probs: torch.Tensor, legal_actions: Iterable[int]) -> torch.Tensor:
        legal_actions_t = torch.as_tensor(list(legal_actions), device=self.device, dtype=torch.long)
        if legal_actions_t.numel() == 0:
            return probs

        noise = torch.distributions.Dirichlet(
            torch.full((int(legal_actions_t.numel()),), float(self.dirichlet_alpha), device=self.device)
        ).sample()
        noise_full = torch.zeros_like(probs)
        noise_full.index_copy_(0, legal_actions_t, noise)
        return (1 - self.dirichlet_epsilon) * probs + self.dirichlet_epsilon * noise_full

    def _env_step(self, env_adapter, action: torch.Tensor):
        """
        Step using EnvAdapter RNG handling, returning raw obs dict on-device.
        """
        if getattr(env_adapter, "_restore_rng_on_step", False) and env_adapter._rng_state is not None:
            env_adapter._set_rng_state(env_adapter._rng_state)

        obs, reward, terminated, truncated, info = env_adapter.env.step(action)

        if getattr(env_adapter, "_restore_rng_on_step", False):
            env_adapter._rng_state = env_adapter._capture_rng_state()

        return obs, reward, terminated, truncated, info

    def _env_rollout_step(self, env_adapter, action: torch.Tensor):
        if self.rollout_mode == "env":
            return self._env_step(env_adapter, action)
        if self.rollout_mode == "belief_surrogate":
            return env_adapter.rollout_step(
                action,
                reward_weights=self.rollout_reward_weights,
                var_reduction_scale=self.rollout_var_reduction_scale,
                lengthscale=self.rollout_lengthscale,
            )
        raise ValueError(f"Unknown rollout_mode: {self.rollout_mode}")
