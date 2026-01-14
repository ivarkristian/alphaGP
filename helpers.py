"""Shared helpers for configuration and utilities."""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = Path("config.json")


@dataclass(frozen=True)
class Config:
    seed: int = 123
    device: str = "cpu"
    run_name: str = "debug"
    env_turn_radius: float = 250.0
    train_num_iterations: int = 5
    tensor_envs_dir: str = "tensor_envs"
    tensor_env_file: str | None = None
    figures_dir: str = "figures"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str | None = None) -> "Config":
        if not isinstance(data, Mapping):
            raise TypeError("Config data must be a mapping.")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(key for key in data.keys() if key not in allowed)
        if unknown:
            where = f" in {source}" if source else ""
            raise ValueError(f"Unknown config keys{where}: {', '.join(unknown)}")
        filtered = {key: data[key] for key in allowed if key in data}
        return cls(**filtered)


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load a config JSON file and return a Config object."""
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = Config.from_dict(data, source=str(path))
    _ensure_config_dirs(cfg)
    return cfg


def _ensure_config_dirs(cfg: Config) -> None:
    """Ensure config-specified directories exist."""
    for dir_path in (cfg.tensor_envs_dir, cfg.figures_dir):
        Path(dir_path).mkdir(parents=True, exist_ok=True)


class EnvAdapter:
    """Thin wrapper around GasSurveyDubinsEnv for MCTS-friendly IO."""

    def __init__(self, cfg: Config, scenario_bank=None, channels=None, return_torch: bool = True) -> None:
        import numpy as np
        import torch

        from rl_gas_survey_dubins_env import GasSurveyDubinsEnv
        from rl_scenario_bank import ScenarioBank

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.return_torch = return_torch
        self._rng_state = None
        self._restore_rng_on_step = False

        if scenario_bank is None:
            if cfg.tensor_env_file is None:
                raise ValueError("Config tensor_env_file is required to load scenarios.")
            self.scenario_bank = ScenarioBank(data_dir=str(cfg.tensor_envs_dir))
            self.scenario_bank.load_envs(cfg.tensor_env_file, device=self.device)
        else:
            self.scenario_bank = scenario_bank

        channels_arr = np.array([1, 1, 0, 0, 0], dtype=np.uint8) if channels is None else channels

        self.env = GasSurveyDubinsEnv(
            scenario_bank=self.scenario_bank,
            turn_radius=cfg.env_turn_radius,
            channels=channels_arr,
            device=self.device,
            return_torch=self.return_torch,
        )

    def preprocess_obs(self, obs):
        """Convert env obs to a float32 torch tensor in CHW layout."""
        import torch

        obs_map = obs["map"]
        if isinstance(obs_map, torch.Tensor):
            obs_t = obs_map.to(device=self.device, dtype=torch.float32)
        else:
            obs_t = torch.as_tensor(obs_map, device=self.device, dtype=torch.float32)
        return obs_t / 255.0

    def reset(self, **kwargs):
        """Reset env and return (obs_t, info)."""
        obs, info = self.env.reset(**kwargs)
        obs_t = self.preprocess_obs(obs)
        return obs_t, info

    def step(self, action):
        """Step env and return (obs_t, reward, done, info)."""
        import torch

        if self._restore_rng_on_step and self._rng_state is not None:
            self._set_rng_state(self._rng_state)

        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_t = self.preprocess_obs(obs)

        if self._restore_rng_on_step:
            self._rng_state = self._capture_rng_state()

        done = terminated or truncated
        info = dict(info or {})
        info["truncated"] = truncated
        if self.return_torch:
            reward_t = reward if isinstance(reward, torch.Tensor) else torch.as_tensor(reward, device=self.device, dtype=torch.float32)
            return obs_t, reward_t, done, info
        return obs_t, float(reward), done, info

    def legal_actions(self):
        """
        Return a list of legal discrete actions from the current state.

        Uses deterministic geometry (no motion noise) for boundary checks.
        """
        from rl_gas_survey_dubins_env import move_with_heading

        n_actions = int(self.env.action_space.n)
        if not hasattr(self.env, "heading") or not hasattr(self.env, "loc_xy_np"):
            return list(range(n_actions))

        n_headings = len(self.env.heading)
        legal = []
        for action in range(n_actions):
            delta_xy, new_heading = move_with_heading(
                heading_1hot=self.env.heading,
                action=action,
                turn_radius=self.env.turn_radius,
                turn_degrees=int(360 / n_headings),
                n_headings=n_headings,
                straight_matches_arc=True,
            )
            new_xy = self.env.loc_xy_np + delta_xy
            out_of_bounds = not (
                0 <= new_xy[0] <= self.env.env_x_max and 0 <= new_xy[1] <= self.env.env_y_max
            )
            if out_of_bounds or self.env._facing_the_boundary(new_xy, new_heading):
                continue
            legal.append(action)

        # Fallback: if geometry rejects everything, allow all actions.
        return legal if legal else list(range(n_actions))

    def clone(self):
        """
        Deep-copy the environment state for MCTS rollouts.

        This is intentionally simple but expensive; later we can replace it with a
        lighter-weight state snapshot/restore and per-env RNG to avoid global RNG mutation.
        """
        import copy

        rng_state = self._capture_rng_state()
        clone_adapter = self.__class__.__new__(self.__class__)
        clone_adapter.cfg = self.cfg
        clone_adapter.device = self.device
        clone_adapter.return_torch = self.return_torch
        clone_adapter.scenario_bank = self.scenario_bank
        clone_adapter._rng_state = rng_state
        clone_adapter._restore_rng_on_step = True

        # Avoid duplicating the scenario bank (large tensors); share it between clones.
        env_copy = copy.deepcopy(self.env, memo={id(self.scenario_bank): self.scenario_bank})
        env_copy.return_torch = self.return_torch
        clone_adapter.env = env_copy
        return clone_adapter

    @staticmethod
    def _capture_rng_state():
        import random
        import numpy as np
        import torch

        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _set_rng_state(state) -> None:
        import random
        import numpy as np
        import torch

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def plot_env(
        self,
        x=None,
        y=None,
        c=None,
        path=None,
        x_range=None,
        y_range=None,
        value_title: str = "",
    ):
        """Plot the current environment state via GasSurveyDubinsEnv.plot_env()."""
        import numpy as np
        import torch

        def _to_cpu_np(arr):
            if torch.is_tensor(arr):
                return arr.detach().cpu().numpy()
            return np.asarray(arr)

        if x is None or y is None or c is None:
            env_xy = self.env.env_xy
            values = self.env.values
            x = env_xy[:, 0]
            y = env_xy[:, 1]
            c = values

        x_np = _to_cpu_np(x)
        y_np = _to_cpu_np(y)
        c_np = _to_cpu_np(c)
        path_np = _to_cpu_np(path) if path is not None else None

        if x_range is None:
            x_range = [0, float(self.env.env_x_max)]
        if y_range is None:
            y_range = [0, float(self.env.env_y_max)]

        return self.env.plot_env(
            x=x_np,
            y=y_np,
            c=c_np,
            path=path_np,
            x_range=x_range,
            y_range=y_range,
            value_title=value_title,
        )


def run_tensor_env_smoke_test(
    tensor_envs_dir: str | Path = "tensor_envs",
    env_file: str | Path | None = None,
    env_turn_radius: int = 25,
    steps: int = 4,
    seed: int | None = None,
    device: str | None = None,
    figures_dir: str | Path = "figures",
    plot_path: str | Path | None = None,
) -> None:
    """
    Load a tensor environment, take a few steps, and plot the scenario.
    This is a quick sanity check that envs and belief state live on torch tensors.
    """
    import random
    import numpy as np
    import torch

    torch_device = torch.device(device or "cpu")
    print(f"Using device: {torch_device}, seed: {seed}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    env_dir = Path(tensor_envs_dir)
    figures_dir = Path(figures_dir)

    if env_file is None:
        candidates = sorted(env_dir.glob("*.pt")) + sorted(env_dir.glob("*.pth")) + sorted(env_dir.glob("*.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No tensor env files found in {env_dir}")
        env_path = candidates[0]
    else:
        env_path = Path(env_file)
        if not env_path.is_absolute():
            env_path = env_dir / env_path

    print(f"Loading tensor environments from: {env_path}")

    cfg = Config(
        seed=seed if seed is not None else 0,
        device=str(torch_device),
        env_turn_radius=env_turn_radius,
        tensor_envs_dir=str(env_dir),
        tensor_env_file=str(env_path),
        figures_dir=str(figures_dir),
    )
    _ensure_config_dirs(cfg)
    adapter = EnvAdapter(cfg, return_torch=True)
    print(f"Scenario bank size: {len(adapter.scenario_bank.environments)}")

    scenario = adapter.scenario_bank.sample()
    print(
        "Sampled scenario: "
        f"parameter={scenario.get('parameter')}, depth={scenario.get('depth')}, "
        f"time={scenario.get('time')}, cur_dir={scenario.get('cur_dir')}, "
        f"cur_str={scenario.get('cur_str')}"
    )
    if torch.is_tensor(scenario["coords"]) and torch.is_tensor(scenario["values"]):
        print(f"Scenario tensors on device: coords={scenario['coords'].device}, values={scenario['values'].device}")

    obs, info = adapter.reset(
        random_scenario=scenario,
        env_xy=scenario["coords"],
        values=scenario["values"],
    )
    print(f"Reset done. Obs tensor: shape={tuple(obs.shape)}, device={obs.device}, dtype={obs.dtype}")

    for step_idx in range(steps):
        action = adapter.env.action_space.sample()
        obs, reward, done, info = adapter.step(action)
        truncated = bool(info.get("truncated"))
        reward_val = reward.item() if torch.is_tensor(reward) else reward
        print(
            f"step {step_idx + 1}/{steps}: action={action}, reward={reward_val:.3f}, "
            f"done={done}, truncated={truncated}"
        )
        if done or truncated:
            print("Episode ended early.")
            break

    fig_ax = adapter.plot_env(
        x=scenario["coords"][:, 0],
        y=scenario["coords"][:, 1],
        c=scenario["values"],
        path=adapter.env.sampled_coords[:adapter.env.sample_idx],
    )
    if fig_ax is not None:
        fig, _ = fig_ax
        if plot_path is None:
            plot_path = figures_dir / "last_env_plot.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {plot_path}")
