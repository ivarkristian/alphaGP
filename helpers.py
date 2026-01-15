"""Shared helpers for configuration and utilities."""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping
import EnvAdapterClass


DEFAULT_CONFIG_PATH = Path("config.json")

def _filter_config(cls, data: Mapping[str, Any], *, source: str | None = None) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError("Config data must be a mapping.")
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(key for key in data.keys() if key not in allowed)
    if unknown:
        where = f" in {source}" if source else ""
        raise ValueError(f"Unknown config keys{where}: {', '.join(unknown)}")
    return {key: data[key] for key in allowed if key in data}


@dataclass(frozen=True)
class GeneralConfig:
    seed: int = 123
    device: str = "cpu"
    run_name: str = "debug"
    env_turn_radius: float = 250.0
    tensor_envs_dir: str = "tensor_envs"
    tensor_env_file: str | None = None
    figures_dir: str = "figures"
    gas_type: str = "pCO2"
    gas_threshold: float = 550.0
    sensor_range: tuple[float, float] = (0.0, 2000.0)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str | None = None) -> "GeneralConfig":
        filtered = _filter_config(cls, data, source=source)
        if "sensor_range" in filtered and isinstance(filtered["sensor_range"], list):
            filtered["sensor_range"] = tuple(filtered["sensor_range"])
        return cls(**filtered)


@dataclass(frozen=True)
class PolicyNetConfig:
    in_channels: int = 2
    num_actions: int = 3
    aux_dim: int = 10
    trunk_channels: tuple[int, ...] = (32, 64, 128)
    latent_dim: int = 256
    head_dim: int = 128
    value_head_dim: int = 128

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str | None = None) -> "PolicyNetConfig":
        filtered = _filter_config(cls, data, source=source)
        if "trunk_channels" in filtered and isinstance(filtered["trunk_channels"], list):
            filtered["trunk_channels"] = tuple(filtered["trunk_channels"])
        return cls(**filtered)


@dataclass(frozen=True)
class MCTSTestConfig:
    num_simulations: int = 50
    cpuct: float = 1.5
    discount: float = 1.0
    temperature: float = 1.0
    dirichlet_alpha: float | None = None
    dirichlet_epsilon: float = 0.0
    steps: int = 3
    tree_depth_printout: int = 2
    tree_depth_max: int | None = None
    top_k: int = 3
    plot_path: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str | None = None) -> "MCTSTestConfig":
        return cls(**_filter_config(cls, data, source=source))


@dataclass(frozen=True)
class MCTSConfig:
    num_simulations: int = 50
    cpuct: float = 1.5
    discount: float = 1.0
    temperature: float = 1.0
    dirichlet_alpha: float | None = None
    dirichlet_epsilon: float = 0.0
    tree_depth_max: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str | None = None) -> "MCTSConfig":
        return cls(**_filter_config(cls, data, source=source))


@dataclass(frozen=True)
class MCTSSelfPlayConfig:
    num_episodes: int = 1
    max_steps: int | None = None
    temperature: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str | None = None) -> "MCTSSelfPlayConfig":
        return cls(**_filter_config(cls, data, source=source))


@dataclass(frozen=True)
class TrainConfig:
    buffer_capacity: int = 10_000
    batch_size: int = 32
    train_steps: int = 10
    learning_rate: float = 1e-3
    l2_weight: float = 1e-4
    train_num_iterations: int = 5
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1
    resume_path: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str | None = None) -> "TrainConfig":
        return cls(**_filter_config(cls, data, source=source))


@dataclass(frozen=True)
class ParallelConfig:
    num_workers: int = 2
    episodes_per_iter: int = 4
    queue_maxsize: int = 16
    weight_sync_every: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str | None = None) -> "ParallelConfig":
        return cls(**_filter_config(cls, data, source=source))
@dataclass(frozen=True)
class Config:
    general: GeneralConfig = GeneralConfig()
    policy_net: PolicyNetConfig = PolicyNetConfig()
    mcts_test: MCTSTestConfig = MCTSTestConfig()
    mcts: MCTSConfig = MCTSConfig()
    mcts_self_play: MCTSSelfPlayConfig = MCTSSelfPlayConfig()
    train_config: TrainConfig = TrainConfig()
    parallel: ParallelConfig = ParallelConfig()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str | None = None) -> "Config":
        if not isinstance(data, Mapping):
            raise TypeError("Config data must be a mapping.")
        allowed = {
            "general",
            "policy_net",
            "mcts_test",
            "mcts",
            "mcts_self_play",
            "train_config",
            "parallel",
        }
        unknown = sorted(key for key in data.keys() if key not in allowed)
        if unknown:
            where = f" in {source}" if source else ""
            raise ValueError(f"Unknown config keys{where}: {', '.join(unknown)}")
        general = GeneralConfig.from_dict(data.get("general", {}), source="general")
        policy_net = PolicyNetConfig.from_dict(data.get("policy_net", {}), source="policy_net")
        mcts_test = MCTSTestConfig.from_dict(data.get("mcts_test", {}), source="mcts_test")
        mcts = MCTSConfig.from_dict(data.get("mcts", {}), source="mcts")
        mcts_self_play = MCTSSelfPlayConfig.from_dict(data.get("mcts_self_play", {}), source="mcts_self_play")
        train_config = TrainConfig.from_dict(data.get("train_config", {}), source="train_config")
        parallel = ParallelConfig.from_dict(data.get("parallel", {}), source="parallel")
        return cls(
            general=general,
            policy_net=policy_net,
            mcts_test=mcts_test,
            mcts=mcts,
            mcts_self_play=mcts_self_play,
            train_config=train_config,
            parallel=parallel,
        )


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load a config JSON file and return a Config object."""
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = Config.from_dict(data, source=str(path))
    _ensure_config_dirs(cfg)
    return cfg

def printd(cfg: Config, string: None) -> None:
    if cfg.general.run_name == "debug":
        print(string)
    return

def _ensure_config_dirs(cfg: Config) -> None:
    """Ensure config-specified directories exist."""
    for dir_path in (
        cfg.general.tensor_envs_dir,
        cfg.general.figures_dir,
        cfg.train_config.checkpoint_dir,
    ):
        if dir_path is None:
            continue
        Path(dir_path).mkdir(parents=True, exist_ok=True)


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
        general=GeneralConfig(
            seed=seed if seed is not None else 0,
            device=str(torch_device),
            env_turn_radius=env_turn_radius,
            tensor_envs_dir=str(env_dir),
            tensor_env_file=str(env_path),
            figures_dir=str(figures_dir),
        )
    )
    _ensure_config_dirs(cfg)
    adapter = EnvAdapterClass.EnvAdapter(cfg, return_torch=True)
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


def run_policy_smoke_test(
    tensor_envs_dir: str | Path = "tensor_envs",
    env_file: str | Path | None = None,
    env_turn_radius: int = 25,
    steps: int = 4,
    seed: int | None = None,
    device: str | None = None,
    figures_dir: str | Path = "figures",
    plot_path: str | Path | None = None,
    num_actions: int | None = None,
    action_mode: str = "argmax", # "sample"
    temperature: float = 1.0,
    policy_net=None,
) -> None:
    """
    Smoke test that runs a policy/value net to select actions and plots the result.
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
    adapter = EnvAdapterClass.EnvAdapter(cfg, return_torch=True)

    net = policy_net
    if net is None:
        raise ValueError("policy_net must be provided by the caller.")
    net = net.to(torch_device)
    net.eval()

    env_actions = int(adapter.env.action_space.n)
    if num_actions is None and hasattr(net, "policy_head"):
        num_actions = net.policy_head.out_features
    if num_actions is None:
        raise ValueError("num_actions must be provided when policy_net lacks a policy_head.")
    assert int(num_actions) == env_actions, (
        f"policy num_actions={num_actions} but env has {env_actions}"
    )

    scenario = adapter.scenario_bank.sample()
    obs, _info = adapter.env.reset(
        random_scenario=scenario,
        env_xy=scenario["coords"],
        values=scenario["values"],
    )
    print(f"Reset done. Obs map shape: {tuple(obs['map'].shape)}")

    def select_action(policy_logits: torch.Tensor) -> int:
        if action_mode == "sample" and temperature > 0:
            logits = policy_logits / float(temperature)
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())
        return int(torch.argmax(policy_logits, dim=-1).item())

    with torch.no_grad():
        for step_idx in range(steps):
            policy_logits, value = net(obs)
            action = select_action(policy_logits)
            obs, reward, terminated, truncated, info = adapter.env.step(action)
            reward_val = reward.item() if torch.is_tensor(reward) else reward
            print(
                f"step {step_idx + 1}/{steps}: action={action}, reward={reward_val:.3f}, "
                f"value={value.item():.3f}, done={terminated}, truncated={truncated}"
            )
            if terminated or truncated:
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
            plot_path = figures_dir / "policy_smoke_plot.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {plot_path}")


def _policy_with_temperature(policy: "torch.Tensor", temperature: float) -> "torch.Tensor":
    import torch

    if temperature <= 0:
        probs = torch.zeros_like(policy)
        probs[torch.argmax(policy)] = 1.0
        return probs
    if abs(temperature - 1.0) < 1e-6:
        return policy
    scaled = policy.pow(1.0 / temperature)
    total = scaled.sum()
    if total <= 0:
        return torch.full_like(policy, 1.0 / policy.numel())
    return scaled / total


def self_play_episode(cfg: Config, env_adapter, mcts, max_steps: int | None = None, temperature: float | None = None, reset_kwargs=None):
    """
    Run a single self-play episode using MCTS for action selection.

    Returns a list of (obs, policy, reward, done, value) tuples.
    """
    import torch

    if reset_kwargs is None:
        reset_kwargs = {}

    obs, _info = env_adapter.env.reset(**reset_kwargs)
    episode = []
    done = False
    step_limit = max_steps if max_steps is not None else getattr(env_adapter.env, "n_steps_max", None)

    step_idx = 0
    while not done and (step_limit is None or step_idx < step_limit):
        policy, root_value = mcts.run_search(env_adapter)
        temp = temperature if temperature is not None else mcts.temperature
        probs = _policy_with_temperature(policy, temp)
        action = int(torch.multinomial(probs, num_samples=1).item())
        action_t = torch.as_tensor(action, device=env_adapter.device)

        next_obs, reward, terminated, truncated, _info = env_adapter.env.step(action_t)
        done = bool(terminated or truncated)

        episode.append((obs, policy.detach(), reward, done, root_value.detach()))
        obs = next_obs
        step_idx += 1
        if done:
            break

    return episode


def build_mcts_from_config(cfg: Config, policy_net, device=None):
    """Create an MCTS instance using cfg.mcts settings."""
    from alphaMCTS import MCTS

    mcts_cfg = cfg.mcts
    return MCTS(
        policy_net,
        num_simulations=mcts_cfg.num_simulations,
        cpuct=mcts_cfg.cpuct,
        discount=mcts_cfg.discount,
        temperature=mcts_cfg.temperature,
        dirichlet_alpha=mcts_cfg.dirichlet_alpha,
        dirichlet_epsilon=mcts_cfg.dirichlet_epsilon,
        tree_depth_max=mcts_cfg.tree_depth_max,
        device=device,
    )

def run_self_play(
    cfg: Config,
    policy_net,
    num_episodes: int | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    env_adapter=None,
):
    """
    Run multiple self-play episodes using cfg.mcts parameters.

    Returns a list of episodes.
    """
    if env_adapter is None:
        env_adapter = EnvAdapterClass.EnvAdapter(cfg, return_torch=True)

    policy_net = policy_net.to(env_adapter.device)
    policy_net.eval()
    mcts = build_mcts_from_config(cfg, policy_net, device=env_adapter.device)

    play_cfg = cfg.mcts_self_play
    episodes_target = num_episodes if num_episodes is not None else play_cfg.num_episodes
    steps_target = max_steps if max_steps is not None else play_cfg.max_steps
    temp_target = temperature if temperature is not None else play_cfg.temperature

    episodes = []
    printd(cfg, "Running self play")
    for i in range(episodes_target):
        printd(cfg, f"Episode #{i}")
        episode = self_play_episode(cfg,
            env_adapter,
            mcts,
            max_steps=steps_target,
            temperature=temp_target,
        )
        episodes.append(episode)
    return episodes


def evaluate(
    cfg: Config,
    policy_net,
    num_episodes: int = 5,
    seed_base: int | None = None,
    max_steps: int | None = None,
    temperature: float = 0.0,
    env_adapter=None,
) -> dict[str, float]:
    """
    Run fixed-seed evaluation episodes and report mean/median return.

    Seeding random/NumPy/torch before each reset ensures deterministic scenario
    selection, rotation, and translation.
    """
    import numpy as np
    import random
    import torch

    if env_adapter is None:
        env_adapter = EnvAdapterClass.EnvAdapter(cfg, return_torch=True)

    rng_state = EnvAdapterClass.EnvAdapter._capture_rng_state()

    policy_net = policy_net.to(env_adapter.device)
    policy_net.eval()
    mcts = build_mcts_from_config(cfg, policy_net, device=env_adapter.device)

    if seed_base is None:
        seed_base = cfg.general.seed

    returns = []
    for ep_idx in range(num_episodes):
        seed = int(seed_base + ep_idx)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if env_adapter.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        env_adapter.env.reset()
        total_reward = 0.0
        step_limit = max_steps if max_steps is not None else getattr(env_adapter.env, "n_steps_max", None)

        done = False
        step_idx = 0
        while not done and (step_limit is None or step_idx < step_limit):
            policy, _root_value = mcts.run_search(env_adapter)
            probs = _policy_with_temperature(policy, temperature)
            if temperature > 0:
                action = int(torch.multinomial(probs, num_samples=1).item())
            else:
                action = int(torch.argmax(probs).item())
            action_t = torch.as_tensor(action, device=env_adapter.device)
            _obs, reward, terminated, truncated, _info = env_adapter.env.step(action_t)
            reward_val = float(reward.item()) if torch.is_tensor(reward) else float(reward)
            total_reward += reward_val
            done = bool(terminated or truncated)
            step_idx += 1

        returns.append(total_reward)

    mean_return = float(np.mean(returns)) if returns else 0.0
    median_return = float(np.median(returns)) if returns else 0.0
    print(f"Evaluation over {num_episodes} episodes: mean={mean_return:.3f}, median={median_return:.3f}")
    EnvAdapterClass.EnvAdapter._set_rng_state(rng_state)
    return {"mean": mean_return, "median": median_return}


def run_mcts_smoke_test(
    env_adapter: "EnvAdapter",
    policy_net,
    num_simulations: int = 50,
    steps: int = 3,
    temperature: float = 1.0,
    cpuct: float = 1.5,
    discount: float = 1.0,
    dirichlet_alpha: float | None = None,
    dirichlet_epsilon: float = 0.0,
    tree_depth_printout: int = 2,
    tree_depth_max: int | None = None,
    top_k: int = 3,
    plot_path: str | Path | None = None,
) -> None:
    """
    Run MCTS rollouts from the current env state, print tree stats, and plot the path.

    Parameters
    ----------
    env_adapter:
        EnvAdapter with return_torch=True. MCTS clones this adapter to simulate rollouts
        without mutating the root environment state.
    policy_net:
        Policy/value network used to expand nodes and estimate leaf values.
    num_simulations:
        Number of rollouts per MCTS step.
    steps:
        Number of MCTS-driven actions to execute in the real environment.
    temperature:
        Softmax temperature for visit-count policy (higher = more exploratory).
    cpuct:
        Exploration constant for PUCT action selection.
    discount:
        Discount factor used when backing up leaf values.
    dirichlet_alpha, dirichlet_epsilon:
        Optional root noise to encourage exploration; applied to the root prior only.
    tree_depth_printout:
        Max depth of tree summaries printed to stdout.
    tree_depth_max:
        Maximum tree depth for MCTS rollouts (None = unlimited).
    top_k:
        Number of most-visited actions shown per tree level.
    plot_path:
        Output file for the final plot. Uses cfg.general.figures_dir when omitted.
    """
    import torch

    from alphaMCTS import MCTS

    if not getattr(env_adapter.env, "return_torch", False):
        raise ValueError("run_mcts_smoke_test requires env.return_torch=True.")

    policy_net = policy_net.to(env_adapter.device)
    policy_net.eval()

    env_actions = int(env_adapter.env.action_space.n)
    if hasattr(policy_net, "policy_head"):
        net_actions = int(policy_net.policy_head.out_features)
        if net_actions != env_actions:
            raise ValueError(f"policy net has {net_actions} actions, env has {env_actions}.")

    if hasattr(env_adapter, "cfg"):
        _ensure_config_dirs(env_adapter.cfg)

    scenario = env_adapter.scenario_bank.sample()
    env_adapter.env.reset(
        random_scenario=scenario,
        env_xy=scenario["coords"],
        values=scenario["values"],
    )

    mcts = MCTS(
        policy_net,
        num_simulations=num_simulations,
        cpuct=cpuct,
        discount=discount,
        temperature=temperature,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        tree_depth_max=tree_depth_max,
        device=env_adapter.device,
    )

    def print_tree(node, depth=0):
        if depth >= tree_depth_printout:
            return
        children = sorted(
            node.children.items(),
            key=lambda item: float(item[1].visit_count.item()),
            reverse=True,
        )
        indent = "  " * depth
        for action, child in children[:top_k]:
            q_val = float(child.value().item())
            n_val = float(child.visit_count.item())
            p_val = float(child.prior.item())
            print(f"{indent}a={action} N={n_val:.0f} Q={q_val:.3f} P={p_val:.3f}")
            print_tree(child, depth + 1)

    def _compute_action_paths(policy_probs: torch.Tensor):
        env = env_adapter.env
        heading = getattr(env, "heading_t", None)
        if heading is None:
            heading = torch.as_tensor(env.heading, device=env.device)
        loc_xy = env.loc[:2]
        zero_noise = torch.zeros(2, device=env.device, dtype=loc_xy.dtype)
        paths = {}
        for action in range(int(env.action_space.n)):
            coords_t = env._generate_path_torch(loc_xy, heading, torch.tensor(action, device=env.device), zero_noise)
            paths[action] = coords_t.detach().cpu().numpy()
        return paths

    def _plot_mcts_step(step_idx: int, policy_probs: torch.Tensor, action_paths: dict[int, "np.ndarray"]) -> None:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors

        fig_ax = env_adapter.plot_env(
            x=env_adapter.env.env_xy[:, 0],
            y=env_adapter.env.env_xy[:, 1],
            c=env_adapter.env.values,
            path=env_adapter.env.sampled_coords[: env_adapter.env.sample_idx],
        )
        if fig_ax is None:
            return
        fig, ax = fig_ax

        probs_np = policy_probs.detach().cpu().numpy()
        norm = colors.Normalize(vmin=0.0, vmax=1.0)
        cmap = cm.get_cmap("viridis")

        for action, coords in action_paths.items():
            prob = float(probs_np[action]) if action < len(probs_np) else 0.0
            color = cmap(norm(prob))
            ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1.0 + 1.0 * prob)
            ax.scatter(coords[-1, 0], coords[-1, 1], color=color, s=25)
            ax.text(coords[-1, 0], coords[-1, 1], f"a{action}:{prob:.2f}", fontsize=7, color=color)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(np.array([]))
        fig.set_size_inches(*(fig.get_size_inches() * 0.85), forward=True)
        fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            fraction=0.05,
            pad=0.12,
            label="MCTS action prob",
        )

        out_path = (
            Path(env_adapter.cfg.general.figures_dir) / f"mcts_step_{step_idx + 1}.png"
            if plot_path is None
            else Path(plot_path).with_name(f"{Path(plot_path).stem}_step_{step_idx + 1}{Path(plot_path).suffix}")
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved step plot to: {out_path}")

    for step_idx in range(steps):
        policy, root_value, root_node = mcts.run_search(env_adapter, return_root=True)
        action_paths = _compute_action_paths(policy)
        policy_np = policy.detach().cpu().numpy()
        print(f"MCTS step {step_idx + 1}/{steps} root_value={float(root_value.item()):.3f}")
        for action_idx, prob in enumerate(policy_np):
            print(f"  action {action_idx}: prob={prob:.3f}")
        print("Tree summary:")
        print_tree(root_node, depth=0)

        action = int(torch.argmax(policy).item())
        _obs, reward, terminated, truncated, _info = env_adapter.env.step(
            torch.as_tensor(action, device=env_adapter.device)
        )
        reward_val = float(reward.item()) if torch.is_tensor(reward) else float(reward)
        print(
            f"  chose action={action}, reward={reward_val:.3f}, "
            f"terminated={terminated}, truncated={truncated}"
        )
        _plot_mcts_step(step_idx, policy, action_paths)
        if terminated or truncated:
            break

    fig_ax = env_adapter.plot_env(
        x=env_adapter.env.env_xy[:, 0],
        y=env_adapter.env.env_xy[:, 1],
        c=env_adapter.env.values,
        path=env_adapter.env.sampled_coords[: env_adapter.env.sample_idx],
    )
    if fig_ax is None:
        return
    fig, _ = fig_ax
    if plot_path is None:
        plot_path = Path(env_adapter.cfg.general.figures_dir) / "mcts_smoke_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {plot_path}")
