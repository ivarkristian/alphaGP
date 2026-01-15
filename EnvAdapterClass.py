class EnvAdapter:
    """Thin wrapper around GasSurveyDubinsEnv for MCTS-friendly IO."""
    from helpers import Config

    def __init__(self, cfg: Config, scenario_bank=None, channels=None, return_torch: bool = True) -> None:
        import numpy as np
        import torch

        from rl_gas_survey_dubins_env import GasSurveyDubinsEnv
        from rl_scenario_bank import ScenarioBank

        self.cfg = cfg
        self.device = torch.device(cfg.general.device)
        self.return_torch = return_torch
        self._rng_state = None
        self._restore_rng_on_step = False

        if scenario_bank is None:
            if cfg.general.tensor_env_file is None:
                raise ValueError("Config tensor_env_file is required to load scenarios.")
            self.scenario_bank = ScenarioBank(data_dir=str(cfg.general.tensor_envs_dir))
            self.scenario_bank.load_envs(cfg.general.tensor_env_file, device=self.device)
        else:
            self.scenario_bank = scenario_bank

        channels_arr = np.array([1, 1, 0, 0, 0], dtype=np.uint8) if channels is None else channels

        self.env = GasSurveyDubinsEnv(
            scenario_bank=self.scenario_bank,
            turn_radius=cfg.general.env_turn_radius,
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
        import torch
        from rl_gas_survey_dubins_env import move_with_heading

        n_actions = int(self.env.action_space.n)
        if self.env.return_torch:
            heading = getattr(self.env, "heading_t", None)
            loc_xy = self.env.loc[:2]
            if heading is None:
                heading = torch.as_tensor(self.env.heading, device=self.env.device)
        else:
            heading = self.env.heading
            loc_xy = self.env.loc_xy_np
            if not hasattr(self.env, "loc_xy_np"):
                return list(range(n_actions))

        n_headings = len(heading)
        legal = []
        for action in range(n_actions):
            delta_xy, new_heading = move_with_heading(
                heading_1hot=heading,
                action=action,
                turn_radius=self.env.turn_radius,
                turn_degrees=int(360 / n_headings),
                n_headings=n_headings,
                straight_matches_arc=True,
            )
            new_xy = loc_xy + delta_xy
            if self.env.return_torch:
                out_of_bounds = bool(
                    ((new_xy[0] < 0) | (new_xy[0] > self.env.env_x_max) | (new_xy[1] < 0) | (new_xy[1] > self.env.env_y_max)).item()
                )
            else:
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