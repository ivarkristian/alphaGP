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

    def rollout_step(
        self,
        action,
        reward_weights=(0.34, 0.33, 0.33),
        var_reduction_scale: float = 0.5,
        lengthscale: float | None = None,
    ):
        """
        Planning-only step that uses mean/variance layers instead of the true field.
        """
        import torch
        from rl_gas_survey_dubins_env import move_with_heading

        env = self.env
        if not env.return_torch:
            raise ValueError("rollout_step requires env.return_torch=True.")

        env.n_steps += 1
        if torch.is_tensor(action):
            if action.numel() != 1:
                raise ValueError("action tensor must contain a single scalar.")
            action_t = action.to(device=env.device)
        else:
            action_t = torch.as_tensor(action, device=env.device)

        heading_t = getattr(env, "heading_t", None)
        if heading_t is None:
            heading_t = torch.as_tensor(env.heading, device=env.device)

        n_headings = int(heading_t.numel())
        delta_xy, new_heading = move_with_heading(
            heading_1hot=heading_t,
            action=action_t,
            turn_radius=env.turn_radius,
            turn_degrees=int(360 / n_headings),
            n_headings=n_headings,
            straight_matches_arc=True,
        )

        new_xy = env.loc[:2] + delta_xy
        out_of_bounds = bool(
            ((new_xy[0] < 0) | (new_xy[0] > env.env_x_max) | (new_xy[1] < 0) | (new_xy[1] > env.env_y_max)).item()
        )
        if out_of_bounds or env._facing_the_boundary(new_xy, new_heading):
            obs, truncated, info = env._get_obs_truncated_info()
            reward = torch.tensor(-5.0, device=env.device)
            info = dict(info or {})
            info["truncated"] = truncated
            return obs, reward, env.terminated, truncated, info

        zero_noise = torch.zeros(2, device=env.device, dtype=new_xy.dtype)
        sample_coords = env._generate_path_torch(env.loc[:2], heading_t, action_t, zero_noise)

        reward, var_after = self._compute_rollout_reward(
            sample_coords=sample_coords,
            reward_weights=reward_weights,
            var_reduction_scale=var_reduction_scale,
            lengthscale=lengthscale,
        )

        env.loc = torch.stack((new_xy[0], new_xy[1], env.loc[2])).to(dtype=env.loc.dtype, device=env.device)
        env.heading_t = new_heading.to(device=env.device)
        env.pred_var_norm_t = var_after * 255.0
        env.pred_var_t = var_after * torch.tensor(env.sigma2_all, device=env.device, dtype=var_after.dtype)
        if hasattr(env, "location_t"):
            mask_t = (env._coord_x_t - env.loc[0]) ** 2 + (env._coord_y_t - env.loc[1]) ** 2 <= env.location_radius ** 2
            env.location_t.fill_(255)
            env.location_t[mask_t] = 0

        obs, truncated, info = env._get_obs_truncated_info()
        info = dict(info or {})
        info["truncated"] = truncated
        return obs, reward, env.terminated, truncated, info

    def _compute_rollout_reward(
        self,
        sample_coords,
        reward_weights=(0.34, 0.33, 0.33),
        var_reduction_scale: float = 0.5,
        lengthscale: float | None = None,
    ):
        """
        Compute rollout rewards from mean/variance layers (normalized to 0-1).

        - Sample value reward: use mu_norm at sampled points.
        - Surprise reward: use a variance-based proxy (sqrt of sigma2_norm).
        - Variance reduction: paint down variance using a radial kernel.
        """
        import torch

        env = self.env
        mean_map = torch.clamp(env.pred_mu_norm_t, 0, 255).to(dtype=torch.float32) / 255.0
        var_map = torch.clamp(env.pred_var_norm_t, 0, 255).to(dtype=torch.float32) / 255.0

        mean_samples = self._bilinear_sample(mean_map, sample_coords, env.env_x_max, env.env_y_max)
        var_samples = self._bilinear_sample(var_map, sample_coords, env.env_x_max, env.env_y_max)

        value_reward = mean_samples.mean()
        surprise_reward = torch.sqrt(var_samples.clamp(min=0.0)).mean()

        var_before = var_map.mean()
        paint_coords = torch.cat((env.sampled_coords[: env.sample_idx], sample_coords))
        var_after = self._apply_variance_paint(
            var_map,
            paint_coords,
            lengthscale=lengthscale,
            reduction_scale=var_reduction_scale,
        )
        var_after = torch.clamp(var_after, 0.0, 1.0)
        var_red = (var_before - var_after.mean()).clamp(min=0.0)

        weights = torch.as_tensor(reward_weights, device=env.device, dtype=torch.float32)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            weights = torch.full_like(weights, 1.0 / float(weights.numel()))
        else:
            weights = weights / weight_sum

        reward = weights[0] * value_reward + weights[1] * surprise_reward + weights[2] * var_red
        return reward, var_after

    def _bilinear_sample(self, map_t, coords_xy, x_max: float, y_max: float):
        import torch

        if coords_xy.numel() == 0:
            return torch.empty((0,), device=map_t.device, dtype=map_t.dtype)
        h, w = map_t.shape[-2], map_t.shape[-1]
        x = coords_xy[:, 0] / float(x_max) * (w - 1)
        y = coords_xy[:, 1] / float(y_max) * (h - 1)
        x0 = torch.floor(x).to(dtype=torch.long)
        y0 = torch.floor(y).to(dtype=torch.long)
        x1 = torch.clamp(x0 + 1, max=w - 1)
        y1 = torch.clamp(y0 + 1, max=h - 1)
        x0 = torch.clamp(x0, min=0, max=w - 1)
        y0 = torch.clamp(y0, min=0, max=h - 1)

        v00 = map_t[y0, x0]
        v10 = map_t[y0, x1]
        v01 = map_t[y1, x0]
        v11 = map_t[y1, x1]

        wx = (x - x0.to(dtype=map_t.dtype))
        wy = (y - y0.to(dtype=map_t.dtype))
        v0 = v00 * (1 - wx) + v10 * wx
        v1 = v01 * (1 - wx) + v11 * wx
        return v0 * (1 - wy) + v1 * wy

    def _apply_variance_paint(
        self,
        var_map,
        sample_coords,
        lengthscale: float | None,
        reduction_scale: float,
    ):
        import torch

        if sample_coords.numel() == 0:
            return var_map

        env = self.env
        coords = env._coords_flat.to(device=var_map.device, dtype=var_map.dtype)

        # Lengthscale controls how quickly the variance recovers away from the path.
        if lengthscale is None:
            if hasattr(env, "mdl") and env.mdl is not None:
                lengthscale_t = env.mdl.covar_module.base_kernel.lengthscale.mean().to(device=var_map.device, dtype=var_map.dtype)
            else:
                lengthscale_t = torch.tensor(env.turn_radius, device=var_map.device, dtype=var_map.dtype)
        else:
            lengthscale_t = torch.tensor(float(lengthscale), device=var_map.device, dtype=var_map.dtype)

        # Compute per-grid-point distance to all sampled points.
        diff = coords.unsqueeze(0) - sample_coords.to(device=coords.device, dtype=coords.dtype).unsqueeze(1)
        dist2 = (diff ** 2).sum(dim=-1)
        ls2 = torch.clamp(lengthscale_t, min=1e-6) ** 2

        # Convert distances to a Gaussian influence and keep the strongest effect per grid point.
        influence = torch.exp(-0.5 * dist2 / ls2)
        influence = influence.max(dim=0).values

        # Apply a global reduction scale and clamp to avoid negative variances.
        #influence = torch.clamp(influence * float(reduction_scale), 0.0, 1.0)
        influence = torch.clamp(influence, 0.0, 1.0) # no reduction_scale

        var_flat = 1.0 #var_map.view(-1)
        # Reduce variance proportional to the influence; 1.0 means fully erased.
        var_flat = var_flat * (1.0 - influence)
        return var_flat.view_as(var_map)

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
