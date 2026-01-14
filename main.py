"""Main entry point for the AlphaGP project.

Step 0 wires config loading; later steps add env, MCTS, and training.
"""
# %%
from __future__ import annotations

import importlib
from pathlib import Path

import chem_utils
import helpers
import rl_gas_survey_dubins_env
import rl_scenario_bank


def set_seed(seed: int) -> None:
    """Seed RNGs (placeholder)."""
    # TODO: implement determinism controls once training starts.
    pass


def make_run_dir(cfg: "helpers.Config") -> Path:
    """Return the directory for run artifacts (placeholder)."""
    return Path("runs") / cfg.run_name


def _reload_modules() -> None:
    """Reload local modules to pick up edits without restarting Python."""
    importlib.reload(chem_utils)
    importlib.reload(rl_scenario_bank)
    importlib.reload(rl_gas_survey_dubins_env)
    importlib.reload(helpers)


def main() -> None:
    _reload_modules()
    cfg = helpers.load_config()
    # TODO: wire up env/net/mcts/training loops.
    helpers.run_tensor_env_smoke_test(
        tensor_envs_dir=cfg.tensor_envs_dir,
        env_file=cfg.tensor_env_file,
        env_turn_radius=cfg.env_turn_radius,
        device=cfg.device,
        seed=cfg.seed,
        figures_dir=cfg.figures_dir,
    )


if __name__ == "__main__":
    main()

# %%
