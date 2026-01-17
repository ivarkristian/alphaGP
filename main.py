"""Main entry point for the AlphaGP project.

Step 0 wires config loading; later steps add env, MCTS, and training.
"""
# %%
from __future__ import annotations

import importlib
import time
from pathlib import Path

import chem_utils
import helpers
import rl_gas_survey_dubins_env
import rl_scenario_bank
import policy_value_net
import EnvAdapterClass
import alphaMCTS


def _build_scenario_bank(
    cfg: "helpers.Config",
    cutoff_percentage_min: float = 6.0,
    cutoff_percentage_max: float = 100.0,
) -> rl_scenario_bank.ScenarioBank:
    if cfg.general.tensor_env_file is None:
        raise ValueError("general.tensor_env_file must be set in config.json.")

    bank = rl_scenario_bank.ScenarioBank(data_dir=str(cfg.general.tensor_envs_dir))
    bank.load_envs(cfg.general.tensor_env_file, device=cfg.general.device)

    sensor_min, sensor_max = cfg.general.sensor_range
    bank.clip_sensor_range(parameter=cfg.general.gas_type, min=sensor_min, max=sensor_max)
    bank.gas_coverage_cutoff(
        cutoff_concentration=cfg.general.gas_threshold,
        cutoff_percentage_min=cutoff_percentage_min,
        cutoff_percentage_max=cutoff_percentage_max,
    )
    return bank


def train(cfg: "helpers.Config") -> None:
    import torch

    from replay_buffer import ReplayBuffer
    from policy_value_net import compute_policy_value_loss

    bank = _build_scenario_bank(cfg)
    env_adapt = EnvAdapterClass.EnvAdapter(cfg, scenario_bank=bank, return_torch=True)
    net = policy_value_net.PolicyValueNet(
        in_channels=cfg.policy_net.in_channels,
        num_actions=cfg.policy_net.num_actions,
        aux_dim=cfg.policy_net.aux_dim,
        trunk_channels=cfg.policy_net.trunk_channels,
        latent_dim=cfg.policy_net.latent_dim,
        head_dim=cfg.policy_net.head_dim,
        value_head_dim=cfg.policy_net.value_head_dim,
    ).to(env_adapt.device)

    train_cfg = cfg.train_config
    optimizer = torch.optim.Adam(net.parameters(), lr=train_cfg.learning_rate)
    buffer = ReplayBuffer(capacity=train_cfg.buffer_capacity, device=env_adapt.device)
    batch_size = train_cfg.batch_size
    train_steps = train_cfg.optimizer_steps
    start_iter = 0
    wandb_run = None
    if train_cfg.wandb_enabled:
        try:
            import wandb
        except ImportError:
            print("wandb is not installed; disabling wandb logging.")
        else:
            wandb_kwargs = {
                "project": train_cfg.wandb_project,
                "entity": train_cfg.wandb_entity,
                "name": cfg.general.run_name,
                "mode": train_cfg.wandb_mode,
                "config": helpers.build_wandb_config(cfg),
            }
            wandb_run = wandb.init(**wandb_kwargs)

    if train_cfg.resume_path:
        ckpt = torch.load(train_cfg.resume_path, map_location=env_adapt.device)
        net.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        buffer.load_state_dict(ckpt["buffer_state"], device=env_adapt.device)
        if "rng_state" in ckpt:
            EnvAdapterClass.EnvAdapter._set_rng_state(ckpt["rng_state"])
        start_iter = int(ckpt.get("iteration", 0) + 1)
        print(f"Resumed from {train_cfg.resume_path} at iteration {start_iter}")

    total_episodes = 0
    selfplay_seconds = 0.0

    for it in range(start_iter, train_cfg.train_num_iterations):
        selfplay_start = time.time()
        episodes = helpers.run_self_play(
            cfg,
            policy_net=net,
            env_adapter=env_adapt,
        )
        selfplay_seconds += time.time() - selfplay_start
        total_episodes += len(episodes)
        for ep in episodes:
            buffer.add_episode(ep)

        if len(buffer) < batch_size:
            eps_per_min = total_episodes / max(selfplay_seconds / 60.0, 1e-6)
            print(
                f"Iteration {it + 1}/{train_cfg.train_num_iterations}: loss=nan, policy=nan, "
                f"value=nan, episodes/min={eps_per_min:.2f} (buffer {len(buffer)}/{batch_size})"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "iteration": it + 1,
                        "episodes_per_min": eps_per_min,
                        "buffer_fill": len(buffer) / buffer.capacity if buffer.capacity else 0.0,
                    },
                    step=it + 1,
                )
            continue

        net.train()
        for _ in range(train_steps):
            batch = buffer.sample_batch(batch_size)
            obs = batch["obs"]
            if isinstance(obs, dict) and "map" in obs:
                obs = dict(obs)
                obs["map"] = obs["map"].to(dtype=torch.float32) / 255.0
                if "loc" in obs:
                    obs["loc"] = obs["loc"].to(dtype=torch.float32)
                if "hdg" in obs:
                    obs["hdg"] = obs["hdg"].to(dtype=torch.float32)

            policy_logits, value_pred = net(obs)
            loss_dict = compute_policy_value_loss(
                net,
                policy_logits,
                value_pred,
                batch["policy"],
                batch["value"],
                l2_weight=train_cfg.l2_weight,
            )
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()

        net.eval()
        eps_per_min = total_episodes / max(selfplay_seconds / 60.0, 1e-6)
        print(
            f"Iteration {it + 1}/{train_cfg.train_num_iterations}: loss={loss_dict['total'].item():.4f}, "
            f"policy={loss_dict['policy'].item():.4f}, value={loss_dict['value'].item():.4f}, "
            f"episodes/min={eps_per_min:.2f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "iteration": it + 1,
                    "loss": loss_dict["total"].item(),
                    "policy_loss": loss_dict["policy"].item(),
                    "value_loss": loss_dict["value"].item(),
                    "episodes_per_min": eps_per_min,
                    "buffer_fill": len(buffer) / buffer.capacity if buffer.capacity else 0.0,
                },
                step=it + 1,
            )

        if train_cfg.checkpoint_every > 0 and (it + 1) % train_cfg.checkpoint_every == 0:
            ckpt_path = Path(train_cfg.checkpoint_dir) / f"checkpoint_iter_{it + 1}.pt"
            ckpt = {
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "buffer_state": buffer.state_dict(),
                "iteration": it,
                "rng_state": EnvAdapterClass.EnvAdapter._capture_rng_state(),
            }
            torch.save(ckpt, ckpt_path)
            fill_ratio = len(buffer) / buffer.capacity if buffer.capacity else 0.0
            print(f"Saved checkpoint to {ckpt_path} (buffer fill {fill_ratio:.1%})")
            #eval_stats = helpers.evaluate(cfg, net, max_steps=cfg.mcts_self_play.max_steps, env_adapter=env_adapt)
            #if wandb_run is not None and eval_stats is not None:
            #    wandb_run.log(
            #        {
            #            "eval_mean": eval_stats.get("mean", 0.0),
            #            "eval_median": eval_stats.get("median", 0.0),
            #        },
            #        step=it + 1,
            #    )

def set_seed(seed: int) -> None:
    """Seed RNGs (placeholder)."""
    # TODO: implement determinism controls once training starts.
    pass


def make_run_dir(cfg: "helpers.Config") -> Path:
    """Return the directory for run artifacts (placeholder)."""
    return Path("runs") / cfg.general.run_name


def _reload_modules() -> None:
    """Reload local modules to pick up edits without restarting Python."""
    importlib.reload(chem_utils)
    importlib.reload(rl_scenario_bank)
    importlib.reload(rl_gas_survey_dubins_env)
    importlib.reload(helpers)
    importlib.reload(policy_value_net)
    importlib.reload(EnvAdapterClass)
    importlib.reload(alphaMCTS)

#def main() -> None:
_reload_modules()
cfg = helpers.load_config()
# TODO: wire up env/net/mcts/training loops.

bank = _build_scenario_bank(cfg, cutoff_percentage_min=6, cutoff_percentage_max=100)

test_envs = False
test_policy_with_envs = False
test_mcts = False
test_self_play = False
do_training = True

if test_envs:
    helpers.run_tensor_env_smoke_test(
        tensor_envs_dir=cfg.general.tensor_envs_dir,
        env_file=cfg.general.tensor_env_file,
        env_turn_radius=cfg.general.env_turn_radius,
        device=cfg.general.device,
        seed=cfg.general.seed,
        figures_dir=cfg.general.figures_dir,
    )

env_adapt = EnvAdapterClass.EnvAdapter(cfg, scenario_bank=bank, return_torch=True)
net = policy_value_net.PolicyValueNet(
    in_channels=cfg.policy_net.in_channels,
    num_actions=cfg.policy_net.num_actions,
    aux_dim=cfg.policy_net.aux_dim,
    trunk_channels=cfg.policy_net.trunk_channels,
    latent_dim=cfg.policy_net.latent_dim,
    head_dim=cfg.policy_net.head_dim,
    value_head_dim=cfg.policy_net.value_head_dim,
).to(env_adapt.device)

print(f"Initialized env adapter on {env_adapt.device}")
print(f"Initialized policy/value net with {sum(p.numel() for p in net.parameters()):,} params")

if test_policy_with_envs:
    helpers.run_policy_smoke_test(
        tensor_envs_dir=cfg.general.tensor_envs_dir,
        env_file=cfg.general.tensor_env_file,
        env_turn_radius=cfg.general.env_turn_radius,
        device=cfg.general.device,
        seed=cfg.general.seed,
        figures_dir=cfg.general.figures_dir,
        policy_net=net,
    )

if test_mcts:
    helpers.run_mcts_smoke_test(
        env_adapter=env_adapt,
        policy_net=net,
        num_simulations=cfg.mcts_test.num_simulations,
        steps=cfg.mcts_test.steps,
        temperature=cfg.mcts_test.temperature,
        cpuct=cfg.mcts_test.cpuct,
        discount=cfg.mcts_test.discount,
        dirichlet_alpha=cfg.mcts_test.dirichlet_alpha,
        dirichlet_epsilon=cfg.mcts_test.dirichlet_epsilon,
        rollout_mode=cfg.mcts_test.rollout_mode,
        rollout_reward_weights=cfg.mcts_test.rollout_reward_weights,
        rollout_var_reduction_scale=cfg.mcts_test.rollout_var_reduction_scale,
        rollout_lengthscale=cfg.mcts_test.rollout_lengthscale,
        tree_depth_printout=cfg.mcts_test.tree_depth_printout,
        tree_depth_max=cfg.mcts_test.tree_depth_max,
        top_k=cfg.mcts_test.top_k,
        plot_path=cfg.mcts_test.plot_path,
    )

if test_self_play:
    episodes = helpers.run_self_play(
        cfg,
        policy_net=net,
        num_episodes=3,
        env_adapter=env_adapt,
    )
    print(f"Completed self-play episodes: {len(episodes)}")

if do_training:
    train(cfg)




#if __name__ == "__main__":
#    main()

# %%
