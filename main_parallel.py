"""Parallel self-play entry point: one learner + N workers."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import torch

import helpers
import policy_value_net
import rl_scenario_bank
import EnvAdapterClass
from replay_buffer import ReplayBuffer
from policy_value_net import compute_policy_value_loss


def _build_scenario_bank(cfg: helpers.Config) -> rl_scenario_bank.ScenarioBank:
    if cfg.general.tensor_env_file is None:
        raise ValueError("general.tensor_env_file must be set in config.json.")
    bank = rl_scenario_bank.ScenarioBank(data_dir=str(cfg.general.tensor_envs_dir))
    bank.load_envs(cfg.general.tensor_env_file, device=cfg.general.device)
    sensor_min, sensor_max = cfg.general.sensor_range
    bank.clip_sensor_range(parameter=cfg.general.gas_type, min=sensor_min, max=sensor_max)
    bank.gas_coverage_cutoff(
        cutoff_concentration=cfg.general.gas_threshold,
        cutoff_percentage_min=3,
        cutoff_percentage_max=6,
    )
    return bank


def _to_cpu(obj: Any):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _drain_queue(queue: mp.Queue):
    latest = None
    try:
        while True:
            latest = queue.get_nowait()
    except Exception:
        return latest
    return latest


def worker_loop(
    worker_id: int,
    cfg: helpers.Config,
    episode_queue: mp.Queue,
    weight_queue: mp.Queue,
    stop_event: mp.Event,
):
    import random
    import numpy as np

    torch.set_num_threads(1)
    seed = int(cfg.general.seed + worker_id * 1000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    net.eval()

    latest = _drain_queue(weight_queue)
    if latest is not None:
        net.load_state_dict(latest)

    mcts = helpers.build_mcts_from_config(cfg, net, device=env_adapt.device)
    while not stop_event.is_set():
        latest = _drain_queue(weight_queue)
        if latest is not None:
            net.load_state_dict(latest)

        episode = helpers.self_play_episode(
            cfg,
            env_adapt,
            mcts,
            max_steps=cfg.mcts_self_play.max_steps,
            temperature=cfg.mcts_self_play.temperature,
        )
        episode_cpu = _to_cpu(episode)
        episode_queue.put(episode_cpu)


def main() -> None:
    cfg = helpers.load_config()
    parallel_cfg = cfg.parallel
    train_cfg = cfg.train_config

    mp.set_start_method("spawn", force=True)

    episode_queue = mp.Queue(maxsize=parallel_cfg.queue_maxsize)
    weight_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

    # Initialize learner-side network and buffer.
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
    optimizer = torch.optim.Adam(net.parameters(), lr=train_cfg.learning_rate)
    buffer = ReplayBuffer(capacity=train_cfg.buffer_capacity, device=env_adapt.device)

    start_iter = 0
    if train_cfg.resume_path:
        ckpt = torch.load(train_cfg.resume_path, map_location=env_adapt.device)
        net.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        buffer.load_state_dict(ckpt["buffer_state"], device=env_adapt.device)
        if "rng_state" in ckpt:
            EnvAdapterClass.EnvAdapter._set_rng_state(ckpt["rng_state"])
        start_iter = int(ckpt.get("iteration", 0) + 1)

    # Broadcast initial weights.
    weight_queue.put(_to_cpu(net.state_dict()))

    workers = []
    for worker_id in range(parallel_cfg.num_workers):
        proc = mp.Process(
            target=worker_loop,
            args=(worker_id, cfg, episode_queue, weight_queue, stop_event),
            daemon=True,
        )
        proc.start()
        workers.append(proc)

    try:
        for it in range(start_iter, train_cfg.train_num_iterations):
            episodes_needed = parallel_cfg.episodes_per_iter
            episodes = []
            while len(episodes) < episodes_needed:
                episodes.append(episode_queue.get())

            for ep in episodes:
                buffer.add_episode(ep)

            if len(buffer) < train_cfg.batch_size:
                continue

            net.train()
            for _ in range(train_cfg.optimizer_steps):
                batch = buffer.sample_batch(train_cfg.batch_size)
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
            if parallel_cfg.weight_sync_every > 0 and (it + 1) % parallel_cfg.weight_sync_every == 0:
                while not weight_queue.empty():
                    _ = weight_queue.get()
                weight_queue.put(_to_cpu(net.state_dict()))

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
                helpers.evaluate(cfg, net, env_adapter=env_adapt)
    finally:
        stop_event.set()
        for proc in workers:
            proc.join(timeout=5)


if __name__ == "__main__":
    main()
