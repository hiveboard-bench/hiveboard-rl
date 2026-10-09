#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay expert-bank trajectories in simulation (Viser in the browser by default).

Runs the RL Play task, resets each environment to a chosen bank trajectory's
initial state (valve pose, valve angle, arm joints) and drives it open loop
with that trajectory's recorded joint targets and gripper commands, through the
RL task's own action space (integrated joint increments + binary gripper). When
an episode ends, the environment moves on to the next selected trajectory::

    uv run python scripts/rl/replay_expert_bank.py                       # Viser on :9080
    uv run python scripts/rl/replay_expert_bank.py --num_envs 4 --indices 0,17,42,99
    uv run python scripts/rl/replay_expert_bank.py --bank logs/expert_bank/other.pt --viz newton

Each finished episode prints whether the valve opened in the replay and the
time to open recorded in the bank.
"""

import argparse
import importlib
import os
import sys

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-HiveBoard-Anymal-BallValve-RL-Play-v0")
parser.add_argument("--bank", default=None, help="Bank .pt file (default: the RL task's EXPERT_BANK_PATH).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--indices", default=None, help="Comma-separated trajectory indices (default: random).")
parser.add_argument("--episodes", type=int, default=None, help="Stop after this many episodes (default: run forever).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--viser_port", type=int, default=9080)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
if args.visualizer is None and not getattr(args, "visualizer_explicit", False) and not getattr(args, "headless", False):
    args.visualizer = ["viser"]
sys.argv = [sys.argv[0], *hydra_args]


def main() -> None:
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import expert_bank
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp as rl_mdp

    task_env = importlib.import_module(gym.spec(args.task).kwargs["env_cfg_entry_point"].split(":")[0])
    SUCCESS_TOLERANCE_RAD = task_env.SUCCESS_TOLERANCE_RAD

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    reset_cfg = env_cfg.events.reset_from_bank
    if args.bank:
        reset_cfg.params["path"] = os.path.abspath(args.bank)
    reset_cfg.func = expert_bank.reset_from_expert_bank_in_order
    reset_cfg.params["seed"] = args.seed
    if args.indices:
        reset_cfg.params["order"] = [int(i) for i in args.indices.split(",") if i.strip()]
    if "viser" in (args.visualizer or []):
        from isaaclab_visualizers.viser import ViserVisualizerCfg

        cfgs = env_cfg.sim.visualizer_cfgs
        cfgs = [] if cfgs is None else list(cfgs) if isinstance(cfgs, list) else [cfgs]
        viser_cfg = next((c for c in cfgs if getattr(c, "visualizer_type", None) == "viser"), None)
        if viser_cfg is None:
            cfgs.append(ViserVisualizerCfg(port=args.viser_port))
        else:
            viser_cfg.port = args.viser_port
        env_cfg.sim.visualizer_cfgs = cfgs

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        dev, n_envs = env.device, env.num_envs
        term = env.expert_bank_term
        raw = torch.load(reset_cfg.params["path"], map_location="cpu", weights_only=False)
        t_open = raw["t_open"].float()

        env.reset()
        opened = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        done_episodes = 0
        with torch.inference_mode():
            while True:
                action = expert_bank.expert_action(env)
                prev_idx = term.index.clone()
                _, _, terminated, truncated, _ = env.step(action)
                opened |= rl_mdp.valve_open_success(env, SUCCESS_TOLERANCE_RAD)
                for e in (terminated | truncated).nonzero().flatten().tolist():
                    i = int(prev_idx[e])
                    print(
                        f"[REPLAY] env {e} trajectory {i}: "
                        f"{'opened' if opened[e] else 'NOT opened'} (bank opened at {t_open[i]:.2f}s)",
                        flush=True,
                    )
                    opened[e] = False
                    done_episodes += 1
                if args.episodes is not None and done_episodes >= args.episodes:
                    break
        env.close()


if __name__ == "__main__":
    main()
