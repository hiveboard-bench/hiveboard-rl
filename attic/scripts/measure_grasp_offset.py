#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the expert's grasp TCP pose in the lever's ``lever_pivot`` frame, and
check that the RL task's reward and success terms are reachable.

Replays the cuRobo expert's joint targets from a demonstration through the RL
task's integrated action space at the nominal valve pose. It prints the TCP
pose in the ``lever_pivot`` frame at the first closed-gripper step, which is
``GRASP_OFFSET_POS/QUAT`` in ``tasks/anymal/ball_valve_rl/mdp.py``, then logs
the reward terms until the episode ends. The measurement uses the live
simulated TCP rather than recorded poses, so it stays valid if the TCP frame
convention changes::

    uv run python scripts/rl/measure_grasp_offset.py \\
        --dataset logs/imitation/datasets/anymal_ball_valve_expert_50.hdf5
"""

import argparse
import sys

import gymnasium as gym
import h5py
import isaaclab_hiveboard_rl  # noqa: F401
import torch

import isaaclab.utils.math as math_utils
from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

TASK = "Isaac-HiveBoard-Anymal-BallValve-RL-Play-v0"
# The scripted task runs at dt = 1/200 with decimation 15.
EXPERT_STEP_DT = 15 / 200

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", default="logs/imitation/datasets/anymal_ball_valve_expert_50.hdf5")
parser.add_argument("--demo", default="demo_0")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
if not any(t.startswith(("physics=", "presets=")) for t in hydra_args):
    hydra_args.append("physics=newton_mjwarp")
args.visualizer = []
sys.argv = [sys.argv[0], *hydra_args]

with h5py.File(args.dataset) as f:
    expert_actions = torch.tensor(f["data"][args.demo]["actions"][:], dtype=torch.float32)

env_cfg, _ = resolve_task_config(TASK, None)
env_cfg.scene.num_envs = 1
env_cfg.events.reset_valve_root.params["pose_range"] = {}

with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp

    arm = env.action_manager.get_term("arm_action")
    scale = float(env_cfg.actions.arm_action.scale)
    rewards = env.reward_manager
    expert_actions = expert_actions.to(env.device)
    measured = False
    episode_return = 0.0
    for t in range(int(env.max_episode_length)):
        idx = min(int(t * env.step_dt / EXPERT_STEP_DT), len(expert_actions) - 1)
        action = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
        action[0, :6] = ((expert_actions[idx, :6] - arm._target[0]) / scale).clamp(-1.0, 1.0)
        action[0, 6] = expert_actions[idx, 6]
        _, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward.item()

        if not measured and expert_actions[idx, 6] < 0:
            # First closed-gripper step: the expert has arrived at the grasp.
            tcp_pos, tcp_quat = mdp.tcp_w(env)
            frames = env.scene["target_frame"].data
            i = frames.target_frame_names.index("lever_pivot")
            off_pos, off_quat = math_utils.subtract_frame_transforms(
                frames.target_pos_w.torch[:, i], frames.target_quat_w.torch[:, i], tcp_pos, tcp_quat
            )
            p, q = off_pos[0].tolist(), math_utils.quat_unique(off_quat)[0].tolist()
            print(f"GRASP_OFFSET_POS = ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
            print(f"GRASP_OFFSET_QUAT = ({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})")
            measured = True

        if t % 10 == 0 or terminated.any() or truncated.any():
            dist, ang = mdp.tcp_grasp_error(env)
            terms = {n: rewards._step_reward[0, j].item() for j, n in enumerate(rewards.active_terms)}
            print(
                f"t={t * env.step_dt:5.2f}s dist={dist.item():.4f}m ang={ang.item():.3f}rad "
                f"progress={mdp.valve_progress(env).item():.3f} "
                + " ".join(f"{k}={v:+.3f}" for k, v in terms.items() if abs(v) > 1e-4)
            )
        if terminated.any() or truncated.any():
            outcome = "success" if terminated.any() else "timeout"
            print(f"[replay] {outcome} after {(t + 1) * env.step_dt:.2f}s, return {episode_return:.2f}")
            break
    env.close()
