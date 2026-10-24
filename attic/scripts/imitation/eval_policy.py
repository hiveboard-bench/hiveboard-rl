#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure a trained policy's success rate in the Newton environment.

Robomimic's own rollout evaluation builds a robosuite-style environment from
dataset metadata, which does not exist for a Newton task, so the BC config
disables it. This script is the replacement: it drives the real task with the
checkpoint and reports the task's own ``success`` termination term.

Example:
    uv run python scripts/imitation/eval_policy.py \\
        --checkpoint logs/imitation/runs/bc_spot_lamp/.../models/model_epoch_600.pth \\
        --episodes 50 --num_envs 8
"""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401  (registers the HiveBoard tasks)
import torch
from isaaclab_hiveboard_rl.imitation import RobomimicPolicy, ScriptedExpert, run_rollout

from isaaclab.app import launch_simulation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import add_common_args, build_env_cfg, parse_with_presets  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse the evaluation CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--checkpoint", default=None, help="Robomimic .pth checkpoint to evaluate.")
    parser.add_argument("--episodes", type=int, default=25, help="Episodes to run.")
    parser.add_argument(
        "--expert",
        action="store_true",
        help="Evaluate the scripted expert instead of a checkpoint, to establish the ceiling.",
    )
    args = parse_with_presets(parser)
    if not args.expert and args.checkpoint is None:
        raise SystemExit("Pass --checkpoint, or --expert to measure the scripted baseline.")
    return args


def main() -> int:
    """Evaluate and report the success rate."""
    args = parse_args()
    if args.checkpoint is not None and not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    env_cfg = build_env_cfg(args)
    # No recorder: evaluation reads the success termination term directly, and
    # writing a dataset here would quietly pollute the training corpus.
    env_cfg.recorders = None

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        episode_steps = int(getattr(base, "max_episode_length", 0) or 0)
        budget = max(1, 2 * args.episodes * max(episode_steps, 1) // max(args.num_envs, 1))

        if args.expert:
            expert = ScriptedExpert(base)
            actor = lambda _obs: expert.compute()  # noqa: E731
            label = "scripted expert"
        else:
            policy = RobomimicPolicy(args.checkpoint, device=str(torch.device(base.device)))
            policy.reset()
            actor = lambda obs: policy.act(obs["bc"])  # noqa: E731
            label = os.path.basename(args.checkpoint)

        print(f"[INFO] Evaluating {label} on {args.task} for {args.episodes} episodes.")
        result = run_rollout(
            env,
            actor=actor,
            max_steps=budget,
            stop_after_episodes=args.episodes,
            progress=lambda r: print(
                f"[INFO] step={r.steps} episodes={r.episodes} success_rate={r.success_rate:.1%}",
                flush=True,
            ),
        )
        env.close()

    if result.episodes == 0:
        raise SystemExit(
            f"No episode finished within {result.steps} steps. The policy may never "
            "trigger a termination term; raise --episodes or check the task."
        )

    successes = sum(result.episode_successes)
    print(
        f"\n[RESULT] {label}\n"
        f"  episodes     : {result.episodes}\n"
        f"  successes    : {successes}\n"
        f"  success rate : {result.success_rate:.1%}"
    )
    if result.episodes < args.episodes:
        print(
            f"[WARN] Step budget ended the run at {result.episodes}/{args.episodes} episodes.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
