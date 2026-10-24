#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run DAgger on top of a behaviour-cloned HiveBoard policy.

Each round rolls the current policy out in the environment, asks the scripted
expert what it would have done at every state the policy actually visited,
appends those corrections to the aggregated dataset, and retrains from scratch.
That is the point of DAgger over plain BC: the training distribution stops being
the expert's own trajectory and becomes the states the learner really reaches.

Actions are mixed per environment per step with probability ``beta``, decayed
geometrically each round, so early rounds stay near the expert's distribution
while later ones commit to the learner. The recorded label is always the
expert's, whichever policy actually drove the step.

Example:
    uv run python scripts/imitation/dagger.py \\
        --initial_dataset logs/imitation/datasets/expert_....hdf5 \\
        --rounds 5 --episodes_per_round 40 --num_envs 8
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401  (registers the HiveBoard tasks)
import torch
from isaaclab_hiveboard_rl.imitation import (
    RobomimicPolicy,
    ScriptedExpert,
    dataset_stats,
    merge_datasets,
    rotate_recorder_dataset,
    run_rollout,
)

from isaaclab.app import launch_simulation

from isaaclab_hiveboard.mdp.recorders import RobomimicDaggerRecorderCfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    DEFAULT_RUN_DIR,
    add_common_args,
    attach_recorder,
    build_env_cfg,
    parse_with_presets,
)
from train_bc import train_bc  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse the DAgger CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument(
        "--initial_dataset",
        required=True,
        help="Expert dataset from scripts/imitation/collect_demos.py, used to train round 0.",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of DAgger rounds after the initial BC fit.")
    parser.add_argument(
        "--episodes_per_round",
        type=int,
        default=40,
        help="Episodes to roll out and relabel each round.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per round.")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Probability of acting with the expert in the first round.",
    )
    parser.add_argument(
        "--beta_decay",
        type=float,
        default=0.5,
        help="Multiplier applied to beta after each round.",
    )
    parser.add_argument("--work_dir", default=DEFAULT_DATASET_DIR, help="Directory for per-round datasets.")
    parser.add_argument("--output_dir", default=DEFAULT_RUN_DIR, help="Root directory for training runs.")
    args = parse_with_presets(parser)
    if not 0.0 <= args.beta <= 1.0:
        raise SystemExit("--beta must lie in [0, 1].")
    if not 0.0 <= args.beta_decay <= 1.0:
        raise SystemExit("--beta_decay must lie in [0, 1].")
    return args


def _mixed_actor(policy: RobomimicPolicy, expert: ScriptedExpert, beta: float, device: torch.device):
    """Build an actor that follows the expert with probability ``beta``.

    The coin is flipped per environment per step, which keeps the visited-state
    distribution a genuine mixture rather than a set of whole trajectories drawn
    from one policy or the other.

    Args:
        policy: The current learner.
        expert: The scripted expert.
        beta: Probability of deferring to the expert on any given step.
        device: Device for the mixing mask.

    Returns:
        A callable mapping the observation dict to actions.
    """

    def actor(obs: dict) -> torch.Tensor:
        expert_action = expert.compute()
        if beta >= 1.0:
            return expert_action
        learner_action = policy.act(obs["bc"]).to(expert_action.dtype)
        if beta <= 0.0:
            return learner_action
        use_expert = torch.rand(expert_action.shape[0], 1, device=device) < beta
        return torch.where(use_expert, expert_action, learner_action)

    return actor


def main() -> int:
    """Run the DAgger loop."""
    args = parse_args()
    initial_dataset = os.path.abspath(args.initial_dataset)
    if not os.path.exists(initial_dataset):
        raise SystemExit(
            f"Initial dataset not found: {initial_dataset}\n"
            "Collect one first: uv run python scripts/imitation/collect_demos.py"
        )

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    work_dir = os.path.abspath(os.path.join(args.work_dir, f"dagger_{stamp}"))
    os.makedirs(work_dir, exist_ok=True)

    print(f"[INFO] Initial dataset:\n{dataset_stats(initial_dataset).describe()}\n")

    # Round 0 is plain behaviour cloning on the expert demonstrations.
    aggregated = initial_dataset
    checkpoint = train_bc(
        aggregated,
        task=args.task,
        name=f"dagger_{stamp}_round0",
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"[INFO] Round 0 (BC) checkpoint: {checkpoint}")

    env_cfg = build_env_cfg(args)
    attach_recorder(env_cfg, RobomimicDaggerRecorderCfg(), work_dir, "round_1")

    round_datasets = [initial_dataset]
    beta = args.beta

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        device = torch.device(base.device)
        expert = ScriptedExpert(base)
        episode_steps = int(getattr(base, "max_episode_length", 0) or 0)
        step_budget = max(1, 2 * args.episodes_per_round * max(episode_steps, 1) // max(args.num_envs, 1))

        for round_index in range(1, args.rounds + 1):
            dataset_path = rotate_recorder_dataset(env, work_dir, f"round_{round_index}")
            policy = RobomimicPolicy(checkpoint, device=str(device))
            policy.reset()

            print(f"\n=== DAgger round {round_index}/{args.rounds} (beta={beta:.3f}) ===")
            result = run_rollout(
                env,
                actor=_mixed_actor(policy, expert, beta, device),
                max_steps=step_budget,
                expert=expert,
                stop_after_episodes=args.episodes_per_round,
                progress=lambda r: print(
                    f"[INFO] step={r.steps} episodes={r.episodes} success_rate={r.success_rate:.1%}",
                    flush=True,
                ),
            )
            # Flush the round's completed episodes before the file is reopened.
            base.recorder_manager.export_episodes()
            print(
                f"[INFO] Round {round_index}: {result.episodes} episodes, "
                f"on-policy success rate {result.success_rate:.1%}"
            )

            round_datasets.append(dataset_path)
            aggregated = os.path.join(work_dir, f"aggregated_round_{round_index}.hdf5")
            stats = merge_datasets(round_datasets, aggregated)
            print(f"[INFO] Aggregated dataset: {stats.num_demos} demos, {stats.num_samples} samples")

            checkpoint = train_bc(
                aggregated,
                task=args.task,
                name=f"dagger_{stamp}_round{round_index}",
                epochs=args.epochs,
                seed=args.seed,
                output_dir=args.output_dir,
            )
            print(f"[INFO] Round {round_index} checkpoint: {checkpoint}")
            beta *= args.beta_decay

        env.close()

    print(f"\n[INFO] DAgger complete. Final checkpoint: {checkpoint}")
    print(f"[INFO] Aggregated dataset: {aggregated}")
    print(
        "[INFO] Evaluate it with:\n"
        f"  uv run python scripts/imitation/eval_policy.py --checkpoint {checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
