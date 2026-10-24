#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect scripted-expert demonstrations into a robomimic-layout dataset.

The expert is the task's own ``pose_command`` sequence, so this script does not
teach the robot anything new - it transcribes a solution that already works
into the ``data/demo_<i>/obs/<key>`` + ``actions`` form robomimic trains on.

Example:
    uv run python scripts/imitation/collect_demos.py --num_demos 50 --num_envs 8
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401  (registers the HiveBoard tasks)
from isaaclab_hiveboard_rl.imitation import ScriptedExpert, dataset_stats, run_rollout

from isaaclab.app import launch_simulation

from isaaclab_hiveboard.mdp.recorders import RobomimicRecorderCfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    add_common_args,
    attach_recorder,
    build_env_cfg,
    parse_with_presets,
    require_missing,
)


def parse_args() -> argparse.Namespace:
    """Parse the collection CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--num_demos", type=int, default=50, help="Successful demonstrations to collect.")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR, help="Directory for the HDF5 dataset.")
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Dataset stem without extension. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--keep_failed",
        action="store_true",
        help="Also write rejected episodes, to a separate '<name>_failed.hdf5'.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Abort after this many environment steps. Defaults to enough steps for 4x the requested demos.",
    )
    return parse_with_presets(parser)


def main() -> int:
    """Collect demonstrations and report the resulting dataset."""
    args = parse_args()
    if args.num_demos <= 0:
        raise SystemExit("--num_demos must be greater than zero.")

    env_cfg = build_env_cfg(args)
    dataset_name = args.dataset_name or f"expert_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    dataset_path = attach_recorder(env_cfg, RobomimicRecorderCfg(), args.dataset_dir, dataset_name)
    require_missing(dataset_path)

    if args.keep_failed:
        from isaaclab.managers.recorder_manager import DatasetExportMode

        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped

        if len(base.recorder_manager.active_terms) == 0:
            raise RuntimeError("No active recorder terms; nothing would be written.")
        if "bc" not in base.obs_buf and "bc" not in base.observation_manager.active_terms:
            raise RuntimeError(
                f"Task '{args.task}' has no 'bc' observation group. Add one (see "
                "tasks/spot/lamp/configs/observations.py) before collecting."
            )

        expert = ScriptedExpert(base)
        episode_steps = int(getattr(base, "max_episode_length", 0) or 0)
        max_steps = args.max_steps or max(1, 4 * args.num_demos * max(episode_steps, 1) // max(args.num_envs, 1))

        print(f"[INFO] Task            : {args.task}")
        print(f"[INFO] Environments    : {args.num_envs}")
        print(f"[INFO] Target demos    : {args.num_demos}")
        print(f"[INFO] Dataset         : {dataset_path}")
        print(f"[INFO] Step budget     : {max_steps}")

        def report(result) -> None:
            print(
                f"[INFO] step={result.steps} exported={result.exported_successes}/{args.num_demos} "
                f"rejected={result.exported_failures}",
                flush=True,
            )

        result = run_rollout(
            env,
            actor=lambda _obs: expert.compute(),
            max_steps=max_steps,
            stop_after_successes=args.num_demos,
            progress=report,
        )
        env.close()

    if result.exported_successes == 0:
        raise SystemExit(
            f"Collected no successful demonstrations in {result.steps} steps "
            f"({result.episodes} episodes finished). Check that the scripted "
            f"command solves '{args.task}' by running scripts/play.py first."
        )
    if result.exported_successes < args.num_demos:
        print(
            f"[WARN] Step budget exhausted with {result.exported_successes}/{args.num_demos} demos. "
            "Raise --max_steps or check for stalls.",
            file=sys.stderr,
        )

    print("\n[INFO] Dataset written:\n" + dataset_stats(dataset_path).describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
