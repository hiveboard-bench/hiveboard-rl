#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an RSL-RL agent on a HiveBoard task (Isaac Lab's unified entrypoint).

Registers the HiveBoard tasks, selects Newton MJWarp unless a physics preset
is given, then forwards every argument to ``isaaclab train --rl_library rsl_rl``.

Teacher (PPO on privileged state)::

    uv run python scripts/rl/train.py --task Isaac-HiveBoard-Anymal-BallValve-RL-v0 --num_envs 1024

Student (distilled onto the deployable observations; needs the teacher)::

    uv run python scripts/rl/train.py --task Isaac-HiveBoard-Anymal-BallValve-RL-v0 \\
        --agent rsl_rl_distillation_cfg_entry_point \\
        --checkpoint logs/rsl_rl/anymal_ball_valve_teacher/<run>/model_<it>.pt
"""

import os
import sys

import isaaclab_hiveboard_rl  # noqa: F401  (registers the HiveBoard tasks)
from wandb_checkpoints import best_latest_wandb_checkpoints

from isaaclab_rl import run_train_cli


def with_newton_default(argv: list[str]) -> list[str]:
    """Append ``physics=newton_mjwarp`` unless a preset is already selected."""
    if not any(token.startswith(("physics=", "presets=")) for token in argv):
        argv = [*argv, "physics=newton_mjwarp"]
    return argv


def ensure_wandb_run_id() -> None:
    """Give Isaac Lab a wandb run id up front.

    Isaac Lab's wandb helper otherwise calls ``wandb.util.generate_id``, which
    wandb 0.30 removed. A fresh id per launch keeps runs separate.
    """
    if "WANDB_RUN_ID" in os.environ:
        return
    try:
        from wandb.sdk.lib.runid import generate_id
    except ImportError:
        return
    os.environ["WANDB_RUN_ID"] = generate_id()


if __name__ == "__main__":
    ensure_wandb_run_id()
    with best_latest_wandb_checkpoints():
        sys.exit(run_train_cli(["--rl_library", "rsl_rl", *with_newton_default(sys.argv[1:])]) or 0)
