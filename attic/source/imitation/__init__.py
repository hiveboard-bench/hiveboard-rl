# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behaviour cloning and DAgger support for HiveBoard tasks.

The pieces here are deliberately split from ``scripts/imitation`` so that the
expert, the rollout loop and the dataset utilities stay importable without
parsing command-line arguments or launching a simulation.
"""

from .dataset import (
    CLEAN_MASK,
    FALLBACK_MASK,
    DatasetStats,
    action_stats,
    dataset_stats,
    demo_used_fallback,
    load_action_norm,
    load_dataset_action_norm,
    merge_datasets,
    rotate_recorder_dataset,
    save_action_norm,
    write_normalized_actions,
)
from .expert import ScriptedExpert
from .policy import RobomimicPolicy
from .rollout import RolloutResult, run_rollout

__all__ = [
    "CLEAN_MASK",
    "FALLBACK_MASK",
    "DatasetStats",
    "RobomimicPolicy",
    "RolloutResult",
    "ScriptedExpert",
    "action_stats",
    "dataset_stats",
    "demo_used_fallback",
    "load_action_norm",
    "load_dataset_action_norm",
    "merge_datasets",
    "rotate_recorder_dataset",
    "run_rollout",
    "save_action_norm",
    "write_normalized_actions",
]
