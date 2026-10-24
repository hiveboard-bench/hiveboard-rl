#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a robomimic BC policy on a collected HiveBoard dataset.

This is Isaac Lab's ``scripts/imitation_learning/robomimic/train.py`` with the
Isaac Sim boot removed. That script opens the simulator before importing
anything, which costs a minute per call and is pure overhead here: the BC config
sets ``experiment.rollout.enabled = false``, so robomimic never constructs an
environment, and this repository evaluates policies in Newton through
``scripts/imitation/eval_policy.py`` instead. The training itself is robomimic's
own ``train()``, driven by the JSON registered on the task as
``robomimic_bc_cfg_entry_point``.

Example:
    uv run python scripts/imitation/train_bc.py --dataset logs/imitation/datasets/expert_....hdf5
"""

from __future__ import annotations

import argparse
import importlib
import json
import os

DEFAULT_TASK = "Isaac-HiveBoard-Spot-Lamp-v0"
DEFAULT_RUN_DIR = os.path.join("logs", "imitation", "runs")


def resolve_algo_config(task: str, algo: str):
    """Load the robomimic config JSON registered on a gym task.

    Args:
        task: Gym task id.
        algo: Algorithm name, used to build the ``robomimic_<algo>_cfg_entry_point`` key.

    Returns:
        The robomimic ``Config`` built from the registered JSON.

    Raises:
        ValueError: If the task does not register a config for ``algo``.
    """
    import gymnasium as gym
    import isaaclab_hiveboard_rl  # noqa: F401  (registers the HiveBoard tasks)
    from robomimic.config import config_factory

    key = f"robomimic_{algo}_cfg_entry_point"
    entry_point = gym.spec(task.split(":")[-1]).kwargs.get(key)
    if entry_point is None:
        raise ValueError(
            f"Task '{task}' does not register '{key}'. Add it to the task's gym.register kwargs, "
            "pointing at a robomimic config JSON (see tasks/spot/lamp/agents/robomimic/bc.json)."
        )

    if ":" in entry_point:
        module_name, file_name = entry_point.split(":")
        module = importlib.import_module(module_name)
        config_file = os.path.join(os.path.dirname(module.__file__), file_name)
    else:
        config_file = entry_point

    with open(config_file) as handle:
        external = json.load(handle)
    config = config_factory(external["algo_name"])
    with config.values_unlocked():
        config.update(external)
    return config


def train_bc(
    dataset: str,
    *,
    task: str = DEFAULT_TASK,
    algo: str = "bc",
    name: str | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    output_dir: str = DEFAULT_RUN_DIR,
    device: str | None = None,
    normalize_actions: bool = True,
    action_norm: tuple[list[float], list[float]] | None = None,
) -> str:
    """Train a policy and return the path of its final checkpoint.

    Args:
        dataset: HDF5 dataset to train on.
        task: Gym task supplying the robomimic config.
        algo: Algorithm name to look up.
        name: Experiment name. Defaults to the config's own.
        epochs: Override the configured epoch count.
        seed: Override the configured seed.
        output_dir: Root directory for run artifacts.
        device: Torch device. Defaults to robomimic's CUDA-if-available choice.
        normalize_actions: Rescale actions per dimension into ``[-1, 1]`` before
            training. robomimic's actor ends in a ``tanh`` and cannot represent
            anything outside that range, so joint-position tasks need this.
        action_norm: Reuse these ``(minimum, maximum)`` stats instead of deriving
            them from ``dataset``, so successive DAgger rounds share one scale.

    Returns:
        Path to ``model_epoch_<last>.pth`` for the finished run.

    Raises:
        FileNotFoundError: If ``dataset`` does not exist, or the run produced no
            checkpoint (which robomimic otherwise reports only in its log).
    """
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.scripts.train import train as robomimic_train

    dataset = os.path.abspath(dataset)
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    output_root = os.path.abspath(output_dir)
    config = resolve_algo_config(task, algo)
    with config.values_unlocked():
        config.train.data = dataset
        config.train.output_dir = output_root
        if name is not None:
            config.experiment.name = name
        if epochs is not None:
            config.train.num_epochs = epochs
        if seed is not None:
            config.train.seed = seed
        # Guard the invariant this script depends on: with rollouts enabled,
        # robomimic would try to build a robosuite/gym env from the dataset
        # metadata, which does not exist for a Newton task.
        config.experiment.rollout.enabled = False
        # robomimic only checkpoints on its `every_n_epochs` cadence, so a run
        # shorter than that cadence trains and then saves nothing. Pin the final
        # epoch explicitly: every round of DAgger depends on getting a
        # checkpoint back, whatever epoch count the caller chose.
        config.experiment.save.enabled = True
        config.experiment.save.epochs = sorted({*config.experiment.save.epochs, config.train.num_epochs})
        # robomimic's get_exp_dir prompts on stdin when the experiment folder
        # already exists, and offers only "reuse" or "delete". Neither suits an
        # unattended DAgger loop, so keep the name collision-free instead. The
        # directory is left for robomimic itself to create: calling get_exp_dir
        # here would make it exist and trigger that very prompt.
        config.experiment.name = _unique_experiment_name(output_root, config.experiment.name)

    experiment_dir = os.path.join(output_root, config.experiment.name)

    stats = None
    if normalize_actions:
        from isaaclab_hiveboard_rl.imitation.dataset import action_stats, write_normalized_actions

        stats = action_norm or action_stats(dataset)
        # Deliberately not inside experiment_dir: robomimic creates that itself
        # and prompts on stdin if it already exists.
        normalized = os.path.join(output_root, "_action_normalized", f"{config.experiment.name}.hdf5")
        write_normalized_actions(dataset, normalized, stats)
        with config.values_unlocked():
            config.train.data = normalized
        print(f"[INFO] Actions normalized per dimension into [-1, 1] -> {normalized}")

    config.lock()

    torch_device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda) if device is None else device
    print(f"[INFO] Training {algo} on {dataset}")
    print(f"[INFO] Experiment  : {experiment_dir}")

    robomimic_train(config, device=torch_device)

    checkpoint = _latest_checkpoint(experiment_dir)
    if stats is not None:
        from isaaclab_hiveboard_rl.imitation.dataset import save_action_norm

        # Beside the checkpoint, where RobomimicPolicy looks for it. Written
        # after training so the run directory exists.
        save_action_norm(os.path.dirname(checkpoint), stats)
    return checkpoint


def _unique_experiment_name(output_root: str, name: str) -> str:
    """Return ``name``, suffixed if needed so ``output_root/name`` is new."""
    if not os.path.exists(os.path.join(output_root, name)):
        return name
    from datetime import datetime

    return f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def _latest_checkpoint(experiment_dir: str) -> str:
    """Find the highest-epoch checkpoint of the most recent run.

    robomimic lays runs out as ``<experiment_dir>/<timestamp>/models/*.pth``.

    Args:
        experiment_dir: The ``output_dir/experiment_name`` directory.

    Returns:
        Path to the last checkpoint written.

    Raises:
        FileNotFoundError: If training produced no checkpoint.
    """
    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f"Training produced no experiment directory at {experiment_dir}.")

    runs = sorted(entry for entry in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, entry)))
    for run in reversed(runs):
        ckpt_dir = os.path.join(experiment_dir, run, "models")
        if not os.path.isdir(ckpt_dir):
            continue
        checkpoints = sorted(
            (entry for entry in os.listdir(ckpt_dir) if entry.endswith(".pth")),
            key=_epoch_of,
        )
        if checkpoints:
            return os.path.join(ckpt_dir, checkpoints[-1])

    raise FileNotFoundError(
        f"Training finished but wrote no checkpoint under {experiment_dir}. "
        "Check the run's logs/log.txt, and that experiment.save.enabled is true."
    )


def _epoch_of(filename: str) -> int:
    """Sort ``model_epoch_100.pth`` after ``model_epoch_20.pth``."""
    stem = os.path.splitext(filename)[0]
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def main() -> int:
    """Train from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="HDF5 dataset to train on.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task supplying the robomimic config.")
    parser.add_argument("--algo", default="bc", help="Algorithm name registered on the task.")
    parser.add_argument("--name", default=None, help="Experiment name override.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override.")
    parser.add_argument("--seed", type=int, default=None, help="Training seed override.")
    parser.add_argument("--output_dir", default=DEFAULT_RUN_DIR, help="Root directory for run artifacts.")
    args = parser.parse_args()

    checkpoint = train_bc(
        args.dataset,
        task=args.task,
        algo=args.algo,
        name=args.name,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"\n[INFO] Final checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
