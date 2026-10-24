# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect and aggregate the HDF5 datasets robomimic trains on.

DAgger produces one dataset per round. Robomimic trains on a single file, so
rounds have to be concatenated; that is what :func:`merge_datasets` does, and
it is the only place demo numbering is allowed to be rewritten.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import h5py


@dataclass
class DatasetStats:
    """Summary of a robomimic-layout HDF5 dataset."""

    path: str
    num_demos: int
    num_samples: int
    num_successes: int
    obs_keys: dict[str, tuple[int, ...]] = field(default_factory=dict)
    action_dim: int | None = None
    env_name: str | None = None
    fallback_demos: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of demos flagged successful, or NaN when none are flagged."""
        if self.num_demos == 0:
            return float("nan")
        return self.num_successes / self.num_demos

    def describe(self) -> str:
        """Render a one-block human-readable summary."""
        lines = [
            f"{self.path}",
            f"  env            : {self.env_name}",
            f"  demos          : {self.num_demos}",
            f"  samples        : {self.num_samples}",
            f"  successful     : {self.num_successes} ({self.success_rate:.1%})",
            f"  action dim     : {self.action_dim}",
            "  observations   :",
        ]
        lines.extend(f"    {key:<18} {shape}" for key, shape in sorted(self.obs_keys.items()))
        lines.append(f"  expert fallback: {len(self.fallback_demos)} demos")
        lines.extend(f"    {name}" for name in self.fallback_demos)
        return "\n".join(lines)


def dataset_stats(path: str) -> DatasetStats:
    """Summarize a dataset without loading its arrays into memory.

    Args:
        path: Path to the HDF5 dataset.

    Returns:
        The dataset summary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file has no ``data`` group.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"{path} has no 'data' group; it is not a robomimic-layout dataset.")
        data = handle["data"]

        env_name = None
        if "env_args" in data.attrs:
            env_name = json.loads(data.attrs["env_args"]).get("env_name")

        demo_names = [name for name in data if name.startswith("demo_")]
        num_samples = 0
        num_successes = 0
        obs_keys: dict[str, tuple[int, ...]] = {}
        action_dim = None
        fallback_demos: list[str] = []

        for name in sorted(demo_names, key=_demo_sort_key):
            demo = data[name]
            num_samples += int(demo.attrs.get("num_samples", 0))
            num_successes += int(bool(demo.attrs.get("success", False)))
            if not obs_keys and "obs" in demo:
                obs_keys = {key: tuple(demo["obs"][key].shape[1:]) for key in demo["obs"]}
            if action_dim is None and "actions" in demo:
                action_dim = int(demo["actions"].shape[-1])
            if demo_used_fallback(demo):
                fallback_demos.append(name)

    return DatasetStats(
        path=path,
        num_demos=len(demo_names),
        num_samples=num_samples,
        num_successes=num_successes,
        obs_keys=obs_keys,
        action_dim=action_dim,
        env_name=env_name,
        fallback_demos=fallback_demos,
    )


#: robomimic filter keys written by :func:`merge_datasets`. Pass one as
#: ``train.hdf5_filter_key`` to train on, or exclude, fallback-labelled demos.
FALLBACK_MASK = "expert_fallback"
CLEAN_MASK = "expert_clean"


def demo_used_fallback(demo: h5py.Group) -> bool:
    """Whether any step of ``demo`` was labelled by a cuRobo fallback plan.

    Demos recorded before the ``expert_fallback`` key existed count as clean.
    """
    return "expert_fallback" in demo and bool(demo["expert_fallback"][()].any())


def merge_datasets(inputs: list[str], output: str, env_name: str | None = None) -> DatasetStats:
    """Concatenate robomimic datasets into one file, renumbering demos.

    Args:
        inputs: Datasets to merge, in order. Later files keep their contents;
            only the ``demo_<i>`` group names are rewritten so ids stay unique.
        output: Path of the merged dataset. Overwritten if it exists.
        env_name: Environment name for ``env_args``. Defaults to the first
            input's, so a merged DAgger dataset stays loadable by robomimic.

    Returns:
        Stats for the merged dataset.

    Raises:
        ValueError: If ``inputs`` is empty, or the inputs disagree on their
            observation keys, which would silently produce an untrainable file.
    """
    if not inputs:
        raise ValueError("merge_datasets needs at least one input dataset.")
    missing = [path for path in inputs if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Datasets not found: {missing}")

    output_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(output_dir, exist_ok=True)

    reference_keys: set[str] | None = None
    demo_index = 0
    total_samples = 0
    masks: dict[str, list[str]] = {FALLBACK_MASK: [], CLEAN_MASK: []}

    with h5py.File(output, "w") as dst:
        dst.attrs["format_version"] = 1
        data_group = dst.create_group("data")

        for path in inputs:
            with h5py.File(path, "r") as src:
                src_data = src["data"]
                if env_name is None and "env_args" in src_data.attrs:
                    env_name = json.loads(src_data.attrs["env_args"]).get("env_name")

                for name in sorted(src_data, key=_demo_sort_key):
                    if not name.startswith("demo_"):
                        continue
                    demo = src_data[name]

                    keys = set(demo["obs"].keys()) if "obs" in demo else set()
                    if reference_keys is None:
                        reference_keys = keys
                    elif keys != reference_keys:
                        raise ValueError(
                            f"{path}:{name} has observation keys {sorted(keys)}, but earlier demos have "
                            f"{sorted(reference_keys)}. Merging these would produce a dataset robomimic "
                            "cannot train on. Re-collect the rounds with the same observation group."
                        )

                    new_name = f"demo_{demo_index}"
                    src_data.copy(demo, data_group, name=new_name)
                    masks[FALLBACK_MASK if demo_used_fallback(demo) else CLEAN_MASK].append(new_name)
                    total_samples += int(demo.attrs.get("num_samples", 0))
                    demo_index += 1

        data_group.attrs["total"] = total_samples
        data_group.attrs["env_args"] = json.dumps({"env_name": env_name or "", "type": 2})
        mask_group = dst.create_group("mask")
        for key, names in masks.items():
            mask_group.create_dataset(key, data=[name.encode() for name in names], dtype=h5py.string_dtype())

    return dataset_stats(output)


#: Filename holding per-dimension action normalization stats for a training run.
ACTION_NORM_FILENAME = "action_norm.json"


def action_stats(path: str) -> tuple[list[float], list[float]]:
    """Return per-dimension action minima and maxima across every demo.

    Args:
        path: Robomimic-layout dataset.

    Returns:
        ``(minimum, maximum)``, each with one entry per action dimension.

    Raises:
        ValueError: If the dataset holds no actions.
    """
    import numpy as np

    minimum = None
    maximum = None
    with h5py.File(path, "r") as handle:
        for name in handle["data"]:
            if not name.startswith("demo_") or "actions" not in handle["data"][name]:
                continue
            actions = np.asarray(handle["data"][name]["actions"], dtype=np.float64)
            demo_min, demo_max = actions.min(axis=0), actions.max(axis=0)
            minimum = demo_min if minimum is None else np.minimum(minimum, demo_min)
            maximum = demo_max if maximum is None else np.maximum(maximum, demo_max)

    if minimum is None:
        raise ValueError(f"{path} contains no actions to compute normalization stats from.")
    return minimum.tolist(), maximum.tolist()


def write_normalized_actions(src: str, dst: str, stats: tuple[list[float], list[float]] | None = None) -> str:
    """Copy a dataset with its actions rescaled per dimension into ``[-1, 1]``.

    robomimic's ``ActorNetwork`` ends in a ``tanh``, so it can only ever emit
    actions inside ``[-1, 1]``. Joint-position tasks command raw radians well
    outside that range, which shows up as a training loss that falls a little
    and then sits on a plateau the network cannot cross. Rescaling makes the
    targets representable; :class:`RobomimicPolicy` inverts it at inference
    using the stats this writes alongside.

    A dimension that never varies maps to 0.0 rather than dividing by zero.

    Args:
        src: Dataset to read.
        dst: Path for the normalized copy. Overwritten if present.
        stats: Optional ``(minimum, maximum)`` to reuse, so later DAgger rounds
            stay on the scale the first round established. Computed from ``src``
            when omitted.

    Returns:
        ``dst``.
    """
    import numpy as np

    minimum, maximum = stats if stats is not None else action_stats(src)
    lo = np.asarray(minimum, dtype=np.float64)
    hi = np.asarray(maximum, dtype=np.float64)
    span = hi - lo
    degenerate = span < 1.0e-9

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with h5py.File(src, "r") as source, h5py.File(dst, "w") as target:
        for key, value in source.attrs.items():
            target.attrs[key] = value
        source.copy(source["data"], target, name="data")
        # Filter keys, so hdf5_filter_key still resolves on the copy robomimic reads.
        if "mask" in source:
            source.copy(source["mask"], target, name="mask")

        for name in target["data"]:
            if not name.startswith("demo_") or "actions" not in target["data"][name]:
                continue
            group = target["data"][name]
            actions = np.asarray(group["actions"], dtype=np.float64)
            scaled = np.where(degenerate, 0.0, 2.0 * (actions - lo) / np.where(degenerate, 1.0, span) - 1.0)
            del group["actions"]
            group.create_dataset("actions", data=scaled.astype(np.float32))

        target["data"].attrs["action_min"] = lo
        target["data"].attrs["action_max"] = hi

    return dst


def save_action_norm(directory: str, stats: tuple[list[float], list[float]]) -> str:
    """Write action normalization stats next to a training run's checkpoints."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, ACTION_NORM_FILENAME)
    with open(path, "w") as handle:
        json.dump({"action_min": list(stats[0]), "action_max": list(stats[1])}, handle, indent=2)
    return path


def load_action_norm(checkpoint: str) -> tuple[list[float], list[float]] | None:
    """Find the action normalization stats belonging to a checkpoint.

    Looks beside the checkpoint and then one directory up, which is where
    :func:`save_action_norm` puts them relative to robomimic's ``models/``
    subdirectory.

    Args:
        checkpoint: Path to a robomimic ``.pth`` checkpoint.

    Returns:
        ``(minimum, maximum)``, or None when the run trained on raw actions.
    """
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint))
    for candidate in (ckpt_dir, os.path.dirname(ckpt_dir)):
        path = os.path.join(candidate, ACTION_NORM_FILENAME)
        if os.path.exists(path):
            with open(path) as handle:
                payload = json.load(handle)
            return payload["action_min"], payload["action_max"]
    return None


def load_dataset_action_norm(data) -> tuple[list[float], list[float]] | None:
    """Read the stats :func:`write_normalized_actions` stamped on a dataset.

    Args:
        data: A checkpoint config's ``train.data``: a path, or robomimic's
            list of ``{"path": ...}`` entries.

    Returns:
        ``(minimum, maximum)``, or None when the dataset is missing or holds
        raw actions.
    """
    if isinstance(data, (list, tuple)):
        data = data[0] if data else None
    if isinstance(data, dict):
        data = data.get("path")
    if not isinstance(data, str) or not os.path.exists(data):
        return None
    with h5py.File(data, "r") as handle:
        attrs = handle["data"].attrs if "data" in handle else {}
        if "action_min" not in attrs or "action_max" not in attrs:
            return None
        return list(map(float, attrs["action_min"])), list(map(float, attrs["action_max"]))


def rotate_recorder_dataset(env, dataset_dir: str, dataset_name: str) -> str:
    """Point a live recorder at a new dataset file and drop in-flight episodes.

    DAgger writes one dataset per round, but the recorder opens its file in
    ``RecorderManager.__init__`` and a simulation can only host one
    :class:`InteractiveScene`, so rebuilding the environment per round is not an
    option. This closes the open file, opens the next one and clears the episode
    buffers and counters, which does mean reaching into the manager's internals.

    Any episode still in progress is dropped, so half-recorded trajectories from
    the previous round cannot leak into the new file. Callers that want those
    partial trajectories kept should call
    ``recorder_manager.export_episodes()`` before rotating - for behaviour
    cloning they are perfectly usable, since every sample is an independent
    (state, expert action) pair and truncation costs only the episode's tail.

    Args:
        env: The environment whose recorder should be rotated.
        dataset_dir: Directory for the new HDF5 file.
        dataset_name: Filename stem, without the ``.hdf5`` extension.

    Returns:
        Absolute path of the new dataset file.

    Raises:
        RuntimeError: If the environment has no active recorder terms.
    """
    from isaaclab.utils.datasets import EpisodeData

    base = env.unwrapped
    recorder = base.recorder_manager
    if len(recorder.active_terms) == 0:
        raise RuntimeError("Environment has no active recorder terms to rotate.")

    dataset_dir = os.path.abspath(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    path = os.path.join(dataset_dir, f"{dataset_name}.hdf5")

    recorder.cfg.dataset_export_dir_path = dataset_dir
    recorder.cfg.dataset_filename = dataset_name

    if recorder._dataset_file_handler is not None:
        recorder._dataset_file_handler.close()
        handler = recorder.cfg.dataset_file_handler_class_type()
        handler.create(os.path.join(dataset_dir, dataset_name), env_name=getattr(base.cfg, "env_name", None))
        recorder._dataset_file_handler = handler

    recorder._episodes = {env_id: EpisodeData() for env_id in range(base.num_envs)}
    recorder._exported_successful_episode_count = {}
    recorder._exported_failed_episode_count = {}
    return path


def _demo_sort_key(name: str) -> tuple[int, object]:
    """Sort ``demo_10`` after ``demo_9`` instead of lexicographically."""
    if name.startswith("demo_"):
        suffix = name[len("demo_") :]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, name)
