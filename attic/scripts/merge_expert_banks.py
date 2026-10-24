#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concatenate expert banks built by ``build_expert_bank.py`` for the same task (e.g. with different seeds).

A long bank can be built as several shorter runs, each in its own process::

    uv run python scripts/rl/merge_expert_banks.py logs/expert_bank/a.pt logs/expert_bank/b.pt \\
        --output logs/expert_bank/merged.pt
"""

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("banks", nargs="+", help="Bank files to concatenate, in order.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    banks = [torch.load(path, map_location="cpu", weights_only=False) for path in args.banks]
    first = banks[0]["meta"]
    for path, bank in zip(args.banks[1:], banks[1:]):
        for key in ("task", "rl_task", "dt", "arm_joint_names", "gripper_joint_names", "phases"):
            if bank["meta"][key] != first[key]:
                raise ValueError(f"{path}: meta '{key}' {bank['meta'][key]} differs from {first[key]}")
        if set(bank) != set(banks[0]):
            raise ValueError(f"{path} has keys {sorted(bank)}, not {sorted(banks[0])}")
    merged = {}
    for key in banks[0]:
        if key == "meta":
            continue
        values = [bank[key] for bank in banks]
        # Banks recorded with different episode lengths are cut to the shortest.
        if values[0].dim() >= 2 and key not in ("arm_q0", "valve_pose_env", "valve_dynamics", "expert_diversity", "grasp_in_frame"):
            horizon = min(v.shape[1] for v in values)
            values = [v[:, :horizon] for v in values]
        merged[key] = torch.cat(values)
    meta = dict(first)
    meta["horizon"] = min(bank["meta"]["horizon"] for bank in banks)
    meta["sampled"] = {key: torch.cat([b["meta"]["sampled"][key] for b in banks]) for key in first["sampled"]}
    stats = {}
    for bank in banks:
        for key, value in bank["meta"]["stats"].items():
            stats[key] = stats.get(key, 0) + value
    meta["stats"] = stats
    meta["success_rate"] = stats["successes"] / max(stats["episodes"], 1)
    meta["merged_from"] = list(args.banks)
    merged["meta"] = meta
    torch.save(merged, args.output)
    print(
        f"[MERGE] {len(merged['arm_q'])} trajectories from {len(banks)} banks to {args.output}; "
        f"expert success {stats['successes']}/{stats['episodes']} = {meta['success_rate']:.1%}"
    )


if __name__ == "__main__":
    main()
