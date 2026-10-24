#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Find idle stretches (the expert waiting) in an expert-bank's trajectories.

A step is *idle* when the action is almost zero: in the RL action space
(integrated joint increments, ``scale`` rad per unit action) every arm-joint
target moves less than ``--eps`` rad and the binary gripper command does not
change. Runs of at least ``--min_steps`` idle steps between reset and the
bank's open time are reported, per command phase, together with whether the
arm (``arm_q``) and gripper (``gripper_q``) were still moving meanwhile (e.g.
the expert pausing while its gripper closes). No simulation is needed::

    uv run python scripts/rl/check_bank_idle.py
    uv run python scripts/rl/check_bank_idle.py --bank logs/expert_bank/other.pt --eps 5e-4 --min_steps 3

Exits with status 1 if any trajectory has an idle run, so it can gate a build.
"""

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_BANK = os.path.join(REPO_ROOT, "logs", "expert_bank", "anymal_ball_valve_bank_5000.pt")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--bank", default=DEFAULT_BANK, help="Bank .pt file.")
parser.add_argument("--eps", type=float, default=1.0e-3, help="Largest arm-target change [rad/step] counted as idle.")
parser.add_argument("--min_steps", type=int, default=2, help="Shortest idle run reported [steps].")
parser.add_argument("--whole_episode", action="store_true", help="Also check after the open time.")
parser.add_argument("--show", type=int, default=10, help="Number of worst trajectories to list.")
args = parser.parse_args()


def idle_runs(idle: torch.Tensor) -> list[tuple[int, int]]:
    """``(start, length)`` of each run of True in a 1-D bool tensor."""
    padded = torch.cat((torch.tensor([False]), idle, torch.tensor([False]))).int()
    edges = padded[1:] - padded[:-1]
    starts = (edges == 1).nonzero().flatten()
    ends = (edges == -1).nonzero().flatten()
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def main() -> int:
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    meta = bank["meta"]
    dt = meta["dt"]
    target = bank["arm_target"].float()  # (N, T, 6)
    q = bank["arm_q"].float()
    grip_cmd = bank["gripper_cmd"].float()  # (N, T)
    grip_q = bank["gripper_q"].float()
    phase = bank["phase"]
    n, horizon = grip_cmd.shape
    # Segment index -> "index:phase" when the bank recorded named phases (build_expert_bank.py).
    names = list(meta.get("phases", [])) + ["done"]
    segment_name: dict[int, str] = {}
    if "phase_id" in bank:
        for seg, pid in torch.unique(torch.stack((phase, bank["phase_id"]), dim=-1).reshape(-1, 2), dim=0).tolist():
            segment_name[seg] = f"{seg}:{names[pid] if pid >= 0 else 'untagged'}"

    # Step t is the action taken at t: target[t] - target[t-1] (the first step
    # is measured from the reset configuration).
    prev = torch.cat((bank["arm_q0"].float()[:, None], target[:, :-1]), dim=1)
    arm_step = (target - prev).abs().amax(dim=-1)  # (N, T)
    grip_change = torch.cat((torch.zeros(n, 1, dtype=torch.bool), grip_cmd[:, 1:] != grip_cmd[:, :-1]), dim=1)
    idle = (arm_step < args.eps) & ~grip_change

    q_speed = torch.cat((torch.zeros(n, 1), (q[:, 1:] - q[:, :-1]).abs().amax(dim=-1) / dt), dim=1)
    grip_speed = torch.cat((torch.zeros(n, 1), (grip_q[:, 1:] - grip_q[:, :-1]).abs() / dt), dim=1)

    open_step = (bank["t_open"] / dt).round().long().clamp(max=horizon)
    end = torch.full_like(open_step, horizon) if args.whole_episode else open_step

    per_traj_idle = torch.zeros(n, dtype=torch.long)
    per_phase: dict[int, dict[str, float]] = {}
    lengths: list[int] = []
    worst: list[tuple[int, int, int, int]] = []  # (idle steps, traj, start, length) of the longest run
    for i in range(n):
        longest = (0, 0)
        for start, length in idle_runs(idle[i, : int(end[i])]):
            if length < args.min_steps:
                continue
            per_traj_idle[i] += length
            lengths.append(length)
            longest = max(longest, (length, start))
            seg = slice(start, start + length)
            for p in phase[i, seg].unique().tolist():
                mask = phase[i, seg] == p
                s = per_phase.setdefault(p, {"steps": 0, "runs": 0, "arm_moving": 0, "grip_moving": 0})
                s["steps"] += int(mask.sum())
                s["runs"] += 1
                s["arm_moving"] += int((q_speed[i, seg][mask] > 0.05).sum())
                s["grip_moving"] += int((grip_speed[i, seg][mask] > 0.05).sum())
        if longest[0]:
            worst.append((int(per_traj_idle[i]), i, longest[1], longest[0]))

    window = "whole episode" if args.whole_episode else "reset -> open"
    total_steps = int(end.sum())
    flagged = int((per_traj_idle > 0).sum())
    print(f"bank: {args.bank}")
    print(f"  {n} trajectories, horizon {horizon} steps, dt {dt}s; window {window} ({total_steps} steps)")
    print(f"  idle = max|d arm_target| < {args.eps:g} rad/step and no gripper toggle; runs >= {args.min_steps} steps")
    print(f"  arm-target step over the window: median {arm_step.median():.2e}, max {arm_step.max():.3f} rad")
    print()
    if not flagged:
        print("OK: no idle stretches found.")
        return 0

    idle_total = int(per_traj_idle.sum())
    lengths_t = torch.tensor(lengths, dtype=torch.float)
    print(f"IDLE: {flagged}/{n} trajectories ({100 * flagged / n:.1f}%) have idle runs")
    print(f"  {idle_total} idle steps = {idle_total * dt:.1f}s, {100 * idle_total / total_steps:.1f}% of the window")
    print(
        f"  {len(lengths)} runs; length median {lengths_t.median() * dt:.2f}s, "
        f"p90 {lengths_t.quantile(0.9) * dt:.2f}s, max {lengths_t.max() * dt:.2f}s"
    )
    print(f"  idle time per flagged trajectory: mean {per_traj_idle[per_traj_idle > 0].float().mean() * dt:.2f}s")
    print()
    print("per segment (index:phase)   idle steps   runs   arm still moving   gripper moving")
    for p in sorted(per_phase):
        s = per_phase[p]
        print(
            f"  {segment_name.get(p, str(p)):<26s} {s['steps']:>10d} {s['runs']:>6d} "
            f"{100 * s['arm_moving'] / s['steps']:>17.0f}% {100 * s['grip_moving'] / s['steps']:>15.0f}%"
        )
    print()
    print(f"worst {min(args.show, len(worst))} trajectories (index: total idle, longest run)")
    for total, i, start, length in sorted(worst, reverse=True)[: args.show]:
        print(
            f"  {i:>5d}: {total * dt:.2f}s idle; longest {length * dt:.2f}s at t={start * dt:.2f}s "
            f"(segment {segment_name.get(int(phase[i, start]), int(phase[i, start]))}, open at {bank['t_open'][i]:.2f}s)"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
