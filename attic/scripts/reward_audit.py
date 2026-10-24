#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Score scripted behaviours with the RL task's own rewards and terminations.

Runs the training task with the expert bank's recorded actions (as
``replay_expert_bank.py`` does) and, side by side, counterfactuals built from
them, then prints every reward term's return per phase for each behaviour and
what ended the episodes. It answers "does the reward prefer X to what the
expert does?" without training::

    uv run python scripts/rl/reward_audit.py --num_envs 128

Behaviours (split evenly over the environments, same trajectories each):

* ``expert``: the bank's joint targets and gripper commands, on the
  references' timeline: the idle-free reach (``ExpertBank.reach``), then the
  bank's own steps shifted by the idle steps it dropped.
* ``hold``: the expert until ``--hold_after_s`` past its grasp, then the arm
  holds its target with the gripper closed (grasp, never turn).
* ``expert_keep``: the expert until the bank's open time, then holding the
  lever there (the scripted expert lets go after opening; a policy need not).
* ``flicker``: ``expert_keep``'s arm, but from the grasp on the gripper is
  commanded closed for six steps and open for one (teacher v21's pattern).
"""

import argparse
import sys

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-HiveBoard-Anymal-BallValve-RL-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--hold_after_s", type=float, default=0.5, help="Hold behaviour: seconds after the grasp.")
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]

BEHAVIOURS = ("expert", "hold", "expert_keep", "flicker")


def make_paired_reset():
    """``reset_from_expert_bank`` giving each behaviour's block of envs the same trajectories."""
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.expert_bank import reset_from_expert_bank

    class reset_paired(reset_from_expert_bank):
        def __call__(self, env, env_ids, path, mid_start_prob=0.0, expert_task=None):
            ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
            gen = torch.Generator().manual_seed(args.seed)
            order = torch.randperm(self.bank.size, generator=gen).to(env.device)
            self.write_start_state(env, ids, order[ids % (env.num_envs // len(BEHAVIOURS))])

    return reset_paired


def main() -> None:
    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.events.reset_from_bank.func = make_paired_reset()
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        dev, n = env.device, env.num_envs
        dt = env.step_dt
        term = env.expert_bank_term
        raw = torch.load(env_cfg.events.reset_from_bank.params["path"], map_location="cpu", weights_only=False)
        arm_target = raw["arm_target"].float().to(dev)
        gripper_cmd = raw["gripper_cmd"].float().to(dev)
        bank = term.bank
        arm = env.action_manager.get_term("arm_action")
        scale = float(env.cfg.actions.arm_action.scale)
        rewards = env.reward_manager
        names = rewards.active_terms
        terminations = env.termination_manager

        # A pushing expert closes its hand during the reach; a grasping one keeps it open until the grasp.
        push = bool(env.cfg.valve_task.push)
        env.reset()
        behaviour = torch.arange(n, device=dev) // (n // len(BEHAVIOURS))
        behaviour.clamp_(max=len(BEHAVIOURS) - 1)

        # Returns per term, per phase (0 before the expert grasps, 1 after), per env.
        returns = torch.zeros(n, 2, len(names), device=dev)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        ended_by = [""] * n
        steps_alive = torch.zeros(n, device=dev)
        with torch.inference_mode():
            for _ in range(int(env.max_episode_length)):
                i = term.index
                t = env.episode_length_buf + term.start_step
                b = bank._bank_step(i, t)
                reaching = t < bank.grasp_step[i]
                target = torch.where(reaching[:, None], bank.reach_reference(i, t), arm_target[i, b])
                reach_grip = gripper_cmd[i, b] if push else torch.ones_like(gripper_cmd[i, b])
                grip = torch.where(reaching, reach_grip, gripper_cmd[i, b])
                grasp_step = bank.grasp_step[i]
                holding = ((behaviour == 1) & (t >= grasp_step + int(round(args.hold_after_s / dt)))) | (
                    (behaviour >= 2) & (b >= bank.open_step[i])
                )
                target = torch.where(holding[:, None], arm._target, target)
                grip = torch.where(holding, torch.full_like(grip, -1.0), grip)
                flicker = (behaviour == 3) & (t >= grasp_step) & ((t - grasp_step) % 7 == 6)
                grip = torch.where(flicker, torch.ones_like(grip), grip)
                delta = ((target - arm._target) / scale).clamp(-1.0, 1.0)
                phase = (t >= grasp_step).long()
                env.step(torch.cat((delta, grip[:, None]), dim=-1))
                step = rewards._step_reward * dt
                live = alive.clone()
                returns[live, phase[live]] += step[live]
                steps_alive += live.float()
                done = terminations.dones & alive
                for e in done.nonzero().flatten().tolist():
                    ended_by[e] = ",".join(k for k in terminations.active_terms if bool(terminations.get_term(k)[e]))
                alive &= ~done

        for b, name in enumerate(BEHAVIOURS):
            mask = behaviour == b
            total = returns[mask].sum(dim=1).mean(dim=0)
            print(f"\n=== {name}: return {total.sum():+.2f} per episode, alive {steps_alive[mask].mean() * dt:.1f}s")
            ends = [ended_by[e] or "(still running)" for e in mask.nonzero().flatten().tolist()]
            print("    ended by: " + ", ".join(f"{k} x{ends.count(k)}" for k in sorted(set(ends))))
            print(f"    {'term':28s} {'before grasp':>13s} {'after grasp':>12s} {'total':>9s}")
            per_phase = returns[mask].mean(dim=0)
            for k, term_name in enumerate(names):
                a, c = float(per_phase[0, k]), float(per_phase[1, k])
                if abs(a) + abs(c) > 1e-3:
                    print(f"    {term_name:28s} {a:+13.3f} {c:+12.3f} {a + c:+9.3f}")
        env.close()


if __name__ == "__main__":
    main()
