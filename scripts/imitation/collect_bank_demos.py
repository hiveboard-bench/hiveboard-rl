#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record expert-bank demonstrations in an RL valve task, for behaviour cloning.

Plays bank trajectories in the RL task itself (its randomization, actuator delay and observation
noise, as PPO trains on), each from its start, with the bank expert (``expert_bank.expert_action``)
acting in the arm command ``--command`` (``mdp.ARM_COMMANDS``):

* ``increment``: the PPO student's action space (BC-A, and the BC-A + PPO initialization);
* ``absolute_integrated``: absolute targets through the same integrator (BC-C);
* ``absolute``: the scripted expert's joint-position control (BC-B);
* ``relative_integrated`` / ``relative``: the same targets, relative to the measured joints (BC-C / BC-B
  without the absolute targets' copy-the-joints shortcut).

Each step stores the deployable ``policy`` observation (with its history), the privileged
``teacher`` one (for critic pretraining), the expert action and the task reward. The expert-drift
terminations are off, so episodes run to the timeout as in evaluation. Success is the evaluation's
(opened while held), so the printed success rate is the ceiling of a policy cloned from this
data; failed episodes are dropped unless ``--keep_failed``::

    uv run python scripts/imitation/collect_bank_demos.py --command increment
    uv run python scripts/imitation/collect_bank_demos.py --task Isaac-HiveBoard-Anymal-SmallValve-RL-v0 \\
        --command absolute --bank logs/expert_bank/<nominal bank>.pt
"""

import argparse
import datetime
import importlib
import os
import sys

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401
import torch
from _common import parse_with_presets
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.mdp import ARM_COMMANDS

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-HiveBoard-Anymal-BallValve-RL-v0", help="RL task (training variant).")
parser.add_argument("--command", choices=ARM_COMMANDS, default="increment", help="Arm command the expert acts in.")
parser.add_argument("--bank", default=None, help="Bank .pt file (default: the task's EXPERT_BANK_PATH).")
parser.add_argument("--episodes", type=int, default=None, help="Episodes to play (default: one per bank trajectory).")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=0, help="Environment seed and bank order.")
parser.add_argument("--keep_failed", action="store_true", help="Also store episodes that did not open the valve.")
parser.add_argument("--output", default=None, help="Default: logs/imitation/bank_demos/<bank prefix>_<command>.pt.")
add_launcher_args(parser)
args = parse_with_presets(parser)


def main() -> None:
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import expert_bank, mdp

    task_env = importlib.import_module(gym.spec(args.task).kwargs["env_cfg_entry_point"].split(":")[0])
    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.actions.arm_action.command = args.command
    # Whole episodes, as evaluation runs them (the drift terminations are a PPO training device).
    env_cfg.terminations.expert_drift = None
    env_cfg.terminations.expert_valve_lag = None
    reset_cfg = env_cfg.events.reset_from_bank
    if args.bank:
        reset_cfg.params["path"] = os.path.abspath(args.bank)
    reset_cfg.func = expert_bank.reset_from_expert_bank_in_order
    reset_cfg.params["seed"] = args.seed
    output = args.output or os.path.join(
        "logs", "imitation", "bank_demos", f"{task_env.BANK_PREFIX}_{args.command}.pt"
    )
    if os.path.exists(output):
        raise SystemExit(f"Refusing to overwrite {output}; pass --output.")

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        n_envs, dev = env.num_envs, env.device
        dims = dict(zip(env.action_manager.active_terms, env.action_manager.action_term_dim))
        if env.action_manager.total_action_dim != dims["arm_action"] + dims["gripper_action"]:
            raise RuntimeError(f"Expected [arm, gripper] actions only, got {dims}.")
        term = env.expert_bank_term
        episodes = args.episodes or term.bank.size
        hold = task_env.HOLD

        obs, _ = env.reset()
        horizon = int(env.max_episode_length) + 1
        buffers = {
            "policy": torch.zeros(n_envs, horizon, obs["policy"].shape[-1], device=dev),
            "teacher": torch.zeros(n_envs, horizon, obs["teacher"].shape[-1], device=dev),
            "actions": torch.zeros(n_envs, horizon, env.action_manager.total_action_dim, device=dev),
            "rewards": torch.zeros(n_envs, horizon, device=dev),
        }
        length = torch.zeros(n_envs, dtype=torch.long, device=dev)
        grasped = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        success = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        bank_index = term.index.clone()
        rows = torch.arange(n_envs, device=dev)
        stored: dict[str, list[torch.Tensor]] = {key: [] for key in buffers}
        record = {"lengths": [], "success": [], "bank_index": [], "timeout": []}
        finished = clipped = steps = 0

        with torch.inference_mode():
            while finished < episodes:
                action = expert_bank.expert_action(env)
                buffers["policy"][rows, length] = obs["policy"]
                buffers["teacher"][rows, length] = obs["teacher"]
                buffers["actions"][rows, length] = action
                if args.command == "increment":
                    clipped += int((action[:, :-1].abs() >= 1.0).sum())
                    steps += action.shape[0]
                # The evaluation's success: opened while held, read from the state the action applies to.
                held = mdp.lever_held(env, hold["dist_threshold"], hold["ang_threshold"])
                grasped |= held
                opened = mdp.valve_open_success(env, task_env.SUCCESS_TOLERANCE_RAD)
                success |= opened & mdp.opening_credited(env, held, grasped)

                obs, reward, terminated, truncated, _ = env.step(action)
                buffers["rewards"][rows, length] = reward
                length += 1

                for e in (terminated | truncated).nonzero().flatten().tolist():
                    if finished < episodes:
                        finished += 1
                        if success[e] or args.keep_failed:
                            n = int(length[e])
                            for key, buffer in buffers.items():
                                stored[key].append(buffer[e, :n].cpu())
                            record["lengths"].append(n)
                            record["success"].append(bool(success[e]))
                            record["bank_index"].append(int(bank_index[e]))
                            record["timeout"].append(bool(truncated[e]) and not bool(terminated[e]))
                    length[e] = 0
                    grasped[e] = False
                    success[e] = False
                    bank_index[e] = term.index[e]
                if finished and (terminated | truncated).any():
                    kept = sum(record["success"])
                    print(f"[DEMOS] {finished}/{episodes} episodes, {kept} successful ({kept / finished:.1%})", flush=True)
        dt = env.step_dt
        bank_path = reset_cfg.params["path"]
        env.close()

    successes = sum(record["success"])
    if not record["lengths"]:
        raise SystemExit(f"The bank expert opened none of {finished} episodes; nothing to store.")
    data = {key: torch.cat(values) for key, values in stored.items()}
    data.update({key: torch.tensor(values) for key, values in record.items()})
    data["meta"] = {
        "task": args.task,
        "command": args.command,
        "bank": os.path.abspath(bank_path),
        "seed": args.seed,
        "dt": dt,
        "episodes_played": finished,
        "replay_success_rate": successes / max(finished, 1),
        "keep_failed": args.keep_failed,
        "unique_trajectories": len(set(record["bank_index"])),
        "hydra_overrides": sys.argv[1:],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(data, output)
    print(
        f"\n[DEMOS] {args.task} ({args.command}): the bank expert opened {successes}/{finished}"
        f" = {successes / max(finished, 1):.1%} of episodes; this is the ceiling for BC on this data."
    )
    if steps:
        print(f"[DEMOS] arm increments at the +-1 clip: {clipped / (steps * 6):.2%} of joint-steps")
    print(
        f"[DEMOS] {len(record['lengths'])} episodes, {data['policy'].shape[0]} steps,"
        f" policy obs {data['policy'].shape[-1]}-D, teacher obs {data['teacher'].shape[-1]}-D -> {output}"
    )


if __name__ == "__main__":
    main()
