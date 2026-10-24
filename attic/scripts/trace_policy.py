#!/usr/bin/env python3
"""Record one standard-condition episode, joint motion and weighted rewards."""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401
import numpy as np
import torch
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import expert_bank, mdp
from play import configure_standard

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--unwind-on-release",
    action="store_true",
    help="Diagnostic intervention: reset arm targets to measured joints when the policy opens after gripping.",
)
add_launcher_args(parser)
args, overrides = setup_preset_cli(parser)
if not any(t.startswith(("physics=", "presets=")) for t in overrides):
    overrides.append("physics=newton_mjwarp")
args.visualizer = []
sys.argv = [sys.argv[0], *overrides]


def main():
    task = "Isaac-HiveBoard-Anymal-BallValve-RL-Play-v0"
    cfg, agent = resolve_task_config(task, "rsl_rl_student_ppo_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    configure_standard(cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
        runner = create_rsl_rl_runner(env, agent)
        runner.load(str(Path(args.checkpoint).resolve()))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        u = env.unwrapped
        obs = env.get_observations()
        term = u.expert_bank_term
        arm = u.action_manager.get_term("arm_action")
        robot = u.scene["robot"]
        dt = u.step_dt
        rows = []
        reward_names = list(u.reward_manager.active_terms)
        index = int(term.index[0])
        previous_grip = 1.0
        interventions = []

        def cpu(t):
            return t.detach().cpu().numpy().copy()

        with torch.inference_mode():
            for k in range(int(u.max_episode_length)):
                action = policy(obs)
                grip = float(action[0, -1])
                if args.unwind_on_release and grip > 0 and previous_grip <= 0:
                    arm._target.copy_(robot.data.joint_pos.torch[:, term._arm_ids])
                    arm._filtered.zero_()
                    interventions.append(k * dt)
                previous_grip = grip
                row = dict(
                    time=k * dt,
                    q=cpu(robot.data.joint_pos.torch[0, term._arm_ids]),
                    velocity=cpu(robot.data.joint_vel.torch[0, term._arm_ids]),
                    target=cpu(arm._target[0]),
                    action=cpu(action[0]),
                    reference=cpu(expert_bank.expert_joint_reference(u)[0]),
                    velocity_reference=cpu(term.bank.velocity_reference(term.index, expert_bank.reference_step(u))[0]),
                    finger=float(robot.data.joint_pos.torch[0, term._finger_id]),
                    finger_reference=float(term.bank.gripper_reference(term.index, expert_bank.reference_step(u))[0]),
                    valve=float(mdp.valve_angle(u)[0]),
                    pad_force=cpu(mdp.pad_valve_force(u)[0]),
                    bank_step=int(term.bank._bank_step(term.index, expert_bank.reference_step(u))[0]),
                )
                obs, reward, done, extras = env.step(action)
                row["reward"] = cpu(u.reward_manager._step_reward[0]) * dt
                rows.append(row)
                if bool(done[0]):
                    break
        arrays = {name: np.asarray([r[name] for r in rows]) for name in rows[0]}
        np.savez_compressed(output / "trajectory.npz", **arrays)
        metadata = dict(
            checkpoint=args.checkpoint,
            seed=args.seed,
            dt=dt,
            max_position_error=cfg.actions.arm_action.max_position_error,
            bank_index=index,
            bank_path=cfg.events.reset_from_bank.params["path"],
            reward_names=reward_names,
            samples=len(rows),
            unwind_on_release=args.unwind_on_release,
            interventions=interventions,
            bank_open_step=int(term.bank.open_step[index]),
            ended_by=[
                name for name in u.termination_manager.active_terms if bool(u.termination_manager.get_term(name)[0])
            ],
        )
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        env.close()
    print(f"Saved {len(rows)} samples to {output}", flush=True)


if __name__ == "__main__":
    main()
