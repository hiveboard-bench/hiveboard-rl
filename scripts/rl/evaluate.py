#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL teacher or student checkpoint on an ANYmal RL valve task (``--task``).

Runs ``--episodes`` full episodes, spread over ``--num_envs`` parallel
environments, and reports the metrics the benchmark papers use (NIST task
board protocol, FurnitureBench-style stage progress):

* success rate with a 95% Wilson interval, plus the one-sided 95%
  Clopper-Pearson lower bound on the success probability (NIST "reliability");
* time to success (mean, std, 95% CI);
* stage rates: reached the grasp, grasped the lever, opened the valve;

Success means the valve reached open *while the lever was held* at the
expert grasp pose, matching the reward's definition. Opening it any other way
(pushing, knocking) is reported separately as ``opened_any``;
* peak pad force (and its squeeze and lateral parts) and solver blow-ups.

Results are printed and written as JSON next to the checkpoint::

    uv run python scripts/rl/evaluate.py --checkpoint logs/rsl_rl/anymal_ball_valve_teacher/<run>/model_2999.pt
    uv run python scripts/rl/evaluate.py --agent rsl_rl_distillation_cfg_entry_point \\
        --checkpoint logs/rsl_rl/anymal_ball_valve_student/<run>/model_1499.pt

Hydra overrides go after the flags, e.g. a registration-error sweep:
``env.observations.policy.registered_valve.params.bias_pos=0.02``.
"""

import argparse
import importlib
import json
import math
import os
import sys

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401
import torch
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import expert_bank

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_JOINT_NAMES

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-HiveBoard-Anymal-BallValve-RL-Play-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="Agent entry point of the checkpoint.")
parser.add_argument("--checkpoint", required=True, help="RSL-RL model_*.pt checkpoint.")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default=None, help="JSON output path (default: next to the checkpoint).")
parser.add_argument(
    "--trace", action="store_true", help="Print environment 0 step by step for its first episode, and what ended it."
)
parser.add_argument(
    "--video",
    default=None,
    help="Record environment 0 to this directory (Newton viewer), close up on the mechanism; each of its episodes"
    " is listed with its start time in the video. Use --num_envs 1 to film every episode.",
)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
if not any(t.startswith(("physics=", "presets=")) for t in hydra_args):
    hydra_args.append("physics=newton_mjwarp")
if args.video:
    args.visualizer = ["newton_gl"]
elif args.visualizer is None and not getattr(args, "visualizer_explicit", False):
    args.visualizer = []
sys.argv = [sys.argv[0], *hydra_args]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def reliability_lower_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound on the success probability (NIST task-board reliability)."""
    if successes == 0:
        return 0.0
    alpha = 1.0 - confidence

    def tail(p: float) -> float:  # P(X >= successes | p)
        return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(successes, n + 1))

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tail(mid) < alpha else (lo, mid)
    return lo


def _quantiles(values: list[float], qs: tuple[float, ...] = (0.5, 0.99)) -> list[float]:
    return torch.quantile(torch.tensor(values), torch.tensor(qs)).tolist()


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


REVERSAL_RATE = 0.05
"""Joint speed [rad/s] both sides of a velocity sign change must exceed to count as a reversal."""


def bank_joint_stats(path: str, dt: float) -> dict[str, torch.Tensor]:
    """Per-joint RMS acceleration and reversal rate of the bank's own expert trajectories.

    Uses the same finite differences and window (reset until open) as the policy metrics.
    """
    bank = torch.load(path, map_location="cpu", weights_only=False)
    q = bank["arm_q"].float()
    open_step = (bank["t_open"] / dt).round().long()
    v = (q[:, 1:] - q[:, :-1]) / dt
    a = (v[:, 1:] - v[:, :-1]) / dt
    steps = torch.arange(a.shape[1])[None]
    mask = (steps + 2 <= open_step[:, None]).float()[..., None]
    flip = (
        (torch.sign(v[:, 1:]) != torch.sign(v[:, :-1]))
        & (v[:, 1:].abs() > REVERSAL_RATE)
        & (v[:, :-1].abs() > REVERSAL_RATE)
    )
    count = mask.sum(dim=(0, 1))
    return {
        "rms_accel": (a.square() * mask).sum(dim=(0, 1)).div(count).sqrt(),
        "reversals_per_s": (flip.float() * mask).sum(dim=(0, 1)).div(count * dt),
    }


def joint_table(episodes: list[dict], dt: float, expert: dict[str, torch.Tensor] | None) -> dict[str, dict]:
    """Aggregate per-joint metrics over episodes, next to the expert baseline."""
    steps = sum(e["joint_steps"] for e in episodes)
    table = {}
    for j, name in enumerate(ANYMAL_ARM_JOINT_NAMES):
        err = sum(e["joint_err_sq"][j] for e in episodes)
        acc = sum(e["joint_acc_sq"][j] for e in episodes)
        rev = sum(e["joint_reversals"][j] for e in episodes)
        table[name] = {
            "rms_error_rad": math.sqrt(err / max(steps, 1)),
            "rms_accel": math.sqrt(acc / max(steps, 1)),
            "reversals_per_s": rev / max(steps * dt, 1e-9),
            "expert": {
                "rms_accel": float(expert["rms_accel"][j]) if expert else float("nan"),
                "reversals_per_s": float(expert["reversals_per_s"][j]) if expert else float("nan"),
            },
        }
    return table


def main() -> None:
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent)
    env_cfg.scene.num_envs = min(args.num_envs, args.episodes)
    env_cfg.seed = args.seed
    if args.video:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

        # Side view of the mechanism (which sits ~1 m ahead of the robot) and the hand, in env-0 coordinates.
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.8, 0.5, 0.8)
        env_cfg.viewer.lookat = (0.95, 0.0, 0.65)
        steps = int(math.ceil(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)))
        episodes0 = math.ceil(args.episodes / env_cfg.scene.num_envs) + 1
        env_cfg.video_recorders = [
            VideoRecorderCfg(output_dir=os.path.abspath(args.video), video_length=steps * episodes0, video_interval=0)
        ]

    with launch_simulation(env_cfg, args):
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
        base_env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(os.path.abspath(args.checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp

        # The task's env module: its hold gate and success tolerance.
        task_env = importlib.import_module(gym.spec(args.task).kwargs["env_cfg_entry_point"].split(":")[0])

        uenv = env.unwrapped
        n_envs, dev, dt = uenv.num_envs, uenv.device, uenv.step_dt
        grasp = task_env.HOLD
        turn = uenv.command_manager.get_term("valve_turn")
        robot = uenv.scene["robot"]
        gripper_body = robot.find_bodies("robotiq_base_link")[0][0]
        arm_ids = robot.find_joints(list(ANYMAL_ARM_JOINT_NAMES), preserve_order=True)[0]
        finger_ids = robot.find_joints(uenv.cfg.actions.gripper_action.joint_names)[0]
        gripper_action = uenv.action_manager.get_term("gripper_action")
        bank_term = getattr(uenv, "expert_bank_term", None)
        wrist_col = list(ANYMAL_ARM_JOINT_NAMES).index("dynaarm_wrist_flexion")
        zeros6 = lambda: torch.zeros(n_envs, 6, device=dev)  # noqa: E731

        def fresh() -> dict[str, torch.Tensor]:
            return {
                "reached": torch.zeros(n_envs, dtype=torch.bool, device=dev),
                "grasped": torch.zeros(n_envs, dtype=torch.bool, device=dev),
                "success": torch.zeros(n_envs, dtype=torch.bool, device=dev),
                "opened_any": torch.zeros(n_envs, dtype=torch.bool, device=dev),
                # Valve open with the arm let go and backed off, at the latest step.
                "clear": torch.zeros(n_envs, dtype=torch.bool, device=dev),
                # The latest step's progress, TCP distance from the grasp target [m] and pad force [N], for why
                # an episode did not end released and clear.
                "end_progress": torch.zeros(n_envs, device=dev),
                "end_tcp_dist": torch.zeros(n_envs, device=dev),
                "end_pad_force": torch.zeros(n_envs, device=dev),
                "t_success": torch.full((n_envs,), float("nan"), device=dev),
                "max_progress": torch.zeros(n_envs, device=dev),
                "max_pad_force": torch.zeros(n_envs, device=dev),
                "max_pad_squeeze": torch.zeros(n_envs, device=dev),
                "max_pad_lateral": torch.zeros(n_envs, device=dev),
                "max_valve_rate": torch.zeros(n_envs, device=dev),
                "max_tcp_speed": torch.zeros(n_envs, device=dev),
                "turn_rate_sum": torch.zeros(n_envs, device=dev),
                "turn_steps": torch.zeros(n_envs, device=dev),
                "commanded_rate": turn.rate.clone(),
                # Per-joint smoothness and expert tracking, until open-while-held.
                "joint_err_sq": zeros6(),
                "joint_acc_sq": zeros6(),
                "joint_reversals": zeros6(),
                "joint_steps": torch.zeros(n_envs, device=dev),
                # Motion after opening while commanded to release. The older
                # joint metrics stop at success and cannot see a release snap.
                "release_steps": torch.zeros(n_envs, device=dev),
                "release_peak_speed": torch.zeros(n_envs, device=dev),
                "release_acc_sq": torch.zeros(n_envs, device=dev),
                "release_velocity_err_sq": torch.zeros(n_envs, device=dev),
                "prev_measured_v": zeros6(),
                "prev_q": robot.data.joint_pos.torch[:, arm_ids].clone(),
                "prev_v": zeros6(),
                # Which bank trajectory the episode started from, and the wrist
                # flexion at the first held step (its sign is the IK branch).
                "bank_index": bank_term.index.clone() if bank_term is not None else torch.zeros(n_envs, device=dev),
                "grasp_wrist": torch.full((n_envs,), float("nan"), device=dev),
            }

        stats, step_count = fresh(), torch.zeros(n_envs, device=dev)
        env0_episodes: list[dict] = []
        env0_start = total_steps = 0
        episodes: list[dict] = []
        obs = env.get_observations()
        trace_done = False
        with torch.inference_mode():
            while len(episodes) < args.episodes:
                # Metrics are read from the state the action is applied to;
                # after a done step the environment has already been reset.
                dist, ang = mdp.tcp_grasp_error(uenv)
                progress = mdp.valve_progress(uenv)
                opened = mdp.valve_open_success(uenv, task_env.SUCCESS_TOLERANCE_RAD)
                stats["reached"] |= dist < grasp["dist_threshold"]
                held = mdp.lever_held(uenv, grasp["dist_threshold"], grasp["ang_threshold"])
                first_hold = held & stats["grasp_wrist"].isnan()
                stats["grasp_wrist"][first_hold] = robot.data.joint_pos.torch[first_hold, arm_ids[wrist_col]]
                stats["grasped"] |= held
                stats["opened_any"] |= opened
                stats["clear"] = mdp.released_and_clear(uenv, task_env.SUCCESS_TOLERANCE_RAD)
                stats["end_progress"] = progress.clone()
                stats["end_tcp_dist"] = dist.clone()
                stats["end_pad_force"] = mdp.pad_valve_force(uenv).max(dim=-1).values
                # Held now, or for a pushed lever touched before (it runs ahead of the hand).
                credited = opened & mdp.opening_credited(uenv, held, stats["grasped"])
                new = credited & ~stats["success"]
                stats["t_success"][new] = step_count[new] * dt
                stats["success"] |= credited
                stats["max_progress"] = torch.maximum(stats["max_progress"], progress)
                # Achieved turning rate: mean |dq/dt| while held and mid-turn.
                turning = held & (progress > 0.05) & (progress < 0.95)
                stats["turn_rate_sum"] += mdp.valve_state(uenv)[:, 1].abs() * turning
                stats["turn_steps"] += turning.float()
                stats["commanded_rate"] = turn.rate.clone()
                q = robot.data.joint_pos.torch[:, arm_ids]
                v = (q - stats["prev_q"]) / dt
                active = ((step_count >= 2) & ~stats["success"]).float()[:, None]
                stats["joint_acc_sq"] += ((v - stats["prev_v"]) / dt).square() * active
                flip = (
                    (torch.sign(v) != torch.sign(stats["prev_v"]))
                    & (v.abs() > REVERSAL_RATE)
                    & (stats["prev_v"].abs() > REVERSAL_RATE)
                )
                stats["joint_reversals"] += flip.float() * active
                if bank_term is not None:
                    stats["joint_err_sq"] += expert_bank.expert_joint_error(uenv).square() * active
                stats["joint_steps"] += active[:, 0]
                measured_v = robot.data.joint_vel.torch[:, arm_ids]
                releasing = stats["success"] & (gripper_action.raw_actions[:, 0] > 0) & (step_count >= 2)
                stats["release_steps"] += releasing.float()
                stats["release_peak_speed"] = torch.maximum(
                    stats["release_peak_speed"], measured_v.abs().amax(dim=-1) * releasing
                )
                stats["release_acc_sq"] += (
                    ((measured_v - stats["prev_measured_v"]) / dt).square().mean(dim=-1) * releasing
                )
                if bank_term is not None:
                    velocity_reference = bank_term.bank.velocity_reference(
                        bank_term.index, expert_bank.reference_step(uenv)
                    )
                    stats["release_velocity_err_sq"] += (
                        (measured_v - velocity_reference).square().mean(dim=-1) * releasing
                    )
                stats["prev_measured_v"] = measured_v.clone()
                stats["prev_q"] = q.clone()
                stats["prev_v"] = v.clone()
                speed = torch.norm(robot.data.body_lin_vel_w.torch[:, gripper_body], dim=-1)
                stats["max_tcp_speed"] = torch.maximum(stats["max_tcp_speed"], speed)
                if args.trace and not trace_done:
                    print(
                        f"[trace] t={step_count[0].item() * dt:5.2f}s dist={dist[0]:.3f}m ang={ang[0]:.2f}rad"
                        f" held={int(mdp.lever_held(uenv, grasp['dist_threshold'], grasp['ang_threshold'])[0])}"
                        f" progress={progress[0]:.3f} valve_rate={mdp.valve_state(uenv)[0, 1]:+.2f}rad/s"
                        f" gripper_speed={speed[0]:.2f}m/s pad_force={mdp.pad_valve_force(uenv)[0].max():.1f}N"
                        f" grip_cmd={'close' if gripper_action.raw_actions[0, 0] < 0 else 'open'}"
                        f" finger_q={robot.data.joint_pos.torch[0, finger_ids].abs().max():.2f}"
                        + (
                            f" |q-q_exp|={expert_bank.expert_joint_error(uenv)[0].norm():.3f}rad"
                            f" ep_step={int(uenv.episode_length_buf[0])} bank_idx={int(bank_term.index[0])}"
                            f" grip_ref={int(expert_bank.expert_gripper_reference(uenv)[0, 0])}"
                            if bank_term is not None
                            else ""
                        )
                    )
                stats["max_valve_rate"] = torch.maximum(stats["max_valve_rate"], mdp.valve_state(uenv)[:, 1].abs())
                squeeze, lateral = mdp.pad_force_split(uenv)
                stats["max_pad_squeeze"] = torch.maximum(stats["max_pad_squeeze"], squeeze.max(dim=-1).values)
                stats["max_pad_lateral"] = torch.maximum(stats["max_pad_lateral"], lateral.max(dim=-1).values)
                stats["max_pad_force"] = torch.maximum(
                    stats["max_pad_force"], mdp.pad_valve_force(uenv).max(dim=-1).values
                )

                obs, _, dones, extras = env.step(policy(obs))
                policy.reset(dones)
                step_count += 1
                total_steps += 1
                invalid = uenv.termination_manager.get_term("invalid")
                if args.trace and not trace_done and bool(dones[0]):
                    manager = uenv.termination_manager
                    fired = [name for name in manager.active_terms if bool(manager.get_term(name)[0])]
                    print(f"[trace] episode ended at step {int(step_count[0])}: {', '.join(fired) or 'unknown'}")
                    trace_done = True
                for i in dones.nonzero(as_tuple=False).flatten().tolist():
                    if len(episodes) >= args.episodes:
                        break
                    if i == 0:
                        # Environment 0's episodes, by their span in the video (one frame per env step).
                        env0_episodes.append(
                            {
                                "start_s": env0_start * dt,
                                "end_s": total_steps * dt,
                                "success": bool(stats["success"][0]),
                                "complete": bool(stats["success"][0] and stats["clear"][0]),
                                "end_progress": float(stats["end_progress"][0]),
                                "end_tcp_dist_m": float(stats["end_tcp_dist"][0]),
                            }
                        )
                        env0_start = total_steps
                    episodes.append(
                        {
                            "success": bool(stats["success"][i]),
                            "opened_any": bool(stats["opened_any"][i]),
                            "released_clear_at_end": bool(stats["clear"][i]),
                            "end_progress": float(stats["end_progress"][i]),
                            "end_tcp_dist_m": float(stats["end_tcp_dist"][i]),
                            "end_pad_force_n": float(stats["end_pad_force"][i]),
                            "time_to_success_s": float(stats["t_success"][i]),
                            "reached": bool(stats["reached"][i]),
                            "grasped": bool(stats["grasped"][i]),
                            "max_progress": float(stats["max_progress"][i]),
                            "max_pad_force_n": float(stats["max_pad_force"][i]),
                            "max_pad_squeeze_n": float(stats["max_pad_squeeze"][i]),
                            "max_pad_lateral_n": float(stats["max_pad_lateral"][i]),
                            "max_valve_rate_rad_s": float(stats["max_valve_rate"][i]),
                            "max_tcp_speed_m_s": float(stats["max_tcp_speed"][i]),
                            "commanded_rate_rad_s": float(stats["commanded_rate"][i]),
                            "achieved_rate_rad_s": float(stats["turn_rate_sum"][i] / stats["turn_steps"][i])
                            if stats["turn_steps"][i] > 0
                            else float("nan"),
                            "invalid": bool(invalid[i]),
                            "joint_steps": float(stats["joint_steps"][i]),
                            "joint_err_sq": stats["joint_err_sq"][i].tolist(),
                            "joint_acc_sq": stats["joint_acc_sq"][i].tolist(),
                            "joint_reversals": stats["joint_reversals"][i].tolist(),
                            "release_steps": float(stats["release_steps"][i]),
                            "release_peak_joint_speed_rad_s": float(stats["release_peak_speed"][i]),
                            "release_rms_acceleration_rad_s2": float(
                                (stats["release_acc_sq"][i] / stats["release_steps"][i]).sqrt()
                            ) if stats["release_steps"][i] > 0 else float("nan"),
                            "release_rms_velocity_error_rad_s": float(
                                (stats["release_velocity_err_sq"][i] / stats["release_steps"][i]).sqrt()
                            ) if bank_term is not None and stats["release_steps"][i] > 0 else float("nan"),
                            "length_s": float(step_count[i] * dt),
                            "bank_index": int(stats["bank_index"][i]),
                            "grasp_wrist_flexion": float(stats["grasp_wrist"][i]),
                        }
                    )
                    for key, value in stats.items():
                        value[i] = float("nan") if key in ("t_success", "grasp_wrist") else 0
                    if bank_term is not None:
                        stats["bank_index"][i] = bank_term.index[i]
                    stats["prev_q"][i] = robot.data.joint_pos.torch[i, arm_ids]
                    step_count[i] = 0
        expert_bank_stats = bank_joint_stats(bank_term.cfg.params["path"], dt) if bank_term is not None else None
        env.close()

    n = len(episodes)
    successes = sum(e["success"] for e in episodes)
    per_joint = joint_table(episodes, dt, expert_bank_stats)
    times = [e["time_to_success_s"] for e in episodes if e["success"]]
    t_mean = sum(times) / len(times) if times else float("nan")
    t_std = math.sqrt(sum((t - t_mean) ** 2 for t in times) / (len(times) - 1)) if len(times) > 1 else float("nan")
    lo, hi = wilson_interval(successes, n)
    summary = {
        "task": args.task,
        "agent": args.agent,
        "checkpoint": os.path.abspath(args.checkpoint),
        "hydra_overrides": hydra_args,
        "episodes": n,
        "success_rate": successes / n,
        "success_ci95_wilson": [lo, hi],
        "reliability_lower_bound_95": reliability_lower_bound(successes, n),
        "time_to_success_s": {
            "mean": t_mean,
            "std": t_std,
            "ci95_halfwidth": 1.96 * t_std / math.sqrt(len(times)) if len(times) > 1 else float("nan"),
        },
        "stage_rates": {
            "reached": sum(e["reached"] for e in episodes) / n,
            "grasped": sum(e["grasped"] for e in episodes) / n,
            "opened_held": successes / n,
            "opened_any": sum(e["opened_any"] for e in episodes) / n,
            "released_clear_at_end": sum(e["released_clear_at_end"] for e in episodes) / n,
            "complete": sum(e["success"] and e["released_clear_at_end"] for e in episodes) / n,
        },
        "mean_max_progress": sum(e["max_progress"] for e in episodes) / n,
        "max_pad_force_n": max(e["max_pad_force_n"] for e in episodes),
        # Per-episode peaks, median and 99th percentile.
        "pad_squeeze_n_p50_p99": _quantiles([e["max_pad_squeeze_n"] for e in episodes]),
        "pad_lateral_n_p50_p99": _quantiles([e["max_pad_lateral_n"] for e in episodes]),
        "mean_peak_valve_rate_rad_s": sum(e["max_valve_rate_rad_s"] for e in episodes) / n,
        "mean_peak_tcp_speed_m_s": sum(e["max_tcp_speed_m_s"] for e in episodes) / n,
        "mean_commanded_rate_rad_s": sum(e["commanded_rate_rad_s"] for e in episodes) / n,
        "mean_achieved_rate_rad_s": _nanmean([e["achieved_rate_rad_s"] for e in episodes]),
        "invalid_episodes": sum(e["invalid"] for e in episodes),
        "per_joint": per_joint,
        "release_motion": {
            "episodes_with_samples": sum(e["release_steps"] > 0 for e in episodes),
            "mean_peak_joint_speed_rad_s": _nanmean([
                e["release_peak_joint_speed_rad_s"] for e in episodes if e["release_steps"] > 0
            ]),
            "mean_rms_acceleration_rad_s2": _nanmean([e["release_rms_acceleration_rad_s2"] for e in episodes]),
            "mean_rms_velocity_error_rad_s": _nanmean([e["release_rms_velocity_error_rad_s"] for e in episodes]),
        },
        "per_episode": episodes,
    }
    if args.video:
        summary["video"] = {"dir": os.path.abspath(args.video), "env0_episodes": env0_episodes}
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        f"eval_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.json",
    )
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)

    if args.video:
        print(f"\nvideo of environment 0 in {os.path.abspath(args.video)}:")
        for k, e in enumerate(env0_episodes):
            outcome = "complete" if e["complete"] else ("switched/opened, not clear" if e["success"] else "FAILED")
            print(
                f"  episode {k}: {e['start_s']:6.1f}-{e['end_s']:6.1f} s  {outcome}"
                f" (end progress {e['end_progress']:.2f}, TCP {e['end_tcp_dist_m']:.3f} m from the target)"
            )
    print(f"\n=== {args.task} | {os.path.basename(args.checkpoint)} ({args.agent}) ===")
    print(f"success      {successes}/{n} = {successes / n:.1%}  (95% Wilson [{lo:.1%}, {hi:.1%}])")
    print(f"reliability  >= {summary['reliability_lower_bound_95']:.1%} at 95% confidence (Clopper-Pearson)")
    print(f"time         {t_mean:.2f} +/- {t_std:.2f} s to open")
    stages = summary["stage_rates"]
    print(
        f"stages       reached {stages['reached']:.1%} | grasped {stages['grasped']:.1%}"
        f" | opened held {stages['opened_held']:.1%} (any means {stages['opened_any']:.1%})"
        f" | released & clear at end {stages['released_clear_at_end']:.1%}"
    )
    print(f"complete     {stages['complete']:.1%} (opened while held, then let go and backed off with the valve open)")
    print(
        f"progress     mean max {summary['mean_max_progress']:.3f} | peak pad force {summary['max_pad_force_n']:.1f} N"
    )
    sq, lat = summary["pad_squeeze_n_p50_p99"], summary["pad_lateral_n_p50_p99"]
    print(
        f"pad force    per-episode peak, median / p99: squeeze {sq[0]:.0f} / {sq[1]:.0f} N"
        f" | lateral {lat[0]:.0f} / {lat[1]:.0f} N"
    )
    print(
        f"turn rate    commanded {summary['mean_commanded_rate_rad_s']:.2f} rad/s"
        f" | achieved {summary['mean_achieved_rate_rad_s']:.2f} rad/s (mean while held, 5-95% open)"
    )
    print(
        f"speeds       mean peak valve rate {summary['mean_peak_valve_rate_rad_s']:.2f} rad/s"
        f" | mean peak gripper speed {summary['mean_peak_tcp_speed_m_s']:.2f} m/s"
    )
    print(f"invalid      {summary['invalid_episodes']} episodes")
    print("per joint    (policy | expert bank), from reset until open-while-held")
    print(f"  {'joint':<28}{'RMS err vs expert':>18}{'RMS accel':>22}{'reversals/s':>20}")
    for name, row in per_joint.items():
        expert = row["expert"]
        print(
            f"  {name:<28}{row['rms_error_rad']:>12.3f} rad"
            f"{row['rms_accel']:>9.2f} | {expert['rms_accel']:<6.2f} rad/s2"
            f"{row['reversals_per_s']:>9.2f} | {expert['reversals_per_s']:<6.2f}"
        )
    print(f"written      {output}")


if __name__ == "__main__":
    main()
