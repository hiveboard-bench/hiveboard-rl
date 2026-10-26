import argparse
import datetime
import json
import math
import os
import sys

import gymnasium as gym
import torch
from isaaclab_hiveboard_rl import env as mdp

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils import video_recorder
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

REVERSAL_RATE = 0.05

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-HiveBoard-Anymal-OpenBallValve-RL-Play-v0")
parser.add_argument("--checkpoint")
parser.add_argument("--policy", choices=("checkpoint", "expert"), default="checkpoint")
parser.add_argument("--save", help="record expert demos to this file")
parser.add_argument("--command", choices=mdp.ARM_COMMANDS, default="increment")
parser.add_argument("--keep_failed", action="store_true")
parser.add_argument("--export", action="store_true")
parser.add_argument("--video", help="Newton viewer video dir")
parser.add_argument("--publication", help="RTX video dir")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
expert = args.policy == "expert" or args.save is not None
if not any(t.startswith(("physics=", "presets=")) for t in hydra_args):
    hydra_args.append("physics=newton_mjwarp")
if not expert:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hydra_args += checkpoint.get("bc_overrides", [])
if args.publication:
    args.num_envs = args.episodes = 1
args.visualizer = ["newton_rtx"] if args.publication else ["newton_gl"] if args.video else []
sys.argv = [sys.argv[0], *hydra_args]

env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = min(args.num_envs, args.episodes)
env_cfg.seed = args.seed
reset_cfg = env_cfg.events.reset_from_bank
if expert:
    env_cfg.actions.arm_action.command = args.command
    env_cfg.terminations.expert_drift = None
    env_cfg.terminations.expert_mechanism_lag = None
    reset_cfg.func = mdp.reset_from_expert_bank_in_order
    reset_cfg.params["seed"] = args.seed
if args.publication:
    mdp.merge(env_cfg, mdp.load("train")["publication"])
if args.publication or (expert and args.save is None):
    for term in (env_cfg.events.arm_delay, env_cfg.events.target_delay):
        if term is not None:
            term.params["delay_range_s"] = (0.0, 0.0)
source = reset_cfg.params["path"] if expert else args.checkpoint
steps = int(math.ceil(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)))
if args.video:
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.8, 0.5, 0.8)
    env_cfg.viewer.lookat = (0.95, 0.0, 0.65)
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            output_dir=os.path.abspath(args.video),
            video_length=steps * (math.ceil(args.episodes / env_cfg.scene.num_envs) + 1),
            video_interval=0,
        )
    ]
if args.publication:
    from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

    env_cfg.viewer.resolution = (1920, 1080)
    env_cfg.sim.visualizer_cfgs = [
        NewtonRTXVisualizerCfg(
            headless=True,
            eye=env_cfg.viewer.eye,
            lookat=env_cfg.viewer.lookat,
            rtx_environment="studio",
            render_settings={"omni:rtx:quality": ("Int", 100)},
        )
    ]
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="visualizer:newton_rtx",
            output_dir=os.path.abspath(args.publication),
            output_filename_prefix="publication",
            video_length=steps + 1,
        )
    ]

    class PublicationClip(video_recorder.ImageSequenceClip):
        def write_videofile(self, filename, **kwargs):
            kwargs.update(fps=50, preset="slow", ffmpeg_params=["-crf", "12"])
            super().write_videofile(filename, **kwargs)

    video_recorder.ImageSequenceClip = PublicationClip

with launch_simulation(env_cfg, args):
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    clip_actions = None if expert else agent_cfg.clip_actions
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=clip_actions)
    uenv = env.unwrapped
    if not expert:
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(os.path.abspath(args.checkpoint))
        if args.export:
            export_dir = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "exported")
            runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
        policy = runner.get_inference_policy(device=uenv.device)

    n_envs, dev, dt = uenv.num_envs, uenv.device, uenv.step_dt
    turn = uenv.command_manager.get_term("reference")
    robot = uenv.scene["robot"]
    profile = uenv.cfg.robot
    gripper_body = robot.find_bodies(profile.ee.body_name)[0][0]
    arm_ids = robot.find_joints(list(profile.arm_joint_names), preserve_order=True)[0]
    gripper_action = uenv.action_manager.get_term("gripper_action")
    bank_term = uenv.expert_bank_term
    wrist_col = list(profile.arm_joint_names).index(profile.diagnostic_joint)
    stats = {
        key: torch.zeros(n_envs, dtype=torch.bool, device=dev)
        for key in ("reached", "grasped", "success", "opened_any", "clear")
    }
    stats |= {
        key: torch.zeros(n_envs, device=dev)
        for key in (
            "end_progress",
            "end_tcp_dist",
            "end_pad_force",
            "max_progress",
            "max_pad_force",
            "max_pad_squeeze",
            "max_pad_lateral",
            "max_mechanism_rate",
            "max_tcp_speed",
            "turn_rate_sum",
            "turn_steps",
            "joint_steps",
            "release_steps",
            "release_peak_speed",
            "release_acc_sq",
            "release_velocity_err_sq",
        )
    }
    stats |= {
        key: torch.zeros(n_envs, len(arm_ids), device=dev)
        for key in ("joint_err_sq", "joint_acc_sq", "joint_reversals", "prev_measured_v", "prev_v")
    }
    stats |= {
        "t_success": torch.full((n_envs,), float("nan"), device=dev),
        "grasp_wrist": torch.full((n_envs,), float("nan"), device=dev),
        "commanded_rate": turn.rate.clone(),
        "prev_q": robot.data.joint_pos.torch[:, arm_ids].clone(),
        "bank_index": bank_term.index.clone(),
    }
    step_count = torch.zeros(n_envs, dtype=torch.long, device=dev)
    env0_episodes = []
    env0_start = total_steps = 0
    episodes = []
    obs = env.get_observations()
    if args.save:
        rows = torch.arange(n_envs, device=dev)
        horizon = int(uenv.max_episode_length) + 1
        buffers = {
            "policy": torch.zeros(n_envs, horizon, obs["policy"].shape[-1], device=dev),
            "teacher": torch.zeros(n_envs, horizon, obs["teacher"].shape[-1], device=dev),
            "actions": torch.zeros(
                n_envs, horizon, uenv.action_manager.total_action_dim, device=dev
            ),
            "rewards": torch.zeros(n_envs, horizon, device=dev),
        }
        demos = []
    with torch.inference_mode():
        while len(episodes) < args.episodes:
            dist, ang = mdp.tcp_grasp_error(uenv)
            progress = mdp.mechanism_progress(uenv)
            opened = mdp.succeeded(uenv)
            stats["reached"] |= dist < mdp.HOLD_DIST
            held = mdp.held(uenv)
            first_hold = held & stats["grasp_wrist"].isnan()
            stats["grasp_wrist"][first_hold] = robot.data.joint_pos.torch[
                first_hold, arm_ids[wrist_col]
            ]
            stats["grasped"] |= held
            stats["opened_any"] |= opened
            stats["clear"] = mdp.released_and_clear(uenv)
            stats["end_progress"] = progress.clone()
            stats["end_tcp_dist"] = dist.clone()
            stats["end_pad_force"] = mdp.pad_force(uenv).max(dim=-1).values
            credited = opened & mdp.opening_credited(uenv, held, stats["grasped"])
            new = credited & ~stats["success"]
            stats["t_success"][new] = step_count[new] * dt
            stats["success"] |= credited
            stats["max_progress"] = torch.maximum(stats["max_progress"], progress)
            turning = held & (progress > 0.05) & (progress < 0.95)
            stats["turn_rate_sum"] += mdp.mechanism_state(uenv)[:, 1].abs() * turning
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
            stats["joint_err_sq"] += mdp.expert_joint_error(uenv).square() * active
            stats["joint_steps"] += active[:, 0]
            measured_v = robot.data.joint_vel.torch[:, arm_ids]
            releasing = (
                stats["success"] & (gripper_action.raw_actions[:, 0] > 0) & (step_count >= 2)
            )
            stats["release_steps"] += releasing.float()
            stats["release_peak_speed"] = torch.maximum(
                stats["release_peak_speed"], measured_v.abs().amax(dim=-1) * releasing
            )
            stats["release_acc_sq"] += ((measured_v - stats["prev_measured_v"]) / dt).square().mean(
                dim=-1
            ) * releasing
            velocity_reference = bank_term.bank.velocity_reference(
                bank_term.index, mdp.reference_step(uenv)
            )
            stats["release_velocity_err_sq"] += (measured_v - velocity_reference).square().mean(
                dim=-1
            ) * releasing
            stats["prev_measured_v"] = measured_v.clone()
            stats["prev_q"] = q.clone()
            stats["prev_v"] = v.clone()
            speed = torch.norm(robot.data.body_lin_vel_w.torch[:, gripper_body], dim=-1)
            stats["max_tcp_speed"] = torch.maximum(stats["max_tcp_speed"], speed)
            stats["max_mechanism_rate"] = torch.maximum(
                stats["max_mechanism_rate"], mdp.mechanism_state(uenv)[:, 1].abs()
            )
            squeeze, lateral = mdp.pad_force_split(uenv)
            stats["max_pad_squeeze"] = torch.maximum(
                stats["max_pad_squeeze"], squeeze.max(dim=-1).values
            )
            stats["max_pad_lateral"] = torch.maximum(
                stats["max_pad_lateral"], lateral.max(dim=-1).values
            )
            stats["max_pad_force"] = torch.maximum(
                stats["max_pad_force"], mdp.pad_force(uenv).max(dim=-1).values
            )

            action = mdp.expert_action(uenv) if expert else policy(obs)
            if args.save:
                buffers["policy"][rows, step_count] = obs["policy"]
                buffers["teacher"][rows, step_count] = obs["teacher"]
                buffers["actions"][rows, step_count] = action
            obs, reward, dones, _ = env.step(action)
            if args.save:
                buffers["rewards"][rows, step_count] = reward
            if not expert:
                policy.reset(dones)
            step_count += 1
            total_steps += 1
            invalid = uenv.termination_manager.get_term("invalid")
            for i in dones.nonzero(as_tuple=False).flatten().tolist():
                if len(episodes) >= args.episodes:
                    break
                if i == 0:
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
                release_steps = stats["release_steps"][i]
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
                        "max_mechanism_rate_rad_s": float(stats["max_mechanism_rate"][i]),
                        "max_tcp_speed_m_s": float(stats["max_tcp_speed"][i]),
                        "commanded_rate_rad_s": float(stats["commanded_rate"][i]),
                        "achieved_rate_rad_s": float(
                            stats["turn_rate_sum"][i] / stats["turn_steps"][i]
                        )
                        if stats["turn_steps"][i] > 0
                        else float("nan"),
                        "invalid": bool(invalid[i]),
                        "joint_steps": float(stats["joint_steps"][i]),
                        "joint_err_sq": stats["joint_err_sq"][i].tolist(),
                        "joint_acc_sq": stats["joint_acc_sq"][i].tolist(),
                        "joint_reversals": stats["joint_reversals"][i].tolist(),
                        "release_steps": float(release_steps),
                        "release_peak_joint_speed_rad_s": float(stats["release_peak_speed"][i]),
                        "release_rms_acceleration_rad_s2": float(
                            (stats["release_acc_sq"][i] / release_steps).sqrt()
                        )
                        if release_steps > 0
                        else float("nan"),
                        "release_rms_velocity_error_rad_s": float(
                            (stats["release_velocity_err_sq"][i] / release_steps).sqrt()
                        )
                        if release_steps > 0
                        else float("nan"),
                        "length_s": float(step_count[i] * dt),
                        "bank_index": int(stats["bank_index"][i]),
                        "grasp_wrist_flexion": float(stats["grasp_wrist"][i]),
                    }
                )
                if args.save and (episodes[-1]["success"] or args.keep_failed):
                    n = int(step_count[i])
                    terms = uenv.termination_manager
                    demos.append(
                        {
                            **{key: buffer[i, :n].cpu() for key, buffer in buffers.items()},
                            "lengths": n,
                            "success": episodes[-1]["success"],
                            "bank_index": episodes[-1]["bank_index"],
                            "timeout": bool(terms.time_outs[i] and not terms.terminated[i]),
                        }
                    )
                for key, value in stats.items():
                    value[i] = float("nan") if key in ("t_success", "grasp_wrist") else 0
                stats["bank_index"][i] = bank_term.index[i]
                stats["prev_q"][i] = robot.data.joint_pos.torch[i, arm_ids]
                step_count[i] = 0
    if args.publication:
        uenv.video_recorders[0]._frames.pop()
    bank_path = bank_term.cfg.params["path"]
    env.close()

n = len(episodes)
successes = sum(e["success"] for e in episodes)
if args.save:
    data = {key: torch.cat([d[key] for d in demos]) for key in buffers}
    for key in ("lengths", "success", "bank_index", "timeout"):
        data[key] = torch.tensor([d[key] for d in demos])
    data["meta"] = {
        "task": args.task,
        "command": args.command,
        "bank": os.path.abspath(bank_path),
        "seed": args.seed,
        "dt": dt,
        "episodes_played": n,
        "replay_success_rate": successes / n,
        "keep_failed": args.keep_failed,
        "unique_trajectories": len(set(d["bank_index"] for d in demos)),
        "hydra_overrides": hydra_args,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    torch.save(data, args.save)
    print(f"[DEMOS] {len(demos)} episodes, {data['policy'].shape[0]} steps -> {args.save}")

bank = torch.load(bank_path, map_location="cpu", weights_only=False)
bank_q = bank["arm_q"].float()
open_step = (bank["t_open"] / dt).round().long()
bank_v = (bank_q[:, 1:] - bank_q[:, :-1]) / dt
bank_a = (bank_v[:, 1:] - bank_v[:, :-1]) / dt
mask = (torch.arange(bank_a.shape[1])[None] + 2 <= open_step[:, None]).float()[..., None]
bank_flip = (
    (torch.sign(bank_v[:, 1:]) != torch.sign(bank_v[:, :-1]))
    & (bank_v[:, 1:].abs() > REVERSAL_RATE)
    & (bank_v[:, :-1].abs() > REVERSAL_RATE)
)
count = mask.sum(dim=(0, 1))
expert_accel = (bank_a.square() * mask).sum(dim=(0, 1)).div(count).sqrt()
expert_reversals = (bank_flip.float() * mask).sum(dim=(0, 1)).div(count * dt)
joint_steps = max(sum(e["joint_steps"] for e in episodes), 1)
per_joint = {}
for j, name in enumerate(profile.arm_joint_names):
    per_joint[name] = {
        "rms_error_rad": math.sqrt(sum(e["joint_err_sq"][j] for e in episodes) / joint_steps),
        "rms_accel": math.sqrt(sum(e["joint_acc_sq"][j] for e in episodes) / joint_steps),
        "reversals_per_s": sum(e["joint_reversals"][j] for e in episodes) / (joint_steps * dt),
        "expert": {
            "rms_accel": float(expert_accel[j]),
            "reversals_per_s": float(expert_reversals[j]),
        },
    }

times = [e["time_to_success_s"] for e in episodes if e["success"]]
t_mean = sum(times) / len(times) if times else float("nan")
t_std = (
    math.sqrt(sum((t - t_mean) ** 2 for t in times) / (len(times) - 1))
    if len(times) > 1
    else float("nan")
)
p, z = successes / n, 1.96
center = (p + z * z / (2 * n)) / (1 + z * z / n)
half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
lo, hi = max(0.0, center - half), min(1.0, center + half)
reliability, upper = 0.0, 1.0
for _ in range(60):
    mid = 0.5 * (reliability + upper)
    tail = sum(math.comb(n, k) * mid**k * (1 - mid) ** (n - k) for k in range(successes, n + 1))
    reliability, upper = (mid, upper) if tail < 0.05 else (reliability, mid)


def nanmean(values):
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


quantiles = torch.tensor((0.5, 0.99))
summary = {
    "task": args.task,
    "robot": profile.name,
    "diagnostic_joint": profile.diagnostic_joint,
    "agent": "rsl_rl_cfg_entry_point",
    "checkpoint": os.path.abspath(source),
    "hydra_overrides": hydra_args,
    "episodes": n,
    "success_rate": successes / n,
    "success_ci95_wilson": [lo, hi],
    "reliability_lower_bound_95": reliability,
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
    "pad_squeeze_n_p50_p99": torch.quantile(
        torch.tensor([e["max_pad_squeeze_n"] for e in episodes]), quantiles
    ).tolist(),
    "pad_lateral_n_p50_p99": torch.quantile(
        torch.tensor([e["max_pad_lateral_n"] for e in episodes]), quantiles
    ).tolist(),
    "mean_peak_mechanism_rate_rad_s": sum(e["max_mechanism_rate_rad_s"] for e in episodes) / n,
    "mean_peak_tcp_speed_m_s": sum(e["max_tcp_speed_m_s"] for e in episodes) / n,
    "mean_commanded_rate_rad_s": sum(e["commanded_rate_rad_s"] for e in episodes) / n,
    "mean_achieved_rate_rad_s": nanmean([e["achieved_rate_rad_s"] for e in episodes]),
    "invalid_episodes": sum(e["invalid"] for e in episodes),
    "per_joint": per_joint,
    "release_motion": {
        "episodes_with_samples": sum(e["release_steps"] > 0 for e in episodes),
        "mean_peak_joint_speed_rad_s": nanmean(
            [e["release_peak_joint_speed_rad_s"] for e in episodes if e["release_steps"] > 0]
        ),
        "mean_rms_acceleration_rad_s2": nanmean(
            [e["release_rms_acceleration_rad_s2"] for e in episodes]
        ),
        "mean_rms_velocity_error_rad_s": nanmean(
            [e["release_rms_velocity_error_rad_s"] for e in episodes]
        ),
    },
    "per_episode": episodes,
}
if args.video:
    summary["video"] = {"dir": os.path.abspath(args.video), "env0_episodes": env0_episodes}
output = args.output or os.path.join(
    args.publication or os.path.dirname(os.path.abspath(source)),
    f"eval_{os.path.splitext(os.path.basename(source))[0]}.json",
)
with open(output, "w") as f:
    json.dump(summary, f, indent=2)

if args.video:
    print(f"\nvideo of environment 0 in {os.path.abspath(args.video)}:")
    for k, e in enumerate(env0_episodes):
        outcome = "complete" if e["complete"] else "opened, not clear" if e["success"] else "FAILED"
        print(
            f"  episode {k}: {e['start_s']:6.1f}-{e['end_s']:6.1f} s  {outcome}"
            f" (end progress {e['end_progress']:.2f},"
            f" TCP {e['end_tcp_dist_m']:.3f} m from the target)"
        )
stages = summary["stage_rates"]
sq, lat = summary["pad_squeeze_n_p50_p99"], summary["pad_lateral_n_p50_p99"]
print(f"\n=== {args.task} | {os.path.basename(source)} ===")
print(f"success      {successes}/{n} = {successes / n:.1%}  (95% Wilson [{lo:.1%}, {hi:.1%}])")
print(f"reliability  >= {reliability:.1%} at 95% confidence (Clopper-Pearson)")
print(f"time         {t_mean:.2f} +/- {t_std:.2f} s to open")
print(
    f"stages       reached {stages['reached']:.1%} | grasped {stages['grasped']:.1%}"
    f" | opened held {stages['opened_held']:.1%} (any means {stages['opened_any']:.1%})"
    f" | released & clear at end {stages['released_clear_at_end']:.1%}"
)
print(f"complete     {stages['complete']:.1%} (opened while held, then released and clear)")
print(
    f"progress     mean max {summary['mean_max_progress']:.3f}"
    f" | peak pad force {summary['max_pad_force_n']:.1f} N"
)
print(
    f"pad force    per-episode peak, median / p99: squeeze {sq[0]:.0f} / {sq[1]:.0f} N"
    f" | lateral {lat[0]:.0f} / {lat[1]:.0f} N"
)
print(
    f"turn rate    commanded {summary['mean_commanded_rate_rad_s']:.2f} rad/s"
    f" | achieved {summary['mean_achieved_rate_rad_s']:.2f} rad/s (mean while held, 5-95% open)"
)
print(
    f"speeds       mean peak mechanism rate {summary['mean_peak_mechanism_rate_rad_s']:.2f} rad/s"
    f" | mean peak gripper speed {summary['mean_peak_tcp_speed_m_s']:.2f} m/s"
)
print(f"invalid      {summary['invalid_episodes']} episodes")
print("per joint    (policy | expert bank), from reset until open-while-held")
print(f"  {'joint':<28}{'RMS err vs expert':>18}{'RMS accel':>22}{'reversals/s':>20}")
for name, row in per_joint.items():
    print(
        f"  {name:<28}{row['rms_error_rad']:>12.3f} rad"
        f"{row['rms_accel']:>9.2f} | {row['expert']['rms_accel']:<6.2f} rad/s2"
        f"{row['reversals_per_s']:>9.2f} | {row['expert']['reversals_per_s']:<6.2f}"
    )
print(f"written      {output}")
