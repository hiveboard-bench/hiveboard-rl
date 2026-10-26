import argparse
import math
import os
import time

import torch
from isaaclab_hiveboard_rl import env as mdp

import isaaclab.utils.math as math_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import PHASES

parser = argparse.ArgumentParser()
parser.add_argument("--robot", default="anymal")
parser.add_argument("--task", default="open_ball_valve")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_trajectories", type=int, default=5000)
parser.add_argument("--max_waves", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default=None)
parser.add_argument("--video", default=None, help="record env 0 to this directory")
# TODO: robustness study knobs, remove after the study
parser.add_argument("--arm_gain_scale", type=float, default=1.0)
parser.add_argument("--arm_delay_s", type=float, default=0.0)
parser.add_argument("--reg_bias_pos", type=float, default=0.0, help="shift after planning")
parser.add_argument("--reg_bias_rot", type=float, default=0.0, help="tilt after planning")
add_launcher_args(parser)
args = parser.parse_args()
args.visualizer = ["newton_gl"] if args.video else []

robot = mdp.load(f"robots/{args.robot}")
output = args.output or mdp.bank_path(args.robot, args.task)
env_cfg = mdp.compose(args.robot, args.task, "expert")
profile = env_cfg.robot
arm = list(profile.arm_joint_names)
env_cfg.sim.physics = env_cfg.sim.physics.newton_mjwarp
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env_cfg.episode_length_s += 1.0
env_cfg.observations.teacher = None
env_cfg.observations.policy.rate = None
env_cfg.observations.policy.episode_time = None
env_cfg.rewards = {}
env_cfg.terminations = {"time_out": env_cfg.terminations.time_out}
env_cfg.recorders = RecorderManagerBaseCfg()
max_open_s = mdp.board.expert(env_cfg)
env_cfg.commands.reference = None
actuators = dict(env_cfg.scene.robot.actuators)
for name in profile.actuator_groups:
    a = actuators[name]
    actuators[name] = a.replace(
        stiffness=a.stiffness * args.arm_gain_scale, damping=a.damping * args.arm_gain_scale
    )
env_cfg.scene.robot = mdp.delayed_actuators(
    env_cfg.scene.robot.replace(actuators=actuators),
    robot["delayed_groups"],
    args.arm_delay_s,
    env_cfg.sim.dt,
)
if args.arm_delay_s:
    env_cfg.events.arm_delay = EventTermCfg(
        func=mdp.randomize_actuator_delay,
        mode="reset",
        params={"delay_range_s": (args.arm_delay_s,) * 2, "groups": robot["delayed_groups"]},
    )
if args.video:
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.55, 1.05, 1.05)
    env_cfg.viewer.lookat = (0.9, 0.0, 0.8)
    horizon = math.ceil(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            output_dir=os.path.abspath(args.video),
            video_length=horizon * args.max_waves,
            video_interval=0,
        )
    ]

with launch_simulation(env_cfg, args):
    env = ManagerBasedRLEnv(cfg=env_cfg)
    dev, n_envs = env.device, env.num_envs
    robot, mechanism = env.scene["robot"], mdp.mechanism_asset(env)
    arm_ids = robot.find_joints(env_cfg.actions.arm_action.joint_names, preserve_order=True)[0]
    grip_id = robot.find_joints(env_cfg.actions.gripper_action.joint_names)[0][0]
    hand_ids, hand_names = robot.find_joints(list(profile.hand_joint_names), preserve_order=True)
    command = env.command_manager.get_term("pose_command")
    wrist_id = list(env_cfg.actions.arm_action.joint_names).index(profile.diagnostic_joint)
    actuate = PHASES.index("actuate")
    horizon = int(env.max_episode_length)
    kept = {}
    sampled = {
        k: []
        for k in (
            "dynamics",
            "expert_diversity",
            "arm_q0",
            "turn_wrist",
            "mechanism_quat",
            "ok",
            "progress",
            "grasp_in_frame",
        )
    }
    stats = {
        k: 0
        for k in (
            "waves",
            "episodes",
            "successes",
            "fallback",
            "kept",
            "slow",
            "not_clear",
            "complete",
            "opened_held",
            "opened_any",
        )
    }
    stats["wall_s"] = 0.0
    for wave in range(args.max_waves):
        t0 = time.perf_counter()
        env.reset()
        init = {
            "arm_q0": robot.data.joint_pos.torch[:, arm_ids].clone(),
            "mechanism_position0": mdp.mechanism_position(env).clone(),
            "mechanism_pose_env": torch.cat(
                (
                    mechanism.data.root_pos_w.torch - env.scene.env_origins,
                    mechanism.data.root_quat_w.torch,
                ),
                dim=-1,
            ).clone(),
            "dynamics": env.dynamics_term.params.clone(),
            "expert_diversity": env.expert_diversity_term.values.clone(),
        }
        steps = {
            k: []
            for k in (
                "arm_q",
                "arm_target",
                "gripper_cmd",
                "gripper_q",
                "mechanism_position",
                "tcp_pose",
                "phase",
                "phase_id",
                "pad_force",
                "gripper_joint_pos",
            )
        }
        opened_held = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        ever_held = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        opened_any = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        fallback = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        best_progress = torch.zeros(n_envs, device=dev)
        t_open = torch.full((n_envs,), float("nan"), device=dev)
        grasp_in_frame = torch.full((n_envs, 7), float("nan"), device=dev)
        with torch.inference_mode():
            for t in range(horizon):
                if t == 3 and (args.reg_bias_pos > 0.0 or args.reg_bias_rot > 0.0):
                    pose = torch.cat(
                        (mechanism.data.root_pos_w.torch, mechanism.data.root_quat_w.torch), dim=-1
                    ).clone()
                    pose[:, :3] += (torch.rand(n_envs, 3, device=dev) * 2 - 1) * args.reg_bias_pos
                    euler = (torch.rand(n_envs, 3, device=dev) * 2 - 1) * args.reg_bias_rot
                    delta = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
                    pose[:, 3:] = math_utils.quat_mul(delta, pose[:, 3:])
                    mechanism.write_root_pose_to_sim_index(root_pose=pose)
                    mechanism.write_root_velocity_to_sim_index(
                        root_velocity=torch.zeros(n_envs, 6, device=dev)
                    )
                if t == horizon - 1:
                    clear = mdp.released_and_clear(env)
                action = command.command
                starts = (command.phase == actuate) & grasp_in_frame[:, 0].isnan()
                if torch.any(starts):
                    rel = math_utils.subtract_frame_transforms(
                        *mdp.grasp_target_w(env), *mdp.tcp_w(env)
                    )
                    grasp_in_frame[starts] = torch.cat(rel, dim=-1)[starts]
                steps["arm_q"].append(robot.data.joint_pos.torch[:, arm_ids].clone())
                steps["arm_target"].append(action[:, :-1].clone())
                steps["gripper_cmd"].append(action[:, -1].clone())
                steps["gripper_q"].append(robot.data.joint_pos.torch[:, grip_id].clone())
                steps["mechanism_position"].append(mdp.mechanism_position(env).clone())
                steps["tcp_pose"].append(mdp.tcp_pose_b(env).clone())
                steps["phase"].append(command._current_command_idx.clone())
                steps["phase_id"].append(command.phase.clone())
                steps["pad_force"].append(mdp.pad_force(env).clone())
                steps["gripper_joint_pos"].append(robot.data.joint_pos.torch[:, hand_ids].clone())
                env.step(action)
                held = mdp.held(env)
                is_open = mdp.succeeded(env)
                ever_held |= held
                opened_any |= is_open
                best_progress = torch.maximum(best_progress, mdp.mechanism_progress(env))
                now_open = is_open & mdp.opening_credited(env, held, ever_held) & ~opened_held
                t_open[now_open] = (t + 1) * env.step_dt
                opened_held |= now_open
                fallback |= command.expert_fallback()
        wave_s = time.perf_counter() - t0
        ok = opened_held & ~fallback & clear
        slow = ok & (t_open > max_open_s)
        ok &= ~slow
        counts = {
            "successes": opened_held,
            "fallback": opened_held & fallback,
            "kept": ok,
            "slow": slow,
            "not_clear": opened_held & ~fallback & ~clear,
            "complete": opened_held & clear,
            "opened_held": opened_held,
            "opened_any": opened_any,
        }
        for key, value in counts.items():
            counts[key] = int(value.sum())
            stats[key] += counts[key]
        stats["waves"] += 1
        stats["episodes"] += n_envs
        stats["wall_s"] += wave_s
        for key, values in steps.items():
            kept.setdefault(key, []).append(torch.stack(values, dim=1)[ok].cpu())
        for key, value in init.items():
            kept.setdefault(key, []).append(value[ok].cpu())
        kept.setdefault("t_open", []).append(t_open[ok].cpu())
        kept.setdefault("grasp_in_frame", []).append(grasp_in_frame[ok].cpu())
        kept.setdefault("expert_fallback", []).append(fallback[ok].cpu())
        arm_q, phase = torch.stack(steps["arm_q"], dim=1), torch.stack(steps["phase_id"], dim=1)
        turning = phase == actuate
        wrist = arm_q[torch.arange(n_envs, device=dev), turning.float().argmax(dim=1), wrist_id]
        sampled["dynamics"].append(init["dynamics"].cpu())
        sampled["expert_diversity"].append(init["expert_diversity"].cpu())
        sampled["arm_q0"].append(init["arm_q0"].cpu())
        sampled["mechanism_quat"].append(init["mechanism_pose_env"][:, 3:].cpu())
        sampled["turn_wrist"].append(torch.where(turning.any(dim=1), wrist, torch.nan).cpu())
        sampled["ok"].append(ok.cpu())
        sampled["progress"].append(best_progress.cpu())
        sampled["grasp_in_frame"].append(grasp_in_frame.cpu())
        print(
            f"[BANK] wave {wave}: {counts['opened_held']}/{n_envs} opened while held, "
            f"{counts['fallback']} discarded for a fallback, "
            f"{counts['not_clear']} not released and clear at the end, "
            f"{counts['slow']} opened too late, in {wave_s:.1f}s; kept {stats['kept']}",
            flush=True,
        )
        if stats["kept"] >= args.num_trajectories:
            break
    n = stats["episodes"]
    rate = stats["successes"] / n
    assert stats["kept"] > 0, "no successful expert trajectories"
    bank = {key: torch.cat(values)[: args.num_trajectories] for key, values in kept.items()}
    bank["meta"] = {
        "robot": profile.name,
        "task": args.task,
        "diagnostic_joint": profile.diagnostic_joint,
        "physics_dt": env.physics_dt,
        "seed": args.seed,
        "task_cfg": env_cfg.task.to_dict(),
        "method": env_cfg.task.asset.method,
        "dt": env.step_dt,
        "horizon": horizon,
        "arm_joint_names": arm,
        "gripper_joint_names": list(hand_names),
        "phases": list(PHASES),
        "dynamics": list(mdp.board.DYNAMICS),
        "expert_diversity": list(mdp.board.EXPERT_DIVERSITY),
        "expert_success": rate,
        "stats": stats,
        "sampled": {key: torch.cat(values) for key, values in sampled.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(bank, output)
    print(
        f"[BANK] saved {len(bank['arm_q'])} trajectories to {output}\n"
        f"[BANK] expert success {stats['successes']}/{n} = {rate:.1%}, "
        f"{stats['fallback']} successes discarded for using a cuRobo fallback\n"
        f"[BANK] wall time {stats['wall_s']:.0f}s over {stats['waves']} waves = "
        f"{stats['wall_s'] / stats['kept']:.2f}s per kept trajectory",
        flush=True,
    )
    t = bank["t_open"]
    print(f"[BANK] time to open {t.mean():.2f} +/- {t.std():.2f}s", flush=True)
    g = torch.cat(sampled["grasp_in_frame"])
    g = g[~g[:, 0].isnan()]
    g[:, 3:] *= torch.where(g[:, 6:7] < 0, -1.0, 1.0)
    task = env_cfg.task.asset
    print(
        f"[BANK] TCP in grasp frame '{task.grasp_frame}' when the turn starts, "
        f"median: pos {[round(v, 4) for v in g[:, :3].median(dim=0).values.tolist()]}, "
        f"xyzw {[round(v, 4) for v in g[:, 3:].median(dim=0).values.tolist()]}",
        flush=True,
    )
    env.close()
