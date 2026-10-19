#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Precompute a bank of cuRobo expert trajectories for an RL valve task.

Runs the scripted cuRobo expert of the RL task ``--task`` (its env module's
``EXPERT_TASK``, e.g. ``Isaac-HiveBoard-Anymal-BallValve-v0``, with its saved
command setup) in many parallel environments, with the RL task's
reset randomization (valve pose, initial valve angle, arm start posture) and
valve dynamics (``valve_dynamics.py``, including valves stuck closed), at the
RL task's 20 Hz control rate. Each episode also varies the expert itself
(``expert_diversity.py``: turning speed, reach speed, grasp point and grasp
angle around the lever, and the arm's IK branch). cuRobo plans all
environments entering a segment on the same tick as one batch
(``plan_batch_size``).

Environments run in synchronized waves: every episode lasts the same fixed
time and all reset together, so the planning batches stay large. A trajectory
is kept when the valve reaches open while the lever is held at the expert grasp
pose (the RL task's own success definition). The bank stores, per step, arm
joints and the expert's joint targets, gripper command and position, valve
angle, TCP pose, sequence segment and its named phase (``PHASES``), plus each episode's initial state, valve
dynamics and expert variation::

    uv run python scripts/rl/build_expert_bank.py --num_envs 512 --num_trajectories 5000
    uv run python scripts/rl/build_expert_bank.py --task Isaac-HiveBoard-Anymal-SmallValve-RL-v0
"""

import argparse
import importlib
import math
import os
import sys
import time

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "imitation"))
from _common import _apply_command_setup  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--task", default="Isaac-HiveBoard-Anymal-BallValve-RL-v0", help="RL task whose expert bank to build."
)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_trajectories", type=int, default=5000, help="Successful trajectories to keep.")
parser.add_argument("--plan_batch_size", type=int, default=None, help="cuRobo batch (default: num_envs).")
parser.add_argument(
    "--episode_length_s", type=float, default=None, help="Default: one second past the RL task's episode."
)
parser.add_argument("--max_waves", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default=None, help="Default: logs/expert_bank/<the RL task's BANK_PREFIX>_bank.pt.")
parser.add_argument("--cuda_graph", action="store_true", help="Capture Newton physics steps in a CUDA graph.")
parser.add_argument("--nominal_valve", action="store_true", help="Fixed valve dynamics (no randomization).")
parser.add_argument("--nominal_expert", action="store_true", help="The authored expert (no speed or grasp variation).")
parser.add_argument("--nominal_pose", action="store_true", help="The previous, narrower valve placement range.")
parser.add_argument("--nominal_arm", action="store_true", help="Start the arm around the home posture only.")
parser.add_argument(
    "--standard", action="store_true", help="Standard conditions: valve at its nominal pose and angle, arm at home."
)
parser.add_argument(
    "--no_wrist_flip", action="store_true", help="No wrist-flipped starts; the IK solver picks the branch."
)
parser.add_argument("--no_stuck", action="store_true", help="No valves stuck closed.")
# Baseline evaluation of the expert itself under perturbations (nothing is saved).
parser.add_argument("--eval_only", action="store_true", help="Report the expert's success rates, save no bank.")
parser.add_argument("--arm_gain_scale", type=float, default=1.0, help="Scale on the arm PD stiffness and damping.")
parser.add_argument("--arm_delay_s", type=float, default=0.0, help="Fixed arm command latency [s].")
parser.add_argument(
    "--reg_bias_pos", type=float, default=0.0, help="Registration error [m]: the valve moves this much after planning."
)
parser.add_argument("--reg_bias_rot", type=float, default=0.0, help="Registration error [rad] about each axis.")
parser.add_argument(
    "--dump_all", default=None, help="Also save every episode's steps and outcome (kept or not) to this file."
)
parser.add_argument("--setup", default=None)
parser.add_argument("--no_setup", action="store_true")
parser.add_argument(
    "--video",
    default=None,
    help="Record environment 0 over all waves to this directory (Newton viewer). Use a few envs and waves.",
)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
if not any(t.startswith(("physics=", "presets=")) for t in hydra_args):
    hydra_args.append("physics=newton_mjwarp")
# The RL task's env module holds its reset ranges, expert settings and EXPERT_TASK.
RL_TASK = args.task
rl_env = importlib.import_module(gym.spec(RL_TASK).kwargs["env_cfg_entry_point"].split(":")[0])
RL_CFG_CLASS = getattr(rl_env, gym.spec(RL_TASK).kwargs["env_cfg_entry_point"].split(":")[1])
TASK = rl_env.EXPERT_TASK
args.task = TASK
args.output = args.output or f"logs/expert_bank/{rl_env.BANK_PREFIX}_bank.pt"
args.visualizer = ["newton_gl"] if args.video else []
sys.argv = [sys.argv[0], *hydra_args]


def configure(env_cfg) -> dict:
    """Apply the RL task's randomization, rate and sizing to the scripted task."""
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import actuator_delay, expert_diversity, valve_dynamics
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp as rl_mdp

    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

    rl = RL_CFG_CLASS()
    profile = rl.rl_robot
    env_cfg.rl_robot = profile
    env_cfg.actions.gripper_action = rl.actions.gripper_action.copy()
    # The RL terms used to score the expert find the valve here.
    env_cfg.valve_task = rl.valve_task
    valve_name = rl.valve_task.asset_name
    # apply_setup rewrites the scene TCP frame with the setup's body offset,
    # which is rotated 90 deg from the RL task's TCP. The scripted command
    # plans with its own offset, so restoring the RL frame changes nothing the
    # expert does; it makes the recorded TCP poses and the hold check use the
    # RL task's convention.
    env_cfg.scene.ee_frame = rl.scene.ee_frame
    # The RL task's force-limited gripper and stiff gripper-valve contacts.
    # Its arm command latency is for the policy; the expert records without
    # (max_delay 0: no delay buffer).
    robot = rl.scene.robot
    if not profile.implicit:
        robot = actuator_delay.delayed_actuators(robot, profile.actuator_groups, args.arm_delay_s, rl.sim.dt)
    elif args.arm_delay_s:
        raise ValueError("FR3 expert collection currently requires --arm_delay_s 0; train with target_delay instead")
    if args.arm_gain_scale != 1.0:
        actuators = dict(robot.actuators)
        for name in profile.actuator_groups:
            a = actuators[name]
            actuators[name] = a.replace(stiffness=a.stiffness * args.arm_gain_scale, damping=a.damping * args.arm_gain_scale)
        robot = robot.replace(actuators=actuators)
    env_cfg.scene.robot = robot
    if args.arm_delay_s > 0.0:
        env_cfg.events.arm_delay = actuator_delay.randomize_actuator_delay_cfg((args.arm_delay_s, args.arm_delay_s), profile.actuator_groups)
    env_cfg.events.gripper_valve_contacts = rl.events.gripper_valve_contacts
    env_cfg.events.quiet_solver = rl.events.quiet_solver
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.dt = rl.sim.dt
    env_cfg.sim.render_interval = rl.decimation
    env_cfg.decimation = rl.decimation
    if args.episode_length_s is None:
        args.episode_length_s = rl.episode_length_s + 1.0
    env_cfg.episode_length_s = args.episode_length_s
    physics = env_cfg.sim.physics
    if args.cuda_graph:
        physics.use_cuda_graph = True
    if hasattr(physics, "solver_cfg"):
        # nconmax is per world; the scripted task's 4000 only fits a few envs.
        physics.solver_cfg.nconmax = rl.sim.physics.newton_mjwarp.solver_cfg.nconmax
    # Same reset distribution as the RL task.
    ranges = {
        "valve_pose": dict(rl_env.VALVE_POSE_RANGE),
        "valve_angle": rl_env.VALVE_ANGLE_RANGE,
        "arm": rl_env.ARM_RANGE,
        "arm_postures": dict(rl_env.ARM_POSTURES),
        "arm_flip_prob": 0.0 if args.no_wrist_flip else rl_env.ARM_FLIP_PROB,
        "stuck_prob": 0.0 if args.no_stuck else rl_env.STUCK_PROB,
        "stuck_breakaway": rl_env.STUCK_BREAKAWAY_RANGE,
        "valve_dynamics": dict(rl_env.VALVE_DYNAMICS_RANGES),
        "expert_diversity": dict(rl_env.EXPERT_DIVERSITY_RANGES),
    }
    if args.nominal_valve:
        ranges["valve_dynamics"] = {
            "friction": (0.05, 0.05), "damping": (0.0, 0.0), "spring": (0.0, 0.0), "breakaway": (0.0, 0.0),
            "armature": (0.005, 0.005),
        }
    if args.nominal_expert:
        ranges["expert_diversity"] = {
            "turn_rate": (0.3, 0.3), "reach_scale": (1.0, 1.0), "grasp_shift": (0.0, 0.0), "grasp_roll": (0.0, 0.0),
        }
    if args.nominal_arm:
        ranges["arm_postures"] = {"home": rl_env.ARM_POSTURES["home"]}
    if args.standard:
        ranges["valve_pose"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        ranges["valve_angle"] = (0.0, 0.0)
        ranges["arm_postures"] = {"home": rl_env.ARM_POSTURES["home"]}
        ranges["arm"] = (0.0, 0.0)
    if args.nominal_pose:
        ranges["valve_pose"] = {"x": (-0.03, 0.03), "y": (-0.04, 0.04), "z": (-0.03, 0.03), "yaw": (-0.1, 0.1)}
    # The scripted tasks name the valve's events differently (reset_valve_root,
    # the small mechanisms' reset_object_root, the circuit breaker's
    # reset_breaker_root), and the small mechanisms do not reset the joint:
    # find them by what they do to the valve.
    valve_events = {
        name: term
        for name, term in vars(env_cfg.events).items()
        if isinstance(term, EventTermCfg)
        and getattr(term.params.get("asset_cfg"), "name", None) == valve_name
    }

    def valve_event(func_name: str) -> str | None:
        return next((name for name, term in valve_events.items() if term.func.__name__ == func_name), None)

    root_reset = valve_events[valve_event("reset_root_state_uniform")]
    root_reset.params["pose_range"] = dict(ranges["valve_pose"])
    joint_reset = valve_event("reset_joints_by_offset")
    if joint_reset is not None and joint_reset != "reset_valve_joint":
        env_cfg.events.reset_valve_joint = valve_events[joint_reset]
        setattr(env_cfg.events, joint_reset, None)
    if getattr(env_cfg.events, "reset_valve_joint", None) is None:
        from isaaclab.envs import mdp as base_mdp

        env_cfg.events.reset_valve_joint = EventTermCfg(
            func=base_mdp.reset_joints_by_offset, mode="reset", params={"velocity_range": (0.0, 0.0)}
        )
    env_cfg.events.reset_valve_joint.params["position_range"] = ranges["valve_angle"]
    # Only the turned joint (the small valve's reset also lists its stem's prismatic joint).
    env_cfg.events.reset_valve_joint.params["asset_cfg"] = SceneEntityCfg(
        valve_name, joint_names=[rl.valve_task.joint_name]
    )
    if rl.valve_task.coupled_joint_name is not None:
        # A nut's travel follows its angle (the thread's mimic constraint).
        env_cfg.events.sync_coupled_joint = EventTermCfg(func=rl_mdp.sync_coupled_joint, mode="reset")
    env_cfg.events.reset_arm = EventTermCfg(
        func=rl_mdp.reset_joints_from_postures,
        mode="reset",
        params={
            "postures": dict(ranges["arm_postures"]),
            "position_range": ranges["arm"],
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(profile.arm_joint_names), preserve_order=True),
            "flip_prob": ranges["arm_flip_prob"],
            "flip_joints": rl_env.ARM_WRIST_FLIP,
        },
    )
    # The RL task's per-episode valve dynamics replace the scripted task's
    # fixed-per-env ones; the trajectory records them for the RL reset. After
    # the valve joint reset: a stuck valve reseats the lever.
    for name, term in valve_events.items():
        if term.func.__name__ in ("randomize_actuator_gains", "randomize_joint_parameters"):
            setattr(env_cfg.events, name, None)
    env_cfg.events.valve_dynamics = valve_dynamics.randomize_valve_dynamics_cfg(
        ranges["valve_dynamics"], stuck_prob=ranges["stuck_prob"], stuck_breakaway=ranges["stuck_breakaway"]
    )
    env_cfg.actions.valve_load = valve_dynamics.ValveLoadActionCfg(asset_name=valve_name)
    # Applied last, after the valve pose and angle it offsets the grasp from.
    # The reach plans to the IK branch nearest the start joints, so a
    # wrist-flipped start keeps its flipped wrist.
    env_cfg.events.expert_diversity = expert_diversity.sample_expert_diversity_cfg(
        ranges["expert_diversity"],
        wrist_branch_nearest=not args.no_wrist_flip and rl_env.ARM_WRIST_FLIP is not None,
        grasp_axes=rl_env.EXPERT_GRASP_AXES,
        overshoot=rl_env.EXPERT_OVERSHOOT,
    )
    # Fixed-length synchronized waves: no early termination, no HDF5 recorder.
    for name in ("success", "command_done"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)
    env_cfg.recorders = RecorderManagerBaseCfg()
    if args.video:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

        # Side view of the valve and arm, in env-0 coordinates.
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.55, 1.05, 1.05)
        env_cfg.viewer.lookat = (0.9, 0.0, 0.8)
        horizon = int(math.ceil(args.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)))
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                output_dir=os.path.abspath(args.video), video_length=horizon * args.max_waves, video_interval=0
            )
        ]
    # The RL task always opens the valve. The scripted command picks open or
    # close at random where both are far enough away (open_task_prob), which
    # sent episodes reset part-open (past valve_min_delta_rad) to closed.
    env_cfg.commands.pose_command.open_task_prob = 1.0
    batch = args.plan_batch_size or args.num_envs
    # The retreat after the (last) turn or push backs straight off to the
    # approach point, keeping the hand's orientation: reorienting to the
    # approach frame spun the hand up to 180 degrees next to the lever.
    commands = env_cfg.commands.pose_command.commands
    rotate = max(
        i for i, c in enumerate(commands) if "Rotate" in type(c).__name__ or getattr(c, "phase", None) == "actuate"
    )
    for command in commands[rotate + 1 :]:
        if hasattr(command, "hold_current_orientation"):
            command.hold_current_orientation = True
    for command in env_cfg.commands.pose_command.commands:
        if rl_env.EXPERT_GRIP_S is not None and getattr(command, "phase", None) == "grip":
            command.duration_s = rl_env.EXPERT_GRIP_S
        if hasattr(command, "plan_batch_size"):
            command.plan_batch_size = batch
        if hasattr(command, "wrist_flip_joints") and not args.no_wrist_flip and rl_env.ARM_WRIST_FLIP is not None:
            command.wrist_flip_joints = expert_diversity.WRIST_FLIP_JOINTS
    # Task-specific changes to the expert, e.g. skipping a regrasp once the part is done.
    if hasattr(rl_env, "configure_expert"):
        rl_env.configure_expert(env_cfg)
    return ranges


def main() -> None:
    from isaaclab_hiveboard_rl.imitation.expert import ScriptedExpert
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp as rl_mdp
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.expert_diversity import EXPERT_DIVERSITY
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.valve_dynamics import VALVE_DYNAMICS

    from isaaclab_hiveboard.mdp.commands.sequential_pose_command import PHASES

    HOLD, SUCCESS_TOLERANCE_RAD = rl_env.HOLD, rl_env.SUCCESS_TOLERANCE_RAD

    env_cfg, _ = resolve_task_config(TASK, "")
    _apply_command_setup(env_cfg, args)
    ranges = configure(env_cfg)

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        dev, n_envs = env.device, env.num_envs
        robot, valve = env.scene["robot"], rl_mdp.valve_asset(env)
        arm_ids = robot.find_joints(env_cfg.actions.arm_action.joint_names, preserve_order=True)[0]
        grip_id = robot.find_joints(env_cfg.actions.gripper_action.joint_names)[0][0]
        # All hand joints, so episodes can start mid-grasp (reset_from_expert_bank).
        hand_ids, hand_names = robot.find_joints(list(env_cfg.rl_robot.hand_joint_names), preserve_order=True)
        command = env.command_manager.get_term("pose_command")
        arm_ids_wrist = list(env_cfg.actions.arm_action.joint_names).index(env_cfg.rl_robot.diagnostic_joint)
        horizon = int(env.max_episode_length)
        kept: dict[str, list[torch.Tensor]] = {}
        # Every episode's sampled variation and outcome, for the success breakdown.
        sampled: dict[str, list[torch.Tensor]] = {
            "valve_dynamics": [], "expert_diversity": [], "arm_q0": [], "turn_wrist": [], "valve_quat": [], "ok": [],
            "progress": [], "grasp_in_frame": [],
        }
        stats = {"waves": 0, "episodes": 0, "successes": 0, "fallback": 0, "kept": 0, "wall_s": 0.0}

        for wave in range(args.max_waves):
            t0 = time.perf_counter()
            env.reset()
            expert = ScriptedExpert(env)
            init = {
                "arm_q0": robot.data.joint_pos.torch[:, arm_ids].clone(),
                "valve_angle0": rl_mdp.valve_angle(env).clone(),
                "valve_pose_env": torch.cat(
                    (valve.data.root_pos_w.torch - env.scene.env_origins, valve.data.root_quat_w.torch), dim=-1
                ).clone(),
                "valve_dynamics": env.valve_dynamics_term.params.clone(),
                "expert_diversity": env.expert_diversity_term.values.clone(),
            }
            steps = {
                k: []
                for k in (
                    "arm_q",
                    "arm_target",
                    "gripper_cmd",
                    "gripper_q",
                    "valve_angle",
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
            best_progress = torch.zeros(n_envs, device=dev)
            min_grasp_err = torch.full((n_envs, 2), float("inf"), device=dev)
            t_open = torch.full((n_envs,), float("nan"), device=dev)
            fallback = torch.zeros(n_envs, dtype=torch.bool, device=dev)
            # TCP pose in the grasp frame (ValveTaskCfg.grasp_frame) when the turn
            # starts: what ValveTaskCfg.grasp_offset_pos/quat should hold.
            grasp_in_frame = torch.full((n_envs, 7), float("nan"), device=dev)
            actuate = PHASES.index("actuate")
            # Valve knocked before the turn: peak |rate| [rad/s] and the phase it happened in.
            pre_turn_rate = torch.zeros(n_envs, device=dev)
            pre_turn_phase = torch.full((n_envs,), -1, dtype=torch.long, device=dev)
            pre_turn_step = torch.zeros(n_envs, dtype=torch.long, device=dev)
            with torch.inference_mode():
                for t in range(horizon):
                    if t == 3 and (args.reg_bias_pos > 0.0 or args.reg_bias_rot > 0.0):
                        # Registration error: the expert has planned its reach, grasp and
                        # turn on the valve as it was; the real one sits a little off.
                        from isaaclab.utils import math as math_utils

                        pose = torch.cat((valve.data.root_pos_w.torch, valve.data.root_quat_w.torch), dim=-1).clone()
                        pose[:, :3] += (torch.rand(n_envs, 3, device=dev) * 2 - 1) * args.reg_bias_pos
                        euler = (torch.rand(n_envs, 3, device=dev) * 2 - 1) * args.reg_bias_rot
                        delta = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
                        pose[:, 3:] = math_utils.quat_mul(delta, pose[:, 3:])
                        valve.write_root_pose_to_sim_index(root_pose=pose)
                        valve.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n_envs, 6, device=dev))
                    if t == horizon - 1:
                        # At the end (before the last step, which resets) the valve must
                        # still be open, with the arm let go and backed off.
                        clear = rl_mdp.released_and_clear(env, SUCCESS_TOLERANCE_RAD)
                        end_dist, _ = rl_mdp.tcp_grasp_error(env)
                        print(
                            f"[BANK] wave {wave} end state: valve open "
                            f"{rl_mdp.valve_open_success(env, SUCCESS_TOLERANCE_RAD).float().mean():.1%}, "
                            f"no pad force {(rl_mdp.pad_valve_force(env).max(dim=-1).values < 1.0).float().mean():.1%}, "
                            f"TCP-grasp distance p10/p50 {torch.quantile(end_dist, 0.1):.3f}/{end_dist.median():.3f} m",
                            flush=True,
                        )
                    action = expert.compute()
                    starts = (command.phase == actuate) & grasp_in_frame[:, 0].isnan()
                    if torch.any(starts):
                        from isaaclab.utils import math as math_utils

                        rel = math_utils.subtract_frame_transforms(*rl_mdp._lever_frame_w(env), *rl_mdp.tcp_w(env))
                        grasp_in_frame[starts] = torch.cat(rel, dim=-1)[starts]
                    steps["arm_q"].append(robot.data.joint_pos.torch[:, arm_ids].clone())
                    steps["arm_target"].append(action[:, :-1].clone())
                    steps["gripper_cmd"].append(action[:, -1].clone())
                    steps["gripper_q"].append(robot.data.joint_pos.torch[:, grip_id].clone())
                    steps["valve_angle"].append(rl_mdp.valve_angle(env).clone())
                    steps["tcp_pose"].append(rl_mdp.tcp_pose_b(env).clone())
                    steps["phase"].append(command._current_command_idx.clone())
                    steps["phase_id"].append(command.phase.clone())
                    # Valve-filtered contact force on each finger pad [N].
                    steps["pad_force"].append(rl_mdp.pad_valve_force(env).clone())
                    steps["gripper_joint_pos"].append(robot.data.joint_pos.torch[:, hand_ids].clone())
                    env.step(action)
                    rate = rl_mdp.valve_rate(env).abs()
                    knocked = (command.phase >= 0) & (command.phase < actuate) & (rate > pre_turn_rate)
                    pre_turn_rate[knocked] = rate[knocked]
                    pre_turn_phase[knocked] = command.phase[knocked]
                    pre_turn_step[knocked] = t
                    held = rl_mdp.lever_held(env, HOLD["dist_threshold"], HOLD["ang_threshold"])
                    is_open = rl_mdp.valve_open_success(env, SUCCESS_TOLERANCE_RAD)
                    ever_held |= held
                    opened_any |= is_open
                    best_progress = torch.maximum(best_progress, rl_mdp.valve_progress(env))
                    dist, ang = rl_mdp.tcp_grasp_error(env)
                    min_grasp_err = torch.minimum(min_grasp_err, torch.stack((dist, ang), dim=-1))
                    # A pushed lever runs ahead of the hand: touched before is enough.
                    now_open = is_open & rl_mdp.opening_credited(env, held, ever_held) & ~opened_held
                    t_open[now_open] = (t + 1) * env.step_dt
                    opened_held |= now_open
                    fallback |= command.expert_fallback()
            wave_s = time.perf_counter() - t0
            # Keep only trajectories the expert ran as planned: a cuRobo
            # fallback (e.g. a chain that could not be planned as one motion)
            # is discarded even when it opened the valve.
            ok = opened_held & ~fallback & clear
            # The RL task's latest acceptable open time, e.g. dropping a push that stalled for seconds under a
            # lever creeping to its stop.
            max_open_s = getattr(rl_env, "EXPERT_MAX_OPEN_S", None)
            slow = torch.zeros_like(ok) if max_open_s is None else ok & (t_open > max_open_s)
            ok &= ~slow
            stats["slow"] = stats.get("slow", 0) + int(slow.sum())
            stats["not_clear"] = stats.get("not_clear", 0) + int((opened_held & ~fallback & ~clear).sum())
            stats["waves"] += 1
            stats["episodes"] += n_envs
            stats["successes"] += int(opened_held.sum())
            stats["fallback"] += int((opened_held & fallback).sum())
            stats["kept"] += int(ok.sum())
            stats["wall_s"] += wave_s
            if args.dump_all:
                dump = {key: torch.stack(values, dim=1).cpu() for key, values in steps.items()}
                dump.update({key: value.cpu() for key, value in init.items()})
                dump.update(ok=ok.cpu(), opened_held=opened_held.cpu(), clear=clear.cpu(), t_open=t_open.cpu())
                torch.save(dump, args.dump_all.replace(".pt", f"_wave{wave}.pt"))
            for key, values in steps.items():
                kept.setdefault(key, []).append(torch.stack(values, dim=1)[ok].cpu())
            for key, value in init.items():
                kept.setdefault(key, []).append(value[ok].cpu())
            sampled["valve_dynamics"].append(init["valve_dynamics"].cpu())
            sampled["expert_diversity"].append(init["expert_diversity"].cpu())
            sampled["arm_q0"].append(init["arm_q0"].cpu())
            sampled["valve_quat"].append(init["valve_pose_env"][:, 3:].cpu())
            # Wrist flexion when the turn starts (NaN if it never did): its
            # sign is the IK branch the expert ended up on.
            arm_q, phase = torch.stack(steps["arm_q"], dim=1), torch.stack(steps["phase_id"], dim=1)
            turning = phase == PHASES.index("actuate")
            first = turning.float().argmax(dim=1)
            wrist = arm_q[torch.arange(n_envs, device=dev), first, arm_ids_wrist]
            sampled["turn_wrist"].append(torch.where(turning.any(dim=1), wrist, torch.nan).cpu())
            sampled["ok"].append(ok.cpu())
            sampled["progress"].append(best_progress.cpu())
            sampled["grasp_in_frame"].append(grasp_in_frame.cpu())
            kept.setdefault("t_open", []).append(t_open[ok].cpu())
            kept.setdefault("grasp_in_frame", []).append(grasp_in_frame[ok].cpu())
            kept.setdefault("expert_fallback", []).append(fallback[ok].cpu())
            total = stats["kept"]
            moved = pre_turn_rate > 1.0
            by_phase = ", ".join(
                f"{PHASES[p]} {int((moved & (pre_turn_phase == p)).sum())}" for p in range(actuate) if (moved & (pre_turn_phase == p)).any()
            )
            print(
                f"[BANK] wave {wave} valve moved > 1 rad/s before the turn in {int(moved.sum())}/{n_envs} envs"
                f" ({by_phase or 'none'}), peak {pre_turn_rate.max():.1f} rad/s; at step"
                f" {pre_turn_step[moved].tolist()[:16]}",
                flush=True,
            )
            print(
                f"[BANK] wave {wave} diagnostics: ever held {int(ever_held.sum())}/{n_envs}, "
                f"opened by any means {int(opened_any.sum())}/{n_envs}, "
                f"mean best progress {best_progress.mean():.3f}, "
                f"median closest grasp error {min_grasp_err[:, 0].median() * 1000:.1f} mm / "
                f"{min_grasp_err[:, 1].median():.3f} rad",
                flush=True,
            )
            print(
                f"[BANK] wave {wave}: {int(opened_held.sum())}/{n_envs} opened while held, "
                f"{int((opened_held & fallback).sum())} discarded for a fallback, "
                f"{int((opened_held & ~fallback & ~clear).sum())} not released and clear at the end, "
                f"{int(slow.sum())} opened too late, "
                f"in {wave_s:.1f}s; kept {total}",
                flush=True,
            )
            complete = opened_held & clear
            stats["complete"] = stats.get("complete", 0) + int(complete.sum())
            stats["opened_held"] = stats.get("opened_held", 0) + int(opened_held.sum())
            stats["opened_any"] = stats.get("opened_any", 0) + int(opened_any.sum())
            if total >= args.num_trajectories:
                break

        if args.eval_only:
            n = stats["episodes"]
            print(
                f"[BASELINE] episodes {n} | complete {stats['complete'] / n:.1%} | opened held "
                f"{stats['opened_held'] / n:.1%} | opened any {stats['opened_any'] / n:.1%} | cuRobo fallback used "
                f"{stats['fallback'] / n:.1%} (gain x{args.arm_gain_scale}, delay {args.arm_delay_s}s, "
                f"registration {args.reg_bias_pos} m / {args.reg_bias_rot} rad)",
                flush=True,
            )
            env.close()
            return
        if stats["kept"] == 0:
            raise RuntimeError("No successful expert trajectories; inspect grasp/contact diagnostics before training")
        bank = {key: torch.cat(values)[: args.num_trajectories] for key, values in kept.items()}
        bank["meta"] = {
            "robot": env_cfg.rl_robot.name,
            "diagnostic_joint": env_cfg.rl_robot.diagnostic_joint,
            "physics_dt": env.physics_dt,
            "seed": args.seed,
            "task": TASK,
            "rl_task": RL_TASK,
            "valve_task": env_cfg.valve_task.to_dict(),
            "dt": env.step_dt,
            "horizon": horizon,
            "arm_joint_names": list(env_cfg.actions.arm_action.joint_names),
            "gripper_joint_names": list(hand_names),
            # phase_id indexes these; -1 is an untagged segment, len(phases) the finished sequence.
            "phases": list(PHASES),
            "valve_dynamics": list(VALVE_DYNAMICS),
            "expert_diversity": list(EXPERT_DIVERSITY),
            "ranges": ranges,
            "success_rate": stats["successes"] / max(stats["episodes"], 1),
            "stats": stats,
            "hold": dict(HOLD),
            "success_tolerance_rad": SUCCESS_TOLERANCE_RAD,
        }
        bank["meta"]["sampled"] = {key: torch.cat(values) for key, values in sampled.items()}
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        torch.save(bank, args.output)
        rate = stats["successes"] / max(stats["episodes"], 1)
        per_traj = stats["wall_s"] / max(stats["kept"], 1)
        print(
            f"[BANK] saved {len(bank['arm_q'])} trajectories to {args.output}\n"
            f"[BANK] expert success {stats['successes']}/{stats['episodes']} = {rate:.1%}, "
            f"{stats['fallback']} successes discarded for using a cuRobo fallback\n"
            f"[BANK] wall time {stats['wall_s']:.0f}s over {stats['waves']} waves = "
            f"{per_traj:.2f}s per kept trajectory",
            flush=True,
        )
        if len(bank["t_open"]):
            t = bank["t_open"]
            print(f"[BANK] time to open {t.mean():.2f} +/- {t.std():.2f}s", flush=True)
        # Over every episode that started the turn, kept or not.
        g = torch.cat(sampled["grasp_in_frame"])
        g = g[~g[:, 0].isnan()]
        if len(g):
            # One sign per quaternion before taking the median.
            g[:, 3:] *= torch.where(g[:, 6:7] < 0, -1.0, 1.0)
            print(
                f"[BANK] TCP in grasp frame '{env_cfg.valve_task.grasp_frame}' when the turn starts, median: "
                f"pos {[round(v, 4) for v in g[:, :3].median(dim=0).values.tolist()]}, "
                f"xyzw {[round(v, 4) for v in g[:, 3:].median(dim=0).values.tolist()]} "
                f"(configured {env_cfg.valve_task.grasp_offset_pos}, {env_cfg.valve_task.grasp_offset_quat})",
                flush=True,
            )
        # Kept fraction in the lower and upper third of each sampled range: a
        # low upper third means the expert fails there and the bank thins out.
        ok_all = torch.cat(sampled["ok"]).float()
        for group, names in (("valve_dynamics", VALVE_DYNAMICS), ("expert_diversity", EXPERT_DIVERSITY)):
            values = torch.cat(sampled[group])
            for i, name in enumerate(names):
                if name not in ranges[group]:
                    continue
                lo, hi = ranges[group][name]
                frac = (values[:, i] - lo) / max(hi - lo, 1e-9)
                low, high = ok_all[frac < 1 / 3].mean(), ok_all[frac > 2 / 3].mean()
                print(f"[BANK] kept by {name:12s}: lower third {low:.1%}, upper third {high:.1%}", flush=True)
        # Arm start posture (nearest one, ignoring the wrist flip), wrist
        # branch and stuck valves.
        arm_q0 = torch.cat(sampled["arm_q0"])
        flipped = torch.zeros(len(arm_q0), dtype=torch.bool)
        unflipped = arm_q0.clone()
        if rl_env.ARM_WRIST_FLIP is not None:
            roll1, pitch, roll2 = rl_env.ARM_WRIST_FLIP
            flipped = arm_q0[:, roll1].abs() > math.pi / 2
            unflipped[flipped, roll1] -= torch.sign(unflipped[flipped, roll1]) * math.pi
            unflipped[flipped, pitch] *= -1.0
            unflipped[flipped, roll2] += math.pi
        postures = torch.tensor(list(ranges["arm_postures"].values()))
        nearest = torch.cdist(unflipped, postures).argmin(dim=1)
        for i, name in enumerate(ranges["arm_postures"]):
            sel = nearest == i
            print(f"[BANK] kept from posture {name:10s}: {ok_all[sel].mean():.1%} of {int(sel.sum())}", flush=True)
        if rl_env.ARM_WRIST_FLIP is not None:
            wrist = torch.cat(sampled["turn_wrist"])
            for label, sel in (("usual start", ~flipped), ("flipped start", flipped)):
                turned = sel & ~wrist.isnan()
                on_flip = (wrist[turned] > 0).float().mean() if turned.any() else torch.tensor(float("nan"))
                print(
                    f"[BANK] {label:13s}: kept {ok_all[sel].mean():.1%} of {int(sel.sum())}, "
                    f"turned with wrist flexion > 0 in {on_flip:.1%}",
                    flush=True,
                )
        # Valve orientation offset (roll about the stem, pitch, yaw) from the nominal one.
        from isaaclab.utils import math as math_utils

        nominal = torch.tensor(getattr(env_cfg.scene, env_cfg.valve_task.asset_name).init_state.rot).expand(len(ok_all), 4)
        delta = math_utils.quat_mul(math_utils.quat_inv(nominal), torch.cat(sampled["valve_quat"]))
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(delta, wrap_to_2pi=False)
        for lo in range(-180, 180, 45):
            sel = (torch.rad2deg(roll) >= lo) & (torch.rad2deg(roll) < lo + 45)
            print(f"[BANK] kept with roll {lo:+4d}..{lo + 45:+4d} deg: {ok_all[sel].mean():.1%} of {int(sel.sum())}", flush=True)
        for name, angle in (("pitch", pitch), ("yaw", yaw)):
            big = angle.abs() > math.radians(10.0)
            print(f"[BANK] kept with |{name}| > 10 deg: {ok_all[big].mean():.1%} of {int(big.sum())}", flush=True)
        breakaway = torch.cat(sampled["valve_dynamics"])[:, VALVE_DYNAMICS.index("breakaway")]
        stuck = breakaway > ranges["valve_dynamics"]["breakaway"][1]
        for label, sel in (("stuck closed", stuck), ("not stuck", ~stuck)):
            print(f"[BANK] kept {label:12s}: {ok_all[sel].mean():.1%} of {int(sel.sum())}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
