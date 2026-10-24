import argparse
import math
import os
import sys
import time

import gymnasium as gym
import torch
from isaaclab_hiveboard_rl import env as mdp
from isaaclab_hiveboard_rl import specs

import isaaclab.utils.math as math_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    PHASES,
    WRIST_BRANCH_NEAREST,
    _CuroboPlannedGoToFrameHandler,
    _GoToFrameHandler,
    _RotateFrameHandler,
)
from isaaclab_hiveboard.utils.command_setup import apply_setup, load_setup, resolve_setup_path

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-HiveBoard-Anymal-BallValve-RL-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_trajectories", type=int, default=5000)
parser.add_argument("--plan_batch_size", type=int, default=None)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--max_waves", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default=None)
parser.add_argument("--cuda_graph", action="store_true")
parser.add_argument("--nominal_valve", action="store_true", help="no valve randomization")
parser.add_argument("--nominal_expert", action="store_true", help="no expert diversity")
parser.add_argument("--nominal_pose", action="store_true", help="narrow valve placement")
parser.add_argument("--nominal_arm", action="store_true", help="arm starts around home")
parser.add_argument("--standard", action="store_true", help="nominal valve pose, arm at home")
parser.add_argument("--no_wrist_flip", action="store_true")
parser.add_argument("--no_stuck", action="store_true")
parser.add_argument("--eval_only", action="store_true", help="report success, save nothing")
parser.add_argument("--arm_gain_scale", type=float, default=1.0)
parser.add_argument("--arm_delay_s", type=float, default=0.0)
parser.add_argument("--reg_bias_pos", type=float, default=0.0, help="valve shift after planning")
parser.add_argument("--reg_bias_rot", type=float, default=0.0, help="valve tilt after planning")
parser.add_argument("--dump_all", default=None, help="also save every episode here")
parser.add_argument("--setup", default=None)
parser.add_argument("--no_setup", action="store_true")
parser.add_argument("--video", default=None, help="record env 0 to this directory")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
if not any(t.startswith(("physics=", "presets=")) for t in hydra_args):
    hydra_args.append("physics=newton_mjwarp")
args.visualizer = ["newton_gl"] if args.video else []
sys.argv = [sys.argv[0], *hydra_args]

EXPERT_DIVERSITY = ("turn_rate", "reach_scale", "grasp_shift", "grasp_roll", "turn_overshoot")


class sample_expert_diversity(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.values = torch.zeros(env.num_envs, len(EXPERT_DIVERSITY), device=env.device)
        self.rotate = None
        env.expert_diversity_term = self

    def __call__(self, env, env_ids, ranges, wrist_branch_nearest, grasp_axes, overshoot):
        if self.rotate is None:
            handlers = env.command_manager.get_term("pose_command")._command_handlers
            rotates = [i for i, h in enumerate(handlers) if isinstance(h, _RotateFrameHandler)]
            pushes = (
                []
                if rotates
                else [
                    i
                    for i, h in enumerate(handlers)
                    if isinstance(h, _GoToFrameHandler) and h.cfg.phase == "actuate"
                ]
            )
            actuate = rotates or pushes
            self.rotate = [handlers[i] for i in rotates]
            self.push = [handlers[i] for i in pushes]
            self.reach = [h for h in handlers[: actuate[0]] if isinstance(h, _GoToFrameHandler)]
            self.grasp = self.reach[-1]
            self.retreat = [
                h
                for h in handlers[actuate[-1] + 1 :]
                if isinstance(h, _CuroboPlannedGoToFrameHandler)
            ]

            def goal(handler):
                cfg, name = handler._frame.cfg, handler.cfg.target_frame_name
                frame = next(f for f in cfg.target_frames if f.name == name)
                assert frame.prim_path == cfg.target_frames[0].prim_path
                return math_utils.combine_frame_transforms(
                    *(
                        torch.tensor([x], device=env.device)
                        for x in (
                            frame.offset.pos,
                            frame.offset.rot,
                            handler.cfg.target_offset_pos,
                            handler.cfg.target_offset_rot,
                        )
                    )
                )

            grasp = goal(self.grasp)
            self.moved = self.reach[:-1] + self.push
            self.moved_in_grasp = [
                math_utils.subtract_frame_transforms(*grasp, *goal(h)) for h in self.moved
            ]
        ids = mdp.all_ids(env, env_ids)
        n = len(ids)
        values = torch.empty(n, len(EXPERT_DIVERSITY), device=env.device)
        for i, name in enumerate(EXPERT_DIVERSITY[:-1]):
            values[:, i].uniform_(*ranges[name])
        task = env.cfg.valve_task
        friction, _, spring, _, _ = env.valve_dynamics_term.params[ids].unbind(dim=-1)
        torque = friction + spring * abs(task.open_rad - task.closed_rad)
        values[:, -1] = overshoot[0] + overshoot[1] * torque
        self.values[ids] = values
        turn_rate, reach_scale, shift, roll, extra = values.unbind(dim=-1)
        for rotate in self.rotate:
            rotate.speed_scale[ids] = turn_rate / rotate.cfg.angular_velocity
            rotate.angle_extra_rad[ids] = extra
        for push in self.push:
            push.speed_scale[ids] = turn_rate / push.cfg.velocity
        for handler in self.reach:
            handler.speed_scale[ids] = reach_scale
        if wrist_branch_nearest:
            for handler in self.reach + self.retreat:
                assert handler.cfg.wrist_flip_joints is not None
                handler.wrist_branch[ids] = WRIST_BRANCH_NEAREST
                handler.forearm_winding[ids] = 0.0
        shift_axis = torch.tensor(grasp_axes[0], device=env.device).expand(n, 3)
        roll_axis = torch.tensor(grasp_axes[1], device=env.device).expand(n, 3)
        grasp_pos = shift_axis * shift[:, None]
        grasp_quat = math_utils.quat_from_angle_axis(roll, roll_axis)
        self.grasp.env_offset_pos[ids] = grasp_pos
        self.grasp.env_offset_rot[ids] = grasp_quat
        for handler, (k_pos, k_quat) in zip(self.moved, self.moved_in_grasp):
            k_pos, k_quat = k_pos.expand(n, 3), k_quat.expand(n, 4)
            moved = math_utils.combine_frame_transforms(grasp_pos, grasp_quat, k_pos, k_quat)
            handler.env_offset_pos[ids], handler.env_offset_rot[ids] = (
                math_utils.subtract_frame_transforms(k_pos, k_quat, *moved)
            )


rl = gym.spec(args.task).kwargs["env_cfg_entry_point"]()
s = specs.spec(rl.robot, rl.tool)
task = s["expert_task"]
output = args.output or specs.bank_path(rl.robot, rl.tool)
wrist_flip = not args.no_wrist_flip and s["arm_wrist_flip"] is not None
env_cfg, _ = resolve_task_config(task, "")
setup = None if args.no_setup else args.setup or resolve_setup_path(task)
if setup:
    print(f"[INFO] Using saved setup {setup} (pass --no_setup to skip).")
    apply_setup(env_cfg, load_setup(setup), task=task)
profile = rl.rl_robot
env_cfg.rl_robot = profile
env_cfg.actions.gripper_action = rl.actions.gripper_action.copy()
env_cfg.valve_task = rl.valve_task
valve_name = rl.valve_task.asset_name
env_cfg.scene.ee_frame = rl.scene.ee_frame
robot = rl.scene.robot
if profile.implicit:
    assert not args.arm_delay_s
else:
    robot = mdp.delayed_actuators(robot, profile.actuator_groups, args.arm_delay_s, rl.sim.dt)
if args.arm_gain_scale != 1.0:
    actuators = dict(robot.actuators)
    for name in profile.actuator_groups:
        a = actuators[name]
        actuators[name] = a.replace(
            stiffness=a.stiffness * args.arm_gain_scale, damping=a.damping * args.arm_gain_scale
        )
    robot = robot.replace(actuators=actuators)
env_cfg.scene.robot = robot
if args.arm_delay_s > 0.0:
    env_cfg.events.arm_delay = mdp.randomize_actuator_delay_cfg(
        (args.arm_delay_s, args.arm_delay_s), profile.actuator_groups
    )
env_cfg.events.gripper_valve_contacts = rl.events.gripper_valve_contacts
env_cfg.events.quiet_solver = rl.events.quiet_solver
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env_cfg.sim.dt = rl.sim.dt
env_cfg.sim.render_interval = rl.decimation
env_cfg.decimation = rl.decimation
env_cfg.episode_length_s = args.episode_length_s or rl.episode_length_s + 1.0
physics = env_cfg.sim.physics
if args.cuda_graph:
    physics.use_cuda_graph = True
if hasattr(physics, "solver_cfg"):
    physics.solver_cfg.nconmax = rl.sim.physics.newton_mjwarp.solver_cfg.nconmax
ranges = {
    "valve_pose": dict(s["valve_pose_range"]),
    "valve_angle": s["valve_angle_range"],
    "arm": s["arm_range"],
    "arm_postures": dict(s["arm_postures"]),
    "arm_flip_prob": 0.0 if args.no_wrist_flip else s["arm_flip_prob"],
    "stuck_prob": 0.0 if args.no_stuck else s["stuck_prob"],
    "stuck_breakaway": s["stuck_breakaway_range"],
    "valve_dynamics": dict(s["valve_dynamics_ranges"]),
    "expert_diversity": dict(s["expert_diversity_ranges"]),
}
if args.nominal_valve:
    ranges["valve_dynamics"] = {
        "friction": (0.05, 0.05),
        "damping": (0.0, 0.0),
        "spring": (0.0, 0.0),
        "breakaway": (0.0, 0.0),
        "armature": (0.005, 0.005),
    }
if args.nominal_expert:
    ranges["expert_diversity"] = {
        "turn_rate": (0.3, 0.3),
        "reach_scale": (1.0, 1.0),
        "grasp_shift": (0.0, 0.0),
        "grasp_roll": (0.0, 0.0),
    }
if args.nominal_arm:
    ranges["arm_postures"] = {"home": s["arm_postures"]["home"]}
if args.standard:
    ranges["valve_pose"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    ranges["valve_angle"] = (0.0, 0.0)
    ranges["arm_postures"] = {"home": s["arm_postures"]["home"]}
    ranges["arm"] = (0.0, 0.0)
if args.nominal_pose:
    ranges["valve_pose"] = {
        "x": (-0.03, 0.03),
        "y": (-0.04, 0.04),
        "z": (-0.03, 0.03),
        "yaw": (-0.1, 0.1),
    }
valve_events = {
    name: term
    for name, term in vars(env_cfg.events).items()
    if isinstance(term, EventTermCfg)
    and getattr(term.params.get("asset_cfg"), "name", None) == valve_name
}
by_func = {}
for name, term in valve_events.items():
    by_func.setdefault(term.func.__name__, name)
valve_events[by_func["reset_root_state_uniform"]].params["pose_range"] = dict(ranges["valve_pose"])
joint_reset = by_func.get("reset_joints_by_offset")
if joint_reset is not None and joint_reset != "reset_valve_joint":
    env_cfg.events.reset_valve_joint = valve_events[joint_reset]
    setattr(env_cfg.events, joint_reset, None)
if getattr(env_cfg.events, "reset_valve_joint", None) is None:
    env_cfg.events.reset_valve_joint = EventTermCfg(
        func=base_mdp.reset_joints_by_offset, mode="reset", params={"velocity_range": (0.0, 0.0)}
    )
env_cfg.events.reset_valve_joint.params["position_range"] = ranges["valve_angle"]
env_cfg.events.reset_valve_joint.params["asset_cfg"] = SceneEntityCfg(
    valve_name, joint_names=[rl.valve_task.joint_name]
)
if rl.valve_task.coupled_joint_name is not None:
    env_cfg.events.sync_coupled_joint = EventTermCfg(func=mdp.sync_coupled_joint, mode="reset")
env_cfg.events.reset_arm = EventTermCfg(
    func=mdp.reset_joints_from_postures,
    mode="reset",
    params={
        "postures": dict(ranges["arm_postures"]),
        "position_range": ranges["arm"],
        "asset_cfg": SceneEntityCfg(
            "robot", joint_names=list(profile.arm_joint_names), preserve_order=True
        ),
        "flip_prob": ranges["arm_flip_prob"],
        "flip_joints": s["arm_wrist_flip"],
    },
)
for name, term in valve_events.items():
    if term.func.__name__ in ("randomize_actuator_gains", "randomize_joint_parameters"):
        setattr(env_cfg.events, name, None)
env_cfg.events.valve_dynamics = mdp.randomize_valve_dynamics_cfg(
    ranges["valve_dynamics"], ranges["stuck_prob"], ranges["stuck_breakaway"]
)
env_cfg.actions.valve_load = mdp.ValveLoadActionCfg(asset_name=valve_name)
env_cfg.events.expert_diversity = EventTermCfg(
    func=sample_expert_diversity,
    mode="reset",
    params={
        "ranges": dict(ranges["expert_diversity"]),
        "wrist_branch_nearest": wrist_flip,
        "grasp_axes": s["expert_grasp_axes"],
        "overshoot": s["expert_overshoot"],
    },
)
for name in ("success", "command_done"):
    if hasattr(env_cfg.terminations, name):
        setattr(env_cfg.terminations, name, None)
env_cfg.recorders = RecorderManagerBaseCfg()
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
env_cfg.commands.pose_command.open_task_prob = 1.0
commands = env_cfg.commands.pose_command.commands
rotate = max(
    i
    for i, c in enumerate(commands)
    if "Rotate" in type(c).__name__ or getattr(c, "phase", None) == "actuate"
)
for command in commands[rotate + 1 :]:
    if hasattr(command, "hold_current_orientation"):
        command.hold_current_orientation = True
for command in commands:
    if s["expert_grip_s"] is not None and getattr(command, "phase", None) == "grip":
        command.duration_s = s["expert_grip_s"]
    if hasattr(command, "plan_batch_size"):
        command.plan_batch_size = args.plan_batch_size or args.num_envs
    if hasattr(command, "wrist_flip_joints") and wrist_flip:
        command.wrist_flip_joints = tuple(profile.arm_joint_names[i] for i in s["arm_wrist_flip"])
if s["configure_expert"] is not None:
    s["configure_expert"](env_cfg)

with launch_simulation(env_cfg, args):
    env = gym.make(task, cfg=env_cfg).unwrapped
    dev, n_envs = env.device, env.num_envs
    robot, valve = env.scene["robot"], mdp.valve_asset(env)
    arm_ids = robot.find_joints(env_cfg.actions.arm_action.joint_names, preserve_order=True)[0]
    grip_id = robot.find_joints(env_cfg.actions.gripper_action.joint_names)[0][0]
    hand_ids, hand_names = robot.find_joints(list(profile.hand_joint_names), preserve_order=True)
    command = env.command_manager.get_term("pose_command")
    wrist_id = list(env_cfg.actions.arm_action.joint_names).index(profile.diagnostic_joint)
    actuate = PHASES.index("actuate")
    hold, tolerance = s["hold"], s["success_tolerance_rad"]
    horizon = int(env.max_episode_length)
    kept = {}
    sampled = {
        k: []
        for k in (
            "valve_dynamics",
            "expert_diversity",
            "arm_q0",
            "turn_wrist",
            "valve_quat",
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
            "valve_angle0": mdp.valve_angle(env).clone(),
            "valve_pose_env": torch.cat(
                (valve.data.root_pos_w.torch - env.scene.env_origins, valve.data.root_quat_w.torch),
                dim=-1,
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
        fallback = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        best_progress = torch.zeros(n_envs, device=dev)
        t_open = torch.full((n_envs,), float("nan"), device=dev)
        grasp_in_frame = torch.full((n_envs, 7), float("nan"), device=dev)
        with torch.inference_mode():
            for t in range(horizon):
                if t == 3 and (args.reg_bias_pos > 0.0 or args.reg_bias_rot > 0.0):
                    pose = torch.cat(
                        (valve.data.root_pos_w.torch, valve.data.root_quat_w.torch), dim=-1
                    ).clone()
                    pose[:, :3] += (torch.rand(n_envs, 3, device=dev) * 2 - 1) * args.reg_bias_pos
                    euler = (torch.rand(n_envs, 3, device=dev) * 2 - 1) * args.reg_bias_rot
                    delta = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
                    pose[:, 3:] = math_utils.quat_mul(delta, pose[:, 3:])
                    valve.write_root_pose_to_sim_index(root_pose=pose)
                    valve.write_root_velocity_to_sim_index(
                        root_velocity=torch.zeros(n_envs, 6, device=dev)
                    )
                if t == horizon - 1:
                    clear = mdp.released_and_clear(env, tolerance)
                action = command.command
                starts = (command.phase == actuate) & grasp_in_frame[:, 0].isnan()
                if torch.any(starts):
                    rel = math_utils.subtract_frame_transforms(
                        *mdp._lever_frame_w(env), *mdp.tcp_w(env)
                    )
                    grasp_in_frame[starts] = torch.cat(rel, dim=-1)[starts]
                steps["arm_q"].append(robot.data.joint_pos.torch[:, arm_ids].clone())
                steps["arm_target"].append(action[:, :-1].clone())
                steps["gripper_cmd"].append(action[:, -1].clone())
                steps["gripper_q"].append(robot.data.joint_pos.torch[:, grip_id].clone())
                steps["valve_angle"].append(mdp.valve_angle(env).clone())
                steps["tcp_pose"].append(mdp.tcp_pose_b(env).clone())
                steps["phase"].append(command._current_command_idx.clone())
                steps["phase_id"].append(command.phase.clone())
                steps["pad_force"].append(mdp.pad_valve_force(env).clone())
                steps["gripper_joint_pos"].append(robot.data.joint_pos.torch[:, hand_ids].clone())
                env.step(action)
                held = mdp.lever_held(env, hold["dist_threshold"], hold["ang_threshold"])
                is_open = mdp.valve_open_success(env, tolerance)
                ever_held |= held
                opened_any |= is_open
                best_progress = torch.maximum(best_progress, mdp.valve_progress(env))
                now_open = is_open & mdp.opening_credited(env, held, ever_held) & ~opened_held
                t_open[now_open] = (t + 1) * env.step_dt
                opened_held |= now_open
                fallback |= command.expert_fallback()
        wave_s = time.perf_counter() - t0
        ok = opened_held & ~fallback & clear
        slow = torch.zeros_like(ok)
        if s["expert_max_open_s"] is not None:
            slow = ok & (t_open > s["expert_max_open_s"])
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
        if args.dump_all:
            dump = {key: torch.stack(values, dim=1).cpu() for key, values in steps.items()}
            dump.update({key: value.cpu() for key, value in init.items()})
            dump.update(
                ok=ok.cpu(), opened_held=opened_held.cpu(), clear=clear.cpu(), t_open=t_open.cpu()
            )
            torch.save(dump, args.dump_all.replace(".pt", f"_wave{wave}.pt"))
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
        sampled["valve_dynamics"].append(init["valve_dynamics"].cpu())
        sampled["expert_diversity"].append(init["expert_diversity"].cpu())
        sampled["arm_q0"].append(init["arm_q0"].cpu())
        sampled["valve_quat"].append(init["valve_pose_env"][:, 3:].cpu())
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
    if args.eval_only:
        print(
            f"[BASELINE] episodes {n} | complete {stats['complete'] / n:.1%} | opened held "
            f"{stats['opened_held'] / n:.1%} | opened any {stats['opened_any'] / n:.1%} | "
            f"cuRobo fallback used {stats['fallback'] / n:.1%} (gain x{args.arm_gain_scale}, "
            f"delay {args.arm_delay_s}s, "
            f"registration {args.reg_bias_pos} m / {args.reg_bias_rot} rad)",
            flush=True,
        )
    else:
        assert stats["kept"] > 0, "no successful expert trajectories"
        bank = {key: torch.cat(values)[: args.num_trajectories] for key, values in kept.items()}
        bank["meta"] = {
            "robot": profile.name,
            "diagnostic_joint": profile.diagnostic_joint,
            "physics_dt": env.physics_dt,
            "seed": args.seed,
            "task": task,
            "rl_task": args.task,
            "valve_task": env_cfg.valve_task.to_dict(),
            "dt": env.step_dt,
            "horizon": horizon,
            "arm_joint_names": list(env_cfg.actions.arm_action.joint_names),
            "gripper_joint_names": list(hand_names),
            "phases": list(PHASES),
            "valve_dynamics": list(mdp.VALVE_DYNAMICS),
            "expert_diversity": list(EXPERT_DIVERSITY),
            "ranges": ranges,
            "success_rate": rate,
            "expert_success": rate,
            "stats": stats,
            "hold": dict(hold),
            "success_tolerance_rad": tolerance,
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
        print(
            f"[BANK] TCP in grasp frame '{env_cfg.valve_task.grasp_frame}' when the turn starts, "
            f"median: pos {[round(v, 4) for v in g[:, :3].median(dim=0).values.tolist()]}, "
            f"xyzw {[round(v, 4) for v in g[:, 3:].median(dim=0).values.tolist()]} "
            f"(configured {env_cfg.valve_task.grasp_offset_pos}, "
            f"{env_cfg.valve_task.grasp_offset_quat})",
            flush=True,
        )
    env.close()
