import argparse
import math
import os
import time

import torch
from isaaclab_hiveboard_rl import env as mdp

import isaaclab.utils.math as math_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    PHASES,
    WRIST_BRANCH_NEAREST,
    MechanismGoalCfg,
    SequentialPoseCommandCfg,
    _CuroboPlannedGoToFrameHandler,
    _GoToFrameHandler,
    _RotateFrameHandler,
)
from isaaclab_hiveboard.mdp.commands.skills import SkillSet

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
        task = env.cfg.mechanism
        friction, _, spring, _, _ = env.dynamics_term.params[ids].unbind(dim=-1)
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


robot = mdp.load(f"robots/{args.robot}")
task = mdp.load(f"tasks/{args.task}")
asset_name = task["asset"]
diversity = {
    "turn_rate": (0.25, 0.5),
    "reach_scale": (0.75, 1.3),
    "grasp_shift": (0.0, 0.0),
    "grasp_roll": (0.0, 0.0),
}
grasp_axes = ((1.0, 0.0, 0.0),) * 2
overshoot = (0.0, 0.0)
max_open_s = math.inf
output = args.output or mdp.bank_path(args.robot, args.task)
env_cfg = mdp.compose(args.robot, args.task, "expert")
profile, mechanism = env_cfg.robot, env_cfg.mechanism
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
env_cfg.actions.arm_action = JointPositionActionCfg(
    asset_name="robot", joint_names=arm, scale=1.0, use_default_offset=False, preserve_order=True
)
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
events = env_cfg.events
if args.arm_delay_s:
    events.arm_delay = EventTermCfg(
        func=mdp.randomize_actuator_delay,
        mode="reset",
        params={"delay_range_s": (args.arm_delay_s,) * 2, "groups": robot["delayed_groups"]},
    )
initial = task.get("initial", {})
pose_range = {
    "x": (-0.05, 0.05),
    "y": (-0.06, 0.06),
    "z": (-0.05, 0.05),
    **{axis: (-math.pi / 12, math.pi / 12) for axis in ("roll", "pitch", "yaw")},
    **initial.get("pose", {}),
}
events.reset_mechanism_root = EventTermCfg(
    func=base_mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "pose_range": {
            k: (max(low, robot["pose_limits"][k][0]), min(high, robot["pose_limits"][k][1]))
            for k, (low, high) in pose_range.items()
        },
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg(mechanism.asset_name),
    },
)
events.reset_mechanism_joint = EventTermCfg(
    func=base_mdp.reset_joints_by_offset,
    mode="reset",
    params={
        "position_range": initial.get("joint_offset", [0.0, 0.0]),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg(mechanism.asset_name, joint_names=[mechanism.joint_name]),
    },
)
events.reset_arm = EventTermCfg(
    func=mdp.reset_joints_from_postures,
    mode="reset",
    params={
        "postures": robot["postures"],
        "position_range": robot["arm_range"],
        "asset_cfg": SceneEntityCfg("robot", joint_names=arm, preserve_order=True),
    },
)
skills = SkillSet(robot_joint_names=arm, **robot["planner"])
planned = {"plan_batch_size": args.num_envs}
goto = {**planned, "wrist_flip_joints": robot["wrist_flip"]}
minimum = robot["min_duration_s"]
if asset_name == "ball_valve":
    method_name = "ball_valve_turn"
    diversity.update(grasp_shift=(-0.015, 0.015), grasp_roll=(-0.25, 0.25))
    grasp_axes = ((0.0, 0.0, -1.0),) * 2
    overshoot = (0.03, 0.05)
    if profile.name == "anymal":
        goto["upright_axis"] = (0.0, 1.0, 0.0)
    commands = [
        skills.approach("approaching", chain_with_next=True, **goto),
        skills.engage(mechanism.grasp_frame, chain_with_next=not minimum["grip"], **goto),
    ]
    if minimum["grip"]:
        commands.append(skills.grip(duration_s=minimum["grip"]))
    commands.append(skills.turn(use_valve_angle=False, **planned))
    if minimum["hold"]:
        commands.append(skills.hold(phase="actuate", duration_s=minimum["hold"]))
    commands.extend(
        [
            skills.release(duration_s=max(0.3, minimum["release"])),
            skills.retreat(
                "approaching",
                distance_threshold=0.05,
                canonicalize_upward=False,
                hold_current_orientation=True,
                **goto,
            ),
        ]
    )
elif asset_name == "small_valve":
    method_name = "small_valve_turn"
    diversity.update(grasp_shift=(-0.005, 0.005), grasp_roll=(-0.12, 0.12))
    overshoot = (0.0, 0.05)
    commands = [
        skills.approach("approaching", chain_with_next=True, **goto),
        skills.engage(mechanism.grasp_frame, **goto),
        skills.grip(duration_s=max(0.3, minimum["grip"])),
        skills.turn(angle_threshold_deg=0.25, **planned),
        skills.release(duration_s=max(0.375, minimum["release"])),
        skills.retreat("approaching", velocity=0.25, hold_current_orientation=True, **goto),
    ]
elif asset_name == "circuit_breaker":
    method_name = "circuit_breaker_push"
    diversity.update(turn_rate=(0.1, 0.2), grasp_shift=(-0.01, 0.01), grasp_roll=(-0.3, 0.3))
    grasp_axes = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    max_open_s = 5.0
    commands = [
        skills.approach("approaching", gripper_open=False, **goto),
        skills.engage(mechanism.grasp_frame, gripper_open=False, **goto),
        skills.move(
            "lever_pivot_above",
            velocity=0.15,
            distance_threshold=0.02,
            done_when_joint=(
                mechanism.asset_name,
                mechanism.joint_name,
                -1.0,
                mechanism.open_rad + 0.035,
            ),
            **goto,
        ),
        skills.hold(phase="actuate", duration_s=max(0.3, minimum["hold"])),
        skills.retreat(
            "lever_pivot_above",
            target_offset_pos=(-0.07, 0.0, 0.0),
            gripper_open=False,
            velocity=0.15,
            distance_threshold=0.02,
            **goto,
        ),
        skills.retreat(
            "approaching", gripper_open=False, velocity=0.25, hold_current_orientation=True, **goto
        ),
    ]
elif asset_name == "m30_thread":
    method_name = "m30_thread_regrasp"
    diversity["turn_rate"] = (0.5, 0.75)
    turn = {
        **planned,
        "angular_velocity": 0.6,
        "angle_threshold_deg": 2.0,
        "max_ee_rotation_deg": 120.0,
        "screw_pitch_m_per_revolution": mechanism.screw.pitch_m_per_revolution,
    }
    commands = [skills.approach("approaching", **goto)]
    for cycle in range(7):
        commands.extend(
            [
                skills.engage(
                    mechanism.grasp_frame,
                    velocity=0.1,
                    distance_threshold=0.01,
                    until=MechanismGoalCfg(goal_tolerance=0.1),
                    **goto,
                ),
                skills.grip(
                    duration_s=max(0.5, minimum["grip"]), until=MechanismGoalCfg(goal_tolerance=0.1)
                ),
                skills.turn(until=MechanismGoalCfg(goal_tolerance=0.1), **turn),
                skills.release(duration_s=max(0.4, minimum["release"])),
                skills.retreat(
                    "clear",
                    velocity=0.1,
                    distance_threshold=0.01,
                    hold_current_orientation=cycle == 6,
                    **goto,
                ),
            ]
        )
    commands.append(skills.retreat("approaching", hold_current_orientation=True, **goto))
else:
    raise ValueError(f"No expert method for {asset_name}")
events.expert_diversity = EventTermCfg(
    func=sample_expert_diversity,
    mode="reset",
    params={
        "ranges": diversity,
        "wrist_branch_nearest": bool(robot["wrist_flip"]),
        "grasp_axes": grasp_axes,
        "overshoot": overshoot,
    },
)
events.sync_coupled_joint = EventTermCfg(func=mdp.sync_coupled_joint, mode="reset")
env_cfg.commands.reference = None
env_cfg.commands.pose_command = SequentialPoseCommandCfg(
    asset_name="robot",
    body_name=profile.tcp_body,
    body_offset=SequentialPoseCommandCfg.OffsetCfg(
        pos=tuple(profile.tcp_pos), rot=tuple(profile.tcp_rot)
    ),
    resampling_time_range=(1.0e6, 1.0e6),
    open_task_prob=1.0,
    output_joint_positions=True,
    valve_asset_name=mechanism.asset_name,
    valve_joint_name=mechanism.joint_name,
    valve_joint_open=mechanism.open_rad,
    valve_joint_closed=mechanism.closed_rad,
    valve_min_delta_rad=0.35,
    screw_coupling=mechanism.screw,
    commands=commands,
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
        "mechanism": env_cfg.mechanism.to_dict(),
        "method": method_name,
        "dt": env.step_dt,
        "horizon": horizon,
        "arm_joint_names": arm,
        "gripper_joint_names": list(hand_names),
        "phases": list(PHASES),
        "dynamics": list(mdp.DYNAMICS),
        "expert_diversity": list(EXPERT_DIVERSITY),
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
    task = env_cfg.mechanism
    print(
        f"[BANK] TCP in grasp frame '{task.grasp_frame}' when the turn starts, "
        f"median: pos {[round(v, 4) for v in g[:, :3].median(dim=0).values.tolist()]}, "
        f"xyzw {[round(v, 4) for v in g[:, 3:].median(dim=0).values.tolist()]}",
        flush=True,
    )
    env.close()
