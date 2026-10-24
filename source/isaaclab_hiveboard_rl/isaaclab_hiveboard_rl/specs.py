import copy
import math
import os

from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, FRANKA_EE
from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_JOINT_NAMES, ANYMAL_HOME_ARM
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import GripperCommand, MechanismGoalCfg
from isaaclab_hiveboard.tasks.franka.common import (
    FRANKA_ARM_JOINT_NAMES,
    FRANKA_FINGER_BODY_NAMES,
    FRANKA_FINGER_JOINT_NAMES,
    FRANKA_NEWTON_CFG,
    FRANKA_OBJECT_POSE_RANGE,
)
from isaaclab_hiveboard.tasks.scenes.threads import M30_SPEC

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@configclass
class RobotCfg:
    name: str = "anymal"
    arm_joint_names: tuple[str, ...] = tuple(ANYMAL_ARM_JOINT_NAMES)
    hand_joint_names: tuple[str, ...] = (
        "finger_joint",
        "left_outer_finger_joint",
        "left_inner_finger_joint",
        "left_inner_finger_pad_joint",
        "right_outer_knuckle_joint",
        "right_outer_finger_joint",
        "right_inner_finger_joint",
        "right_inner_finger_pad_joint",
    )
    finger_joint_names: tuple[str, ...] = ("finger_joint",)
    finger_body_names: tuple[str, str] = ("left_inner_finger", "right_inner_finger")
    tcp_body: str = ANYMAL_EE.body_name
    tcp_pos: tuple[float, float, float] = ANYMAL_EE.tcp_offset.pos
    tcp_rot: tuple[float, float, float, float] = ANYMAL_EE.tcp_offset.rot
    home: tuple[float, ...] = tuple(ANYMAL_HOME_ARM)
    actuator_groups: tuple[str, ...] = (
        "dynaarm_shoulder",
        "dynaarm_elbow",
        "dynaarm_forearm",
        "dynaarm_wrist_flex",
        "dynaarm_wrist_rot",
    )
    implicit: bool = False
    diagnostic_joint: str = "dynaarm_wrist_flexion"


ROBOTS = {
    "anymal": RobotCfg(),
    "franka": RobotCfg(
        name="franka",
        arm_joint_names=tuple(FRANKA_ARM_JOINT_NAMES),
        hand_joint_names=tuple(FRANKA_FINGER_JOINT_NAMES),
        finger_joint_names=tuple(FRANKA_FINGER_JOINT_NAMES),
        finger_body_names=tuple(FRANKA_FINGER_BODY_NAMES),
        tcp_body=FRANKA_EE.body_name,
        tcp_pos=FRANKA_EE.tcp_offset.pos,
        tcp_rot=FRANKA_EE.tcp_offset.rot,
        home=tuple(FRANKA_NEWTON_CFG.init_state.joint_pos[j] for j in FRANKA_ARM_JOINT_NAMES),
        actuator_groups=("fr3_shoulder", "fr3_forearm"),
        implicit=True,
        diagnostic_joint="fr3_joint6",
    ),
}


def configure_m30_thread_expert(env_cfg):
    seated = MechanismGoalCfg(goal_tolerance=0.1)
    for command in env_cfg.commands.pose_command.commands:
        if getattr(command, "phase", None) in ("engage", "grip", "actuate"):
            command.until = seated


def configure_circuit_breaker_expert(env_cfg):
    env_cfg.scene.circuit_breaker.init_state.joint_pos = {"RevoluteJoint": math.pi / 6}
    command = env_cfg.commands.pose_command
    press = next(
        c for c in command.commands if isinstance(c, GripperCommand) and c.phase == "actuate"
    )
    command.commands = [
        c for c in command.commands if not isinstance(c, GripperCommand) or c is press
    ]
    press.duration_s = 0.3
    for c in command.commands:
        if isinstance(c, GripperCommand):
            continue
        depth = 0.01 if c.phase in ("engage", "actuate") else 0.0
        c.target_offset_pos = (depth, 0.0, 0.02)
        if c.phase == "actuate":
            c.done_when_joint = ("circuit_breaker", "RevoluteJoint", -1.0, -math.pi / 6 + 0.035)
    engage = next(c for c in command.commands if c.phase == "engage")
    retreat = next(i for i, c in enumerate(command.commands) if c.phase == "retreat")
    clear = engage.replace(
        phase="retreat", target_frame_name="lever_pivot_above", target_offset_pos=(-0.06, 0.0, 0.02)
    )
    command.commands = command.commands[:retreat] + [clear] + command.commands[retreat:]


def configure_franka_ball_valve_expert(env_cfg):
    commands = env_cfg.commands.pose_command.commands
    rotate = next(i for i, c in enumerate(commands) if "Rotate" in type(c).__name__)
    for i, command in enumerate(commands):
        if i == rotate:
            command.phase = "actuate"
        elif isinstance(command, GripperCommand):
            command.phase = "grip" if i < rotate else "actuate"
        else:
            command.phase = "approach" if i == 0 else "engage"
    commands.append(GripperCommand(open_gripper=True, duration_s=0.8, phase="release"))
    retreat = copy.deepcopy(commands[0])
    retreat.phase = "retreat"
    retreat.gripper_open = True
    retreat.hold_current_orientation = True
    retreat.chain_with_next = False
    commands.append(retreat)


DIVERSITY = {
    "turn_rate": (0.25, 0.5),
    "reach_scale": (0.75, 1.3),
    "grasp_shift": (-0.015, 0.015),
    "grasp_roll": (-0.25, 0.25),
}
BALL_VALVE = {
    "expert_task": "Isaac-HiveBoard-Anymal-BallValve-v0",
    "valve_pose_range": {
        "x": (-0.05, 0.05),
        "y": (-0.06, 0.06),
        "z": (-0.05, 0.05),
        "roll": (-math.pi, math.pi),
        "pitch": (-math.radians(15.0), math.radians(15.0)),
        "yaw": (-math.radians(15.0), math.radians(15.0)),
    },
    "valve_angle_range": (-0.4, 0.0),
    "valve_dynamics_ranges": {
        "friction": (0.05, 2.0),
        "damping": (0.0, 0.5),
        "spring": (0.0, 0.0),
        "breakaway": (0.0, 2.0),
        "armature": (0.001, 0.02),
    },
    "stuck_prob": 0.25,
    "stuck_breakaway_range": (2.0, 5.0),
    "expert_diversity_ranges": DIVERSITY,
    "expert_grasp_axes": ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    "expert_overshoot": (0.03, 0.05),
    "expert_grip_s": None,
    "arm_postures": {
        "home": ANYMAL_HOME_ARM,
        "raised": (0.0, -0.4, 1.9, 0.0, 0.4, 1.5708),
        "lowered": (0.0, 0.3, 1.9, 0.0, -0.5, 1.5708),
        "left": (0.5, 0.0, 1.7, 0.0, 0.0, 1.5708),
        "right": (-0.5, 0.0, 1.7, 0.0, 0.0, 1.5708),
        "wrist_down": (0.0, 0.2, 1.8, 0.0, -0.8, 1.5708),
        "wrist_up": (0.0, 0.2, 1.8, 0.0, 0.8, 1.5708),
    },
    "arm_range": (-0.1, 0.1),
    "arm_flip_prob": 0.0,
    "arm_wrist_flip": tuple(
        ANYMAL_ARM_JOINT_NAMES.index(name)
        for name in ("dynaarm_forearm_rotation", "dynaarm_wrist_flexion", "dynaarm_wrist_rotation")
    ),
    "hold": {"dist_threshold": 0.03, "ang_threshold": 0.35},
    "success_tolerance_rad": 0.035,
    "expert_max_open_s": None,
    "configure_expert": None,
    "grip_torque_nm": 20.0,
}
TOOLS = {
    "ball_valve": BALL_VALVE,
    "small_valve": {
        **BALL_VALVE,
        "expert_task": "Isaac-HiveBoard-Anymal-SmallValve-v0",
        "valve_pose_range": {
            "x": (-0.05, 0.05),
            "y": (-0.06, 0.06),
            "z": (-0.05, 0.05),
            "roll": (-math.radians(15.0), math.radians(15.0)),
            "pitch": (-math.pi, math.pi),
            "yaw": (-math.radians(15.0), math.radians(15.0)),
        },
        "valve_dynamics_ranges": {
            "friction": (0.02, 1.0),
            "damping": (0.0, 0.1),
            "spring": (0.0, 0.0),
            "breakaway": (0.0, 0.0),
            "armature": (0.0005, 0.005),
        },
        "stuck_prob": 0.0,
        "stuck_breakaway_range": (0.0, 0.0),
        "expert_diversity_ranges": {
            **DIVERSITY,
            "grasp_shift": (-0.005, 0.005),
            "grasp_roll": (-0.12, 0.12),
        },
        "expert_grasp_axes": ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "expert_overshoot": (0.0, 0.05),
        "expert_grip_s": 0.3,
    },
    "m30_thread": {
        **BALL_VALVE,
        "expert_task": "Isaac-HiveBoard-Anymal-M30Thread-v0",
        "valve_angle_range": (0.0, 0.0),
        "valve_dynamics_ranges": {
            "friction": (0.05, 0.3),
            "damping": (0.0, 0.1),
            "spring": (0.0, 0.0),
            "breakaway": (0.0, 0.0),
            "armature": (0.001, 0.003),
        },
        "stuck_prob": 0.0,
        "stuck_breakaway_range": (0.0, 0.0),
        "expert_diversity_ranges": {
            **DIVERSITY,
            "turn_rate": (0.5, 0.75),
            "grasp_shift": (0.0, 0.0),
            "grasp_roll": (0.0, 0.0),
        },
        "expert_grasp_axes": ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "expert_overshoot": (0.0, 0.0),
        "expert_grip_s": 0.5,
        "success_tolerance_rad": 2.0 * math.pi * M30_SPEC.seated_tolerance / M30_SPEC.pitch,
        "configure_expert": configure_m30_thread_expert,
        "grip_torque_nm": 10.0,
    },
    "circuit_breaker": {
        **BALL_VALVE,
        "expert_task": "Isaac-HiveBoard-Anymal-CircuitBreaker-v0",
        "valve_pose_range": {
            "x": (-0.05, 0.05),
            "y": (-0.06, 0.06),
            "z": (-0.05, 0.05),
            "roll": (-math.radians(15.0), math.radians(15.0)),
            "pitch": (-math.radians(15.0), math.radians(15.0)),
            "yaw": (-math.radians(15.0), math.radians(15.0)),
        },
        "valve_angle_range": (0.0, 0.0),
        "valve_dynamics_ranges": {
            "friction": (0.01, 0.1),
            "damping": (0.0, 0.005),
            "spring": (0.0, 0.0),
            "breakaway": (0.0, 0.1),
            "armature": (0.0005, 0.002),
        },
        "stuck_prob": 0.0,
        "stuck_breakaway_range": (0.1, 0.1),
        "expert_diversity_ranges": {
            **DIVERSITY,
            "turn_rate": (0.1, 0.2),
            "grasp_shift": (-0.01, 0.01),
            "grasp_roll": (-0.3, 0.3),
        },
        "expert_grasp_axes": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        "expert_overshoot": (0.0, 0.0),
        "success_tolerance_rad": 0.1,
        "expert_max_open_s": 5.0,
        "configure_expert": configure_circuit_breaker_expert,
    },
}
ROBOT_TOOLS = {
    ("franka", "ball_valve"): {
        "expert_task": "Isaac-HiveBoard-Franka-BallValve-v0",
        "valve_pose_range": dict(FRANKA_OBJECT_POSE_RANGE),
        "arm_postures": {"home": ROBOTS["franka"].home},
        "arm_wrist_flip": None,
        "expert_grasp_axes": ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0)),
        "configure_expert": configure_franka_ball_valve_expert,
    }
}
PAIRS = [
    ("anymal", "ball_valve"),
    ("anymal", "small_valve"),
    ("anymal", "m30_thread"),
    ("anymal", "circuit_breaker"),
    ("franka", "ball_valve"),
]


def spec(robot, tool):
    return {**TOOLS[tool], **ROBOT_TOOLS.get((robot, tool), {})}


def task_id(robot, tool, play=False):
    name = "".join(word.capitalize() for word in tool.split("_"))
    return f"Isaac-HiveBoard-{robot.capitalize()}-{name}-RL{'-Play' if play else ''}-v0"


def bank_path(robot, tool):
    return os.path.join(REPO_ROOT, "logs", "expert_bank", f"{robot}_{tool}.pt")
