"""FR3 scene and robot parameters for the shared ball-valve RL MDP."""

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_hiveboard.assets import FRANKA_EE
from isaaclab_hiveboard.tasks.franka.ball_valve.env import FrankaBallValveEnvCfg
from isaaclab_hiveboard.tasks.franka.common import (
    FRANKA_ARM_JOINT_NAMES,
    FRANKA_FINGER_BODY_NAMES,
    FRANKA_FINGER_JOINT_NAMES,
    FRANKA_NEWTON_CFG,
    FRANKA_OBJECT_POSE_RANGE,
)

from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import expert_bank, mdp
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.env import (  # noqa: F401 (collector settings)
    EXPERT_DIVERSITY_RANGES,
    EXPERT_OVERSHOOT,
    HOLD,
    REPO_ROOT,
    STUCK_BREAKAWAY_RANGE,
    STUCK_PROB,
    SUCCESS_TOLERANCE_RAD,
    VALVE_ANGLE_RANGE,
    VALVE_DYNAMICS_RANGES,
    AnymalBallValveRLEnvCfg,
    BallValveRLSceneCfg,
    configure_play,
)
from isaaclab_hiveboard_rl.tasks.robot import RobotCfg

from .delay import target_delay_cfg

HOME = tuple(FRANKA_NEWTON_CFG.init_state.joint_pos[j] for j in FRANKA_ARM_JOINT_NAMES)
ROBOT = RobotCfg(
    name="franka",
    arm_joint_names=tuple(FRANKA_ARM_JOINT_NAMES),
    hand_joint_names=tuple(FRANKA_FINGER_JOINT_NAMES),
    finger_joint_names=tuple(FRANKA_FINGER_JOINT_NAMES),
    finger_body_names=tuple(FRANKA_FINGER_BODY_NAMES),
    tcp_body=FRANKA_EE.body_name,
    tcp_pos=FRANKA_EE.tcp_offset.pos,
    tcp_rot=FRANKA_EE.tcp_offset.rot,
    home=HOME,
    actuator_groups=("fr3_shoulder", "fr3_forearm"),
    implicit=True,
    diagnostic_joint="fr3_joint6",
)
EXPERT_TASK = "Isaac-HiveBoard-Franka-BallValve-v0"
BANK_PREFIX = "franka_ball_valve"
EXPERT_BANK_PATH = os.path.join(REPO_ROOT, "logs", "expert_bank", BANK_PREFIX + "_bank.pt")
VALVE_POSE_RANGE = dict(FRANKA_OBJECT_POSE_RANGE)
ARM_POSTURES = {"home": HOME}
ARM_RANGE = (-0.1, 0.1)
ARM_FLIP_PROB = 0.0
ARM_WRIST_FLIP = None
EXPERT_GRIP_S = None
# The core FR3 grasp frame is +90 degrees about the shared frame's X axis.
# The physical lever's +Y direction is therefore -Z in the FR3 goal frame.
EXPERT_GRASP_AXES = ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0))


@configclass
class FrankaBallValveRLEnvCfg(AnymalBallValveRLEnvCfg):
    """Asymmetric PPO on the existing FR3 expert scene, with eight actions."""

    rl_robot: RobotCfg = ROBOT

    def __post_init__(self):
        # Compose the core FR3 scene without invoking ANYmal's explicit-drive conversion.
        expert = FrankaBallValveEnvCfg()
        self.scene = expert.scene
        self.scene.num_envs = 1024
        self.scene.env_spacing = 3.0
        self.scene.honeycomb = BallValveRLSceneCfg().honeycomb
        self.decimation = 33
        self.sim.dt = 0.05 / self.decimation
        self.sim.render_interval = self.decimation
        self.episode_length_s = 17.0
        self.viewer = expert.viewer
        arm = SceneEntityCfg("robot", joint_names=list(ROBOT.arm_joint_names), preserve_order=True)
        hand = SceneEntityCfg("robot", joint_names=list(ROBOT.finger_joint_names), preserve_order=True)
        self.actions.arm_action.joint_names = list(ROBOT.arm_joint_names)
        self.actions.gripper_action = expert.actions.gripper_action
        for group in (self.observations.policy, self.observations.teacher):
            group.arm_joint_pos.params["asset_cfg"] = arm
            group.gripper_pos.params["asset_cfg"] = hand
        self.observations.teacher.arm_joint_vel.params["asset_cfg"] = arm
        self.observations.policy.gripper_pos.noise = UniformNoiseCfg(n_min=-0.0002, n_max=0.0002)
        self.observations.policy.gripper_object.params = {
            "asset_cfg": hand,
            "moved_above": 0.0105,
            "stopped_below": 0.035,
            "max_speed": 0.001,
        }
        self.rewards.track_expert_gripper.params["std"] = 0.002
        self.rewards.arm_joint_vel.params["asset_cfg"] = arm
        self.terminations.invalid.params["asset_cfg"] = arm
        self.events.robot_physics_material.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=list(ROBOT.finger_body_names)
        )
        self.events.gripper_valve_contacts = expert.events.finger_contacts
        self.events.arm_delay = None
        self.events.reset_arm = None
        self.events.reset_valve_root = None
        self.events.reset_valve_joint = None
        self.events.reset_from_bank = expert_bank.reset_from_expert_bank_cfg(
            EXPERT_BANK_PATH,
            mid_start_prob=0.5,
            expert_task=EXPERT_TASK,
        )
        self.events.target_delay = target_delay_cfg()
        # Nominal cuRobo trace: <0.04 mm from lever_pivot, with a 180-degree
        # TCP-X rotation (the equivalent symmetric parallel-jaw grasp).
        self.valve_task = mdp.ValveTaskCfg(
            grasp_offset_pos=(0.0, 0.0, 0.0),
            grasp_offset_quat=(1.0, 0.0, 0.0, 0.0),
        )


@configclass
class FrankaBallValveRLEnvCfg_PLAY(FrankaBallValveRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


def configure_expert(env_cfg):
    """Complete the core opening-only sequence with the shared pipeline's release and retreat."""
    import copy

    from isaaclab_hiveboard.mdp.commands.sequential_pose_command import GripperCommand

    commands = env_cfg.commands.pose_command.commands
    rotate = next(i for i, c in enumerate(commands) if "Rotate" in type(c).__name__)
    for i, command in enumerate(commands):
        if i == rotate:
            command.phase = "actuate"
        elif isinstance(command, GripperCommand):
            command.phase = "grip" if i < rotate else "actuate"
        else:
            command.phase = "approach" if i == 0 else "engage"
    # The saved core FR3 setup ends with a closed-hand settle. Open after it,
    # wait for the ramp, then back off along the original approach direction.
    commands.append(GripperCommand(open_gripper=True, duration_s=0.8, phase="release"))
    retreat = copy.deepcopy(commands[0])
    retreat.phase = "retreat"
    retreat.gripper_open = True
    retreat.hold_current_orientation = True
    retreat.chain_with_next = False
    commands.append(retreat)
