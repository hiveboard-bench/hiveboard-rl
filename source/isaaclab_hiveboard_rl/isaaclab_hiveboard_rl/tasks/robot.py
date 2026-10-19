"""Robot-specific contracts used by the shared bank-guided RL pipeline."""

from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE
from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_JOINT_NAMES, ANYMAL_HOME_ARM


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


def robot_cfg(env):
    """ANYmal fallback supports existing tasks, tests, and legacy banks."""
    return getattr(env.cfg, "rl_robot", None) or RobotCfg()
