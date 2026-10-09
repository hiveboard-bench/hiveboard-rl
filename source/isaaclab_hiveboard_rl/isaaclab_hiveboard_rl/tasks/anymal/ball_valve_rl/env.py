# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL (teacher/student) configuration for the fixed-base ANYmal + DynaArm ball valve.

The scene, robot, valve and domain randomization match the scripted-expert
task in ``tasks/anymal/ball_valve``; what changes is the MDP:

* Action: integrated arm joint-position increments (7 = 6 arm + 1 gripper).
  The gripper is binary (negative closes), ramped like the expert's.
* ``teacher`` observations: privileged simulator state, used by PPO for both
  actor and critic.
* ``policy`` observations: what the robot has at deployment - arm encoders,
  gripper angle, forward-kinematics TCP pose, previous action, and the
  registered valve pose with a per-episode registration error - stacked over
  a short history. The student is distilled from the teacher on these.
"""

import math
import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_hiveboard.assets.anymal.bench import (
    ANYMAL_ARM_JOINT_NAMES,
    ANYMAL_HOME_ARM,
    ANYMAL_NEWTON_GRIPPER_OPEN,
    NEWTON_GRIPPER_JOINT_NAMES,
)
from isaaclab_hiveboard.mdp.actions import RateLimitedBinaryJointPositionActionCfg
from isaaclab_hiveboard.mdp.events import apply_articulation_gravcomp, set_contact_stiffness
from isaaclab_hiveboard.tasks.anymal.ball_valve.configs.scene import BallValveSceneCfg
from isaaclab_hiveboard.tasks.viewer import use_play_viewer

from isaaclab_tasks.core.cabinet import mdp as base_mdp
from isaaclab_tasks.utils import PresetCfg

from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import (
    actuator_delay,
    expert_bank,
    expert_diversity,
    mdp,
    valve_dynamics,
)

ARM = SceneEntityCfg("robot", joint_names=list(ANYMAL_ARM_JOINT_NAMES), preserve_order=True)
GRIPPER = SceneEntityCfg("robot", joint_names=list(NEWTON_GRIPPER_JOINT_NAMES), preserve_order=True)
VALVE_JOINT = SceneEntityCfg("ball_valve", joint_names=[mdp.VALVE_JOINT])

# Same tolerance as the scripted-expert task's success term.
SUCCESS_TOLERANCE_RAD = 0.035
# The lever counts as held when the gripper is closed within these TCP errors
# [m, rad] of the expert grasp pose; the same gate scores grasp, turn and open.
HOLD = {"dist_threshold": 0.03, "ang_threshold": 0.35}

# Reset distribution, shared with scripts/rl/build_expert_bank.py so the bank's
# trajectories start from the same states the uniform resets would draw.
VALVE_POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.06, 0.06),
    "z": (-0.05, 0.05),
    "roll": (-math.pi, math.pi),
    "pitch": (-math.radians(15.0), math.radians(15.0)),
    "yaw": (-math.radians(15.0), math.radians(15.0)),
}
"""Valve root pose offset [m, rad], in the valve's frame: board placement beyond the student's registration
error. Roll turns the valve about its stem through a full turn, so the lever can close and open in any
direction (sideways, up); pitch and yaw tilt the panel."""
VALVE_ANGLE_RANGE = (-0.4, 0.0)
"""Initial valve angle offset [rad]: some episodes start part-open."""
ARM_POSTURES = {
    "home": ANYMAL_HOME_ARM,
    "raised": (0.0, -0.4, 1.9, 0.0, 0.4, 1.5708),
    "lowered": (0.0, 0.3, 1.9, 0.0, -0.5, 1.5708),
    "left": (0.5, 0.0, 1.7, 0.0, 0.0, 1.5708),
    "right": (-0.5, 0.0, 1.7, 0.0, 0.0, 1.5708),
    "wrist_down": (0.0, 0.2, 1.8, 0.0, -0.8, 1.5708),
    "wrist_up": (0.0, 0.2, 1.8, 0.0, 0.8, 1.5708),
}
"""Arm start postures [rad, arm joint order], one picked per episode: the stowed home pose, the hand raised,
lowered, swung to either side, and the wrist bent either way."""
ARM_RANGE = (-0.1, 0.1)
"""Initial arm joint offset around the posture [rad]."""
ARM_FLIP_PROB = 0.0
"""Fraction of starts on a posture's wrist-flipped twin (forearm rolled half a turn, same hand pose).

Off: the real arm starts near its home pose (forearm near 0). With the valve clocked through a full turn,
half the flipped starts had to roll the forearm back by pi during the reach, which student v9 failed in a
third of them (68% reached vs 93% from the usual starts)."""
ARM_WRIST_FLIP = tuple(
    ANYMAL_ARM_JOINT_NAMES.index(name)
    for name in ("dynaarm_forearm_rotation", "dynaarm_wrist_flexion", "dynaarm_wrist_rotation")
)
"""The (roll, pitch, roll) wrist joints in the arm joint order."""

# Force-limited gripper, like the real 2F-140 (grip force set between 10 and
# 125 N). The shared ANYmal config drives the finger to 0.7 rad with an
# 80 N.m/rad PD, so on the lever (stopped at ~0.68 rad) it squeezed with
# ~1.6 N.m, ~12 N per pad, and held only ~1 N.m of valve torque. Here the
# fingers are commanded past any contact (and past the finger joint's 0.785
# rad limit) and the drive's torque limit sets the squeeze, which then no
# longer depends on where the fingers stop: ~5 N per pad per N.m.
GRIPPER_CLOSE_RAD = 1.5
"""Finger target when closed [rad], beyond the 0.785 rad joint limit."""
GRIP_TORQUE_NM = 20.0
"""Finger drive torque limit [N.m]; sets the grip force: ~100 N per pad on the lever."""
# Stiff gripper-valve contacts (as for the small valve): at the default
# MJWarp softness a 100 N squeeze sinks the pads millimetres into the lever.
GRIPPER_VALVE_CONTACT = {"shape_regex": "/robotiq_2f_140/|/Valve/", "ke": 4.0e4, "kd": 400.0, "solimp": (0.95, 0.99, 0.001)}


def force_limited_gripper(robot: ArticulationCfg, torque: float = GRIP_TORQUE_NM) -> ArticulationCfg:
    """``robot`` with its finger drive's torque limited to ``torque`` [N.m] (see :data:`GRIP_TORQUE_NM`)."""
    gripper = robot.actuators["gripper"].replace(effort_limit=torque)
    return robot.replace(actuators={**robot.actuators, "gripper": gripper})


# Precomputed cuRobo expert trajectories (scripts/rl/build_expert_bank.py),
# recorded on the scripted task EXPERT_TASK.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))
EXPERT_TASK = "Isaac-HiveBoard-Anymal-BallValve-v0"
EXPERT_BANK_PATH = os.path.join(REPO_ROOT, "logs", "expert_bank", "anymal_ball_valve_bank_5000_noflip.pt")
BANK_PREFIX = "anymal_ball_valve"
"""Default bank file name prefix (``logs/expert_bank/<prefix>_bank.pt``)."""
# What the bank builder samples per episode besides the reset ranges above:
# the valve dynamics (valve_dynamics.py) and the expert's own variation
# (expert_diversity.py). The RL episodes restore them from the bank.
VALVE_DYNAMICS_RANGES = valve_dynamics.VALVE_DYNAMICS_RANGES
STUCK_PROB = valve_dynamics.STUCK_PROB
STUCK_BREAKAWAY_RANGE = valve_dynamics.STUCK_BREAKAWAY_RANGE
EXPERT_DIVERSITY_RANGES = expert_diversity.EXPERT_DIVERSITY_RANGES
EXPERT_GRASP_AXES = expert_diversity.LEVER_GRASP_AXES
EXPERT_OVERSHOOT = (expert_diversity.OVERSHOOT_RAD, expert_diversity.OVERSHOOT_RAD_PER_NM)
EXPERT_GRIP_S = None
"""Duration [s] the bank builder gives the expert's grip segments (None: the command setup's)."""


@configclass
class BallValveRLPhysicsCfg(PresetCfg):
    """Newton MJWarp, sized for thousands of parallel worlds.

    ``nconmax``/``njmax`` are per world; the scripted task's 4000 contacts per
    world only fits a handful of environments.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=600,
            nconmax=300,
            iterations=100,
            ls_iterations=20,
            impratio=10.0,
            ccd_iterations=50,
            use_mujoco_contacts=False,
        ),
        num_substeps=2,
        debug_mode=False,
        # Startup gravity-compensation changes must be evaluated by the solver.
        use_cuda_graph=False,
    )
    default = newton_mjwarp


@configclass
class ActionsCfg:
    """Integrated arm joint increments plus a ramped binary gripper."""

    arm_action = mdp.IntegratedJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ANYMAL_ARM_JOINT_NAMES),
        preserve_order=True,
        # 0.1 rad per 20 Hz step = 2 rad/s at |a| = 1 (clip_actions = 1). The
        # cuRobo expert's shoulder and wrist flexion reach 1.7-1.8 rad/s.
        scale=0.1,
    )
    gripper_action = RateLimitedBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(NEWTON_GRIPPER_JOINT_NAMES),
        open_command_expr=dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_OPEN)),
        close_command_expr={NEWTON_GRIPPER_JOINT_NAMES[0]: GRIPPER_CLOSE_RAD},
        close_speed=2.0,
    )
    # No actions: applies the valve's seat torque (valve_dynamics.py).
    valve_load = valve_dynamics.ValveLoadActionCfg()


@configclass
class ObservationsCfg:
    """Deployable ``policy`` (student) and privileged ``teacher`` groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioception + registered target pose; available on the real robot."""

        arm_joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel, params={"asset_cfg": ARM}, noise=Unoise(n_min=-0.002, n_max=0.002)
        )
        gripper_pos = ObsTerm(
            func=base_mdp.joint_pos, params={"asset_cfg": GRIPPER}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        tcp_pose = ObsTerm(func=mdp.tcp_pose_b)
        registered_valve = ObsTerm(
            func=mdp.registered_valve_b,
            # ~1 cm / ~2 deg registration error and ~2 deg on the lever's
            # initial angle, latched per episode; 2 mm jitter per step.
            params={"bias_pos": 0.01, "bias_rot": 0.035, "jitter_pos": 0.002, "angle_noise": 0.035},
        )
        last_action = ObsTerm(func=base_mdp.last_action)
        # Operator input on the robot: how fast to turn the valve.
        turn_rate = ObsTerm(func=mdp.turn_rate_command, params={"command_name": "valve_turn"})
        # The gripper's own "object detected" status (no force sensing):
        # whether the grasp landed on the lever.
        gripper_object = ObsTerm(func=mdp.gripper_object_detected, params={"asset_cfg": GRIPPER})
        # Time since the task started (the teacher's references are timed).
        episode_time = ObsTerm(func=mdp.episode_time)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class TeacherCfg(ObsGroup):
        """Privileged simulator state for the PPO teacher's actor and critic."""

        arm_joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": ARM})
        arm_joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": ARM}, scale=0.1)
        gripper_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": GRIPPER})
        tcp_pose = ObsTerm(func=mdp.tcp_pose_b)
        valve_pose = ObsTerm(func=mdp.valve_pose_b)
        grasp_target = ObsTerm(func=mdp.grasp_target_b)
        tcp_to_grasp = ObsTerm(func=mdp.tcp_to_grasp_b, scale=5.0)
        valve_state = ObsTerm(func=mdp.valve_state)
        pad_force = ObsTerm(func=mdp.pad_valve_force, scale=0.02, clip=(0.0, 10.0))
        last_action = ObsTerm(func=base_mdp.last_action)
        # Appended last so a pre-command checkpoint can be warm-started by
        # zero-padding its input layer (scripts/rl/expand_checkpoint_inputs.py).
        turn_rate = ObsTerm(func=mdp.turn_rate_command, params={"command_name": "valve_turn"})
        valve_reference = ObsTerm(func=mdp.valve_reference_state, params={"command_name": "valve_turn"})
        # Distance of each arm joint from the expert's at the same episode
        # time. Appended last, like the command terms above.
        expert_joint_error = ObsTerm(func=expert_bank.expert_joint_error, params={"command_name": "valve_turn"})
        expert_gripper = ObsTerm(func=expert_bank.expert_gripper_reference, params={"command_name": "valve_turn"})
        # Appended last as well (teacher v17 warm-started from v16).
        expert_valve_error = ObsTerm(func=expert_bank.expert_valve_error, scale=5.0)
        # How hard this episode's lever is to turn.
        valve_dynamics = ObsTerm(func=valve_dynamics.valve_dynamics_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


@configclass
class EventCfg:
    """Gravity compensation, the scripted task's valve DR, and reset randomization."""

    quiet_solver = EventTerm(func=mdp.silence_solver_overflow_warnings, mode="startup")
    robot_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "gravcomp": 1.0},
    )
    valve_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("ball_valve"), "gravcomp": 1.0},
    )
    robot_physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger", "right_inner_finger"]),
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.3, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    valve_physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball_valve"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )

    # After the material events, which re-sync shape properties.
    gripper_valve_contacts = EventTerm(func=set_contact_stiffness, mode="startup", params=dict(GRIPPER_VALVE_CONTACT))

    # Reset order matters: defaults first, then the randomized offsets. The
    # integrated arm action then starts from the resulting joints (it resets
    # after the events).
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    # The arm's command latency, 0-40 ms per episode (actuator_delay.py).
    arm_delay = actuator_delay.randomize_actuator_delay_cfg()
    reset_valve_root = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Board placement relative to the robot, beyond the registration
            # error the student sees: the policy must use the registered pose.
            "pose_range": dict(VALVE_POSE_RANGE),
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ball_valve"),
        },
    )
    reset_valve_joint = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        # Start some episodes part-open so the turning phase gets explored
        # early (a sampling-based curriculum in the IndustReal sense).
        params={"position_range": VALVE_ANGLE_RANGE, "velocity_range": (0.0, 0.0), "asset_cfg": VALVE_JOINT},
    )
    reset_arm = EventTerm(
        func=mdp.reset_joints_from_postures,
        mode="reset",
        params={
            "postures": dict(ARM_POSTURES),
            "position_range": ARM_RANGE,
            "asset_cfg": ARM,
            "flip_prob": ARM_FLIP_PROB,
            "flip_joints": ARM_WRIST_FLIP,
        },
    )
    # Friction, damping, return spring, seat breakaway and inertia of the
    # lever, per episode, some valves stuck closed (which reseats the lever,
    # so after the joint reset); a bank reset below replaces them by its
    # trajectory's.
    valve_dynamics = valve_dynamics.randomize_valve_dynamics_cfg()
    # Replaces the three uniform resets above when an expert bank is used
    # (see AnymalBallValveRLEnvCfg.__post_init__): same distribution, but each
    # episode starts exactly where a successful expert trajectory started.
    # Half the episodes start mid-trajectory, e.g. already gripping and turning,
    # so gripping is experienced without having to be discovered first.
    reset_from_bank = expert_bank.reset_from_expert_bank_cfg(EXPERT_BANK_PATH, mid_start_prob=0.5, expert_task=EXPERT_TASK)


@configclass
class CommandsCfg:
    """Operator-set turning speed; its reference angle is what the turn is scored against."""

    # The expert bank turns at 0.31 rad/s (median; p10-p90 0.30-0.33), so the
    # teacher is asked for the same speed its valve-tracking reward follows.
    valve_turn = mdp.ValveTurnRateCommandCfg(rate_range=(0.3, 0.3), hold_dist=0.03, hold_ang=0.35)


@configclass
class RewardsCfg:
    """Tracking the expert by episode time: its arm joints, gripper and valve angle.

    There is no "lever held" gate. Teacher v19 showed that a held test built
    from the TCP pose and the commanded gripper could be met without gripping:
    it closed the fingers to just short of the lever, reopened for one step
    every seven, and collected every hold-gated reward. With the references
    all indexed by time, following the expert's joints (see TerminationsCfg),
    closing when it closes and keeping the valve on its schedule is the task.

    Episodes do not end on success: once the expert has opened the valve, the
    valve terms keep paying for holding it open. A terminal bonus would be
    outweighed by the dense reward forfeited for the rest of the episode.
    """

    # Follow the expert joint by joint, not only at the TCP: near the wrist
    # singularity a small TCP error can hide large forearm/wrist excursions.
    # Cartesian reach terms paid for being at the lever as early as possible,
    # which pulled teacher v11 onto a fast approach 0.5-0.8 rad off in
    # orientation. The coarse kernel keeps a gradient back to the expert when
    # the arm is far from it; the fine one asks for precision.
    track_expert_coarse = RewTerm(func=expert_bank.track_expert_joints, weight=2.0, params={"std": 0.5})
    track_expert = RewTerm(func=expert_bank.track_expert_joints, weight=3.0, params={"std": 0.15})
    # Opt-in ablation: enable with env.rewards.track_expert_velocity.weight=1.0.
    track_expert_velocity = RewTerm(func=expert_bank.track_expert_velocity, weight=0.0, params={"std": 0.5})
    # Close the gripper when the expert does (teacher v10, rewarded on joints
    # only, reached the lever every time but never closed it in 600 iterations;
    # at weight 1.0, v13/v14 unlearned closing by iteration 300). Tracks the
    # finger position, not the command (see track_expert_gripper).
    track_expert_gripper = RewTerm(func=expert_bank.track_expert_gripper, weight=3.0, params={"std": 0.1})
    # Touch the lever with both pads exactly when the expert does: no other
    # term pays for contact, while contact can disturb the tracked joints.
    track_expert_contact = RewTerm(func=expert_bank.track_expert_contact, weight=3.0)
    # Turn the valve on the expert's schedule (by episode time). The turning
    # rate command is fixed to the expert's speed to match; a bank with varied
    # speeds would let it vary again. The coarse kernel keeps a gradient when
    # the valve is far behind, where the 0.1 rad one is flat (teacher v17).
    # Opt-in: pay only with both pads on the lever while the expert grips it, so
    # pushing the lever open does not pay (enable gate_on_contact=true on both).
    track_expert_valve = RewTerm(
        func=expert_bank.track_expert_valve, weight=6.0, params={"std": 0.1, "gate_on_contact": False}
    )
    track_expert_valve_coarse = RewTerm(
        func=expert_bank.track_expert_valve, weight=5.0, params={"std": 0.5, "gate_on_contact": False}
    )
    # Strong enough that the raw actions are smooth themselves, not only after
    # the action filter: teacher v7 (-0.05, no magnitude term) dithered between
    # +-1 every step behind the filter (lag-1 autocorrelation -0.85), which the
    # student could not imitate and a real drive should not receive.
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.5)
    action_magnitude = RewTerm(func=base_mdp.action_l2, weight=-0.02)
    arm_joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-1.0e-3, params={"asset_cfg": ARM})
    pad_force = RewTerm(func=mdp.pad_force_excess, weight=-1.0e-3, params={"max_force": 150.0})
    valve_overspeed = RewTerm(func=mdp.valve_rate_excess, weight=-1.0, params={"factor": 1.5})


@configclass
class TerminationsCfg:
    """Timeout, solver blow-up, and leaving the expert (success is scored, not terminated; see RewardsCfg)."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    invalid = DoneTerm(func=mdp.invalid_state, params={"asset_cfg": ARM})
    # Follow the expert or lose the rest of the episode (DeepMimic-style early
    # termination). 0.3 rad is 5-10x the expert's own tracking error.
    expert_drift = DoneTerm(func=expert_bank.expert_joint_drift, params={"max_error": 0.3})
    # Teacher v17 held the lever still, right on the joint reference (which
    # then followed the valve's actual angle); falling off the expert's valve
    # schedule ends the episode instead.
    expert_valve_lag = DoneTerm(func=expert_bank.expert_valve_lag, params={"max_error": 0.3})


@configclass
class AnymalBallValveRLEnvCfg(ManagerBasedRLEnvCfg):
    """ANYmal + DynaArm opens the HiveBoard ball valve (closed -> -90 deg) with RL."""

    scene: BallValveSceneCfg = BallValveSceneCfg(num_envs=1024, env_spacing=3.0)  # type: ignore
    # The valve every RL term acts on (mdp.ValveTaskCfg; defaults are the ball valve's).
    valve_task: mdp.ValveTaskCfg = mdp.ValveTaskCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=10, physics=BallValveRLPhysicsCfg())  # type: ignore

    def __post_init__(self):
        # TODO: Bring back BallValve's arm-target error bound (it stops stored drive error from snapping the
        # arm when it lets go) with a larger value than the 0.1 rad it had, and have the expert bank move
        # smoothly enough to stay inside it. At 0.1 rad only 59% of bank trajectories still opened the valve
        # through the increment action (98% unbounded), which capped the BC baselines; it is off until then.
        # Reproduce the old controller with env.actions.arm_action.max_position_error=0.1.
        # 20 Hz policy. The slowest commanded turn (0.25 rad/s) takes 6.3 s; the
        # expert's slowest reaches (a far start posture, a wrist flip, a slow
        # reach speed) take ~5 s, and 99% of its trajectories open by 12.5 s.
        # Letting go and backing off to the approach point takes ~1.5 s more.
        self.decimation = 10
        self.episode_length_s = 17.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "ball_valve"
        self.viewer.body_name = "alavanca_pivot"
        self.viewer.eye = (-1.5, 1.5, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.scene.robot = force_limited_gripper(self.scene.robot)
        self.scene.robot = actuator_delay.delayed_actuators(
            self.scene.robot, actuator_delay.ARM_ACTUATOR_GROUPS, actuator_delay.ARM_DELAY_RANGE_S[1], self.sim.dt
        )
        # Bank resets replace the uniform ones; they come from the same ranges.
        self.events.reset_valve_root = None
        self.events.reset_valve_joint = None
        self.events.reset_arm = None


@configclass
class AnymalBallValveRLEnvCfg_PLAY(AnymalBallValveRLEnvCfg):
    """Evaluation: nominal physics, bank starts (randomized like training), registration error kept."""

    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


def configure_play(cfg: AnymalBallValveRLEnvCfg) -> None:
    """Turn an RL valve task configuration into its evaluation variant (see :class:`AnymalBallValveRLEnvCfg_PLAY`)."""
    cfg.scene.num_envs = 16
    cfg.events.robot_physics_material = None
    cfg.events.valve_physics_material = None
    cfg.observations.policy.enable_corruption = False
    # Evaluate whole tasks, from the start of each trajectory.
    cfg.events.reset_from_bank.params["mid_start_prob"] = 0.0
    # Episodes run to the end, as on the robot: the expert-reference
    # terminations are a training device for the teacher.
    cfg.terminations.expert_drift = None
    cfg.terminations.expert_valve_lag = None
    # The cuRobo expert's turning speed; sweep with
    # env.commands.valve_turn.rate_range=[r,r].
    cfg.commands.valve_turn.rate_range = (0.3, 0.3)
    use_play_viewer(cfg)
