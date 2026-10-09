# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the RL ANYmal ball-valve task (teacher/student).

Two observation tiers are built from these terms:

* ``policy`` (the deployable student): arm encoders, gripper motor angle, TCP
  pose from forward kinematics, the previous action, and the *registered*
  valve pose. On hardware the registered pose comes from a one-off board
  registration (AprilTag or touch-probing), so in simulation it is the true
  root pose plus a per-episode bias that stays constant for the episode and a
  small per-step jitter. The student never sees the valve angle; once the
  lever is in the gripper the TCP pose already encodes it.
* ``teacher`` (privileged, simulation only): adds joint velocities, the true
  valve pose, the valve angle and rate, the lever grasp target, and the
  valve-filtered pad contact forces.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import BaseArticulation
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

VALVE_JOINT = "RevoluteJoint"
# HiveBoard ball-valve limits are [-pi/2, 0]; negative rotation opens it.
VALVE_CLOSED_RAD = 0.0
VALVE_OPEN_RAD = -1.5707963267948966

# TCP pose in the lever's ``lever_pivot`` frame at the cuRobo expert's grasp,
# measured by ``scripts/rl/measure_grasp_offset.py`` by replaying the expert in
# ``anymal_ball_valve_expert_50.hdf5``. The frame is attached to the lever, so
# the target follows the lever while it turns.
GRASP_OFFSET_POS = (0.0, -0.0244, 0.0)
GRASP_OFFSET_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)


@configclass
class ValveTaskCfg:
    """Which valve the RL terms act on, and where on it the expert grasps.

    Set as ``valve_task`` on the environment configuration; every term in this
    package reads it from there. The defaults are the ball valve's.
    """

    asset_name: str = "ball_valve"
    """Scene articulation of the valve."""
    joint_name: str = VALVE_JOINT
    """The joint that is turned."""
    closed_rad: float = VALVE_CLOSED_RAD
    """Joint angle when closed [rad]."""
    open_rad: float = VALVE_OPEN_RAD
    """Joint angle when open [rad]."""
    grasp_frame: str = "lever_pivot"
    """``target_frame`` entry the grasp offset is relative to. It must be attached to the turning body
    (the last element of its prim path), so the grasp target turns with the valve."""
    grasp_offset_pos: tuple[float, float, float] = GRASP_OFFSET_POS
    """Expert grasp TCP position in :attr:`grasp_frame` [m]."""
    grasp_offset_quat: tuple[float, float, float, float] = GRASP_OFFSET_QUAT
    """Expert grasp TCP orientation in :attr:`grasp_frame`, xyzw."""
    closed_end_stop: bool = True
    """Whether a joint limit stops the valve at :attr:`closed_rad`. Without one (a continuous handwheel), a
    torque toward closed spins it past closed, so ``valve_dynamics`` refuses a seat torque or spring."""
    coupled_joint_name: str | None = None
    """A joint of the same articulation that follows the turned one, e.g. a nut's travel along its thread
    (``coupled = coupled_offset + coupled_ratio * angle``, held by a mimic constraint). Resets that write the
    angle write it too, so the constraint is never violated at the start of an episode."""
    coupled_ratio: float = 0.0
    """Coupled joint travel per radian of the turned joint [m/rad]."""
    coupled_offset: float = 0.0
    """Coupled joint position at angle 0 [m]."""
    hold_by_contact: bool = False
    """Count the valve as held when the gripper is closed with both pads touching it (over
    ``expert_bank.CONTACT_FORCE_N``), instead of at the expert grasp pose. For a part that is regrasped and
    turned through several grasps (a nut), where the hand turns away from the fixed grasp pose."""
    push: bool = False
    """The expert pushes the lever over with its closed hand instead of grasping it (a circuit breaker's
    toggle). The lever counts as held while either pad touches it; reaching open counts once a pad has
    touched it earlier in the episode, since a pushed lever runs ahead of the hand to its stop
    (:func:`opening_credited`); and letting go needs no open gripper (:func:`released_and_clear`)."""


def valve_task(env: ManagerBasedEnv) -> ValveTaskCfg:
    """The environment's :class:`ValveTaskCfg` (``env.cfg.valve_task``)."""
    task = getattr(env.cfg, "valve_task", None)
    if task is None:
        raise RuntimeError("The RL valve terms need env.cfg.valve_task (a ValveTaskCfg).")
    return task


def valve_asset(env: ManagerBasedEnv) -> BaseArticulation:
    """The valve articulation."""
    return env.scene[valve_task(env).asset_name]


def valve_joint_id(env: ManagerBasedEnv) -> int:
    """Index of the turned joint in the valve articulation."""
    task = valve_task(env)
    return env.scene[task.asset_name].find_joints(task.joint_name)[0][0]


def write_valve_angle(
    env: ManagerBasedEnv, env_ids: torch.Tensor, angle: torch.Tensor, rate: torch.Tensor | None = None
) -> None:
    """Write the turned joint's ``angle`` [rad] and ``rate`` [rad/s] (default 0), each ``(len(env_ids),)``,
    for environments ``env_ids``, and the coupled joint's matching state (:attr:`ValveTaskCfg.coupled_joint_name`).
    """
    task = valve_task(env)
    valve = valve_asset(env)
    rate = torch.zeros_like(angle) if rate is None else rate
    joints = [valve_joint_id(env)]
    pos, vel = angle[:, None], rate[:, None]
    if task.coupled_joint_name is not None:
        joints.append(valve.find_joints(task.coupled_joint_name)[0][0])
        pos = torch.cat((pos, task.coupled_offset + task.coupled_ratio * pos), dim=-1)
        vel = torch.cat((vel, task.coupled_ratio * vel), dim=-1)
    valve.write_joint_position_to_sim_index(position=pos.contiguous(), joint_ids=joints, env_ids=env_ids)
    valve.write_joint_velocity_to_sim_index(velocity=vel.contiguous(), joint_ids=joints, env_ids=env_ids)


def sync_coupled_joint(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Reset event: put the coupled joint where the turned joint's angle says (after a reset of that angle)."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    if valve_task(env).coupled_joint_name is not None:
        write_valve_angle(env, ids, valve_angle(env)[ids].clone())


def valve_rate(env: ManagerBasedEnv) -> torch.Tensor:
    """Valve joint rate [rad/s], shape ``(num_envs,)``."""
    return valve_asset(env).data.joint_vel.torch[:, valve_joint_id(env)]


def _progress_of(env: ManagerBasedEnv, angle: torch.Tensor) -> torch.Tensor:
    """Opening progress of ``angle`` [rad], unclamped: 0 closed, 1 open."""
    task = valve_task(env)
    return (angle - task.closed_rad) / (task.open_rad - task.closed_rad)


def _opening_sign(env: ManagerBasedEnv) -> float:
    """+1 if opening increases the joint angle, -1 if it decreases it."""
    task = valve_task(env)
    return 1.0 if task.open_rad > task.closed_rad else -1.0


def _is_open(env: ManagerBasedEnv, angle: torch.Tensor, threshold_rad: float) -> torch.Tensor:
    """True where ``angle`` is within ``threshold_rad`` [rad] of open, or past it."""
    task = valve_task(env)
    return _opening_sign(env) * (task.open_rad - angle) <= threshold_rad


##
# Actions
##


class IntegratedJointPositionAction(JointAction):
    """Joint position target that integrates scaled action increments.

    ``target <- clamp(target + scale * a_f, joint limits)`` with the action
    low-pass filtered, ``a_f <- (1 - smoothing) * a_f + smoothing * action``.
    This is the joint-space version of IndustReal's policy-level action
    integrator: a zero action holds the previous target, so the PD drive keeps
    full stiffness under load (turning the lever) instead of chasing the
    sagging measured position the way :class:`RelativeJointPositionAction`
    does. The filter is part of the action, so the robot runs it too; without
    it the teacher alternated its increments every step and chattered the
    lever at 10 Hz.

    ``cfg.command`` selects what the policy outputs (the behaviour-cloning
    baselines compare them; see :data:`ARM_COMMANDS`).
    """

    cfg: IntegratedJointPositionActionCfg

    def __init__(self, cfg: IntegratedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.command not in ARM_COMMANDS:
            raise ValueError(f"Unknown arm command '{cfg.command}'; expected one of {ARM_COMMANDS}")
        self._offset = 0.0
        self._fresh = False
        self._filtered = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._target = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._lower = limits[..., 0].clone()
        self._upper = limits[..., 1].clone()
        limit = cfg.max_position_error
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            raise ValueError("max_position_error must be finite and positive, or None")

    def increment_toward(self, target: torch.Tensor) -> torch.Tensor:
        """The unit action (clipped to +-1, as PPO executes it) moving the integrated target toward ``target``."""
        return ((target - self._target) / self.cfg.scale).clamp(-1.0, 1.0)

    def measured(self) -> torch.Tensor:
        """Measured positions of the arm joints [rad], the anchor of the ``relative`` commands."""
        return self._asset.data.joint_pos.torch[:, self._joint_ids]

    def apply_actions(self):
        if self.cfg.command in ("absolute", "relative"):
            # Plain joint-position control, as the scripted expert's task runs it.
            if self._fresh:
                torch.clamp(self.processed_actions, self._lower, self._upper, out=self._target)
                self._fresh = False
            self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)
            return
        # Newton runs the decimation loop itself, so this is called once per
        # env step there and once per physics step on PhysX. Integrate only on
        # a fresh action so both backends see the same per-step increment.
        if self._fresh:
            alpha = self.cfg.smoothing
            self._filtered.mul_(1.0 - alpha).add_(self.processed_actions, alpha=alpha)
            self._target += self._filtered
            # Store the bounded target itself: clipping only the drive's output
            # would leave a hidden integrator winding up against contact.
            if self.cfg.max_position_error is not None:
                measured = self._asset.data.joint_pos.torch[:, self._joint_ids]
                limit = self.cfg.max_position_error
                torch.clamp(self._target, measured - limit, measured + limit, out=self._target)
            torch.clamp(self._target, self._lower, self._upper, out=self._target)
            self._fresh = False
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        target = self._raw_actions
        if self.cfg.command.startswith("relative"):
            target = target + self.measured()
        if self.cfg.command in ("absolute", "relative"):
            self._processed_actions = target.clone()
        elif self.cfg.command in ("absolute_integrated", "relative_integrated"):
            self._processed_actions = self.increment_toward(target) * self.cfg.scale
        self._fresh = True

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None or isinstance(env_ids, slice) else env_ids
        # The action manager resets after the reset events, so the measured
        # joints already hold the randomized reset state: start from there.
        self._target[ids] = self._asset.data.joint_pos.torch[ids][:, self._joint_ids]
        self._filtered[ids] = 0.0
        self._fresh = False


@configclass
class IntegratedJointPositionActionCfg(JointActionCfg):
    """Configuration for :class:`IntegratedJointPositionAction`.

    ``scale`` is the joint increment per env step for a unit action [rad].
    """

    class_type: type = IntegratedJointPositionAction
    smoothing: float = 1.0
    """Weight of the new action in the low-pass filter; 1.0 disables filtering."""
    max_position_error: float | None = None
    """Maximum integrated target error per joint [rad]; None keeps unbounded integration.

    Some error is needed to generate drive torque under load. The limit must
    be chosen against the task's required torque and validated in simulation.
    """
    command: str = "increment"
    """What the policy outputs, one of :data:`ARM_COMMANDS`; ``smoothing`` and ``max_position_error`` apply
    to the integrating ones only."""


ARM_COMMANDS = ("increment", "absolute_integrated", "absolute", "relative_integrated", "relative")
"""Arm command of :class:`IntegratedJointPositionAction`:

* ``increment``: target increments in units of ``scale``, the PPO policies' action (clip it to +-1).
* ``absolute_integrated``: absolute joint targets [rad], each turned into the clipped increment toward
  it, so the robot runs the same integrator, error bound and speed limit as ``increment``.
* ``absolute``: absolute joint targets [rad] sent straight to the drives, the scripted expert's
  joint-position control (no integrator, error bound or speed limit).
* ``relative_integrated`` / ``relative``: as the absolute ones, with the target given relative to the
  measured joint positions at the step (target = measured + action). The same targets, so the
  expert replays identically, but a policy must output the motion itself: cloned from absolute
  targets, it copies the measured joints (they are most of the target) and drifts in closed loop.

Select with ``env.actions.arm_action.command=<name>``; all but ``increment`` need ``agent.clip_actions=null``.
"""


def silence_solver_overflow_warnings(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Startup event: stop MJWarp printing one line per world on solver overflow.

    With thousands of worlds exploring contact-rich states, the per-world
    ``linesearch iterations limit reached`` printf floods stdout (hundreds of
    thousands of lines per minute) and stalls training. The overflow is still
    recorded by the solver; only the device-side printf is disabled. The flag is
    a compile-time kernel constant, so it must be set before the first step.
    """
    from isaaclab_newton.physics import NewtonManager

    solver = getattr(NewtonManager, "_solver", None)
    mjw_model = getattr(solver, "mjw_model", None)
    if mjw_model is None:
        print("[WARN] silence_solver_overflow_warnings: no MJWarp model; warnings stay on.")
        return
    mjw_model.opt.warn_overflow = False


##
# Geometry helpers
##


def reset_joints_from_postures(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    postures: dict[str, tuple[float, ...]],
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    flip_prob: float = 0.0,
    flip_joints: tuple[int, int, int] | None = None,
    flip_margin: float = 0.05,
) -> None:
    """Reset event: put ``asset_cfg``'s joints at one of ``postures`` (uniform pick) plus a uniform offset, at rest.

    Each posture lists the joints in ``asset_cfg.joint_ids`` order. With
    probability ``flip_prob`` the start is the posture's wrist-flipped twin:
    the (roll, pitch, roll) wrist joints at ``flip_joints`` (indices into the
    posture) turned by +/-pi (random way), negated and turned by pi, which
    keeps the hand where it is with the forearm rolled half a turn. Joints are
    then kept ``flip_margin`` inside their limits (which may roll the hand a
    little).
    """
    asset: BaseArticulation = env.scene[asset_cfg.name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    table = torch.tensor(list(postures.values()), device=env.device)
    pick = torch.randint(len(table), (len(ids),), device=env.device)
    q = table[pick] + torch.empty(len(ids), table.shape[1], device=env.device).uniform_(*position_range)
    if flip_prob > 0.0 and flip_joints is not None:
        roll1, pitch, roll2 = flip_joints
        flip = torch.rand(len(ids), device=env.device) < flip_prob
        way = torch.where(torch.rand(len(ids), device=env.device) < 0.5, 1.0, -1.0)
        limits = asset.data.joint_pos_limits.torch[ids][:, asset_cfg.joint_ids]
        low, high = limits[..., 0] + flip_margin, limits[..., 1] - flip_margin
        # The last roll turns by -pi: the grasp's flipped twin is there too
        # (the DynaArm's postures sit at +pi/2, its range is [-pi/2, 3pi/2]).
        r2 = q[:, roll2] - torch.pi
        q[:, roll1] = torch.where(flip, q[:, roll1] + way * torch.pi, q[:, roll1])
        q[:, pitch] = torch.where(flip, -q[:, pitch], q[:, pitch])
        q[:, roll2] = torch.where(flip, r2, q[:, roll2])
        q = torch.minimum(torch.maximum(q, low), high)
    asset.write_joint_position_to_sim_index(position=q, joint_ids=asset_cfg.joint_ids, env_ids=ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=asset_cfg.joint_ids, env_ids=ids)


def _body_frame_w(
    env: ManagerBasedEnv, asset_name: str, body_name: str, offset_pos, offset_rot
) -> tuple[torch.Tensor, torch.Tensor]:
    """World pose [m, xyzw] of a frame rigidly offset from an articulation body.

    Read from the articulation's body poses rather than the FrameTransformer
    sensors: on Newton those are not repopulated after a reset until the next
    physics step, so the first observation of every episode read a zero TCP
    and lever pose (off by the environment origin). The teacher's first actions
    then differed by up to 1.2 between identical states and some environments
    never recovered.
    """
    asset: BaseArticulation = env.scene[asset_name]
    body = asset.find_bodies(body_name)[0][0]
    pos, quat = asset.data.body_pos_w.torch[:, body], asset.data.body_quat_w.torch[:, body]
    off_pos = torch.tensor(offset_pos, device=env.device, dtype=pos.dtype).expand_as(pos)
    off_rot = torch.tensor(offset_rot, device=env.device, dtype=quat.dtype).expand_as(quat)
    return math_utils.combine_frame_transforms(pos, quat, off_pos, off_rot)


def _lever_frame_w(env: ManagerBasedEnv, frame_name: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """A ``target_frame`` target (default: the grasp frame, e.g. ``lever_pivot``) from its body and the scene's offset."""
    frame_cfg = next(
        f for f in env.scene.cfg.target_frame.target_frames if f.name == (frame_name or valve_task(env).grasp_frame)
    )
    body = frame_cfg.prim_path.rsplit("/", 1)[-1]
    return _body_frame_w(env, valve_task(env).asset_name, body, frame_cfg.offset.pos, frame_cfg.offset.rot)


def grasp_target_w(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert grasp TCP pose on the lever, in world frame [m], xyzw.

    The parallel-jaw grasp is symmetric under a half turn about the TCP's
    approach axis (+X), and the expert takes whichever of the two keeps the
    hand upright (``canonicalize_upward``), which depends on how the valve is
    clocked. The target is the one nearer the TCP's current orientation.
    """
    pos, quat = _lever_frame_w(env)
    task = valve_task(env)
    off_pos = torch.tensor(task.grasp_offset_pos, device=env.device).expand_as(pos)
    off_quat = torch.tensor(task.grasp_offset_quat, device=env.device).expand_as(quat)
    goal_pos, goal_quat = math_utils.combine_frame_transforms(pos, quat, off_pos, off_quat)
    flipped = math_utils.quat_mul(goal_quat, goal_quat.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(goal_quat))
    _, tcp_quat = tcp_w(env)
    nearer = math_utils.quat_error_magnitude(tcp_quat, flipped) < math_utils.quat_error_magnitude(tcp_quat, goal_quat)
    return goal_pos, torch.where(nearer[:, None], flipped, goal_quat)


def tcp_w(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """TCP pose in world frame [m], xyzw (``ANYMAL_EE``: gripper body plus TCP offset)."""
    from isaaclab_hiveboard.assets import ANYMAL_EE

    return _body_frame_w(env, "robot", ANYMAL_EE.body_name, ANYMAL_EE.tcp_offset.pos, ANYMAL_EE.tcp_offset.rot)


def tcp_grasp_error(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """TCP position [m] and orientation [rad] errors to the grasp target."""
    tcp_pos, tcp_quat = tcp_w(env)
    goal_pos, goal_quat = grasp_target_w(env)
    return torch.norm(tcp_pos - goal_pos, dim=-1), math_utils.quat_error_magnitude(tcp_quat, goal_quat)


def valve_angle(env: ManagerBasedEnv) -> torch.Tensor:
    """Valve joint angle [rad], shape ``(num_envs,)``."""
    return valve_asset(env).data.joint_pos.torch[:, valve_joint_id(env)]


def valve_progress(env: ManagerBasedEnv) -> torch.Tensor:
    """Opening progress in [0, 1]: 0 closed, 1 fully open."""
    return _progress_of(env, valve_angle(env)).clamp(0.0, 1.0)


def _gripper_closed(env: ManagerBasedEnv) -> torch.Tensor:
    """1 when the gripper action commands close, else 0."""
    return (env.action_manager.get_term("gripper_action").raw_actions[:, 0] < 0.0).float()


##
# Observations
##


def rot6d(quat: torch.Tensor) -> torch.Tensor:
    """xyzw orientations as the first two columns of their rotation matrix, ``(N, 6)``.

    Continuous over all orientations, unlike a quaternion with a fixed sign:
    ``quat_unique`` flips it where ``w`` crosses zero, which the valve's
    nominal 180-degree yaw sits on.
    """
    return math_utils.matrix_from_quat(quat)[..., :2].transpose(-1, -2).reshape(*quat.shape[:-1], 6)


def _in_base_quat(env: ManagerBasedEnv, pos_w: torch.Tensor, quat_w: torch.Tensor, env_ids=slice(None)):
    """A world pose of environments ``env_ids`` in their robot base frame, position and xyzw quaternion."""
    robot: BaseArticulation = env.scene["robot"]
    base_pos, base_quat = robot.data.root_pos_w.torch[env_ids], robot.data.root_quat_w.torch[env_ids]
    return math_utils.subtract_frame_transforms(base_pos, base_quat, pos_w, quat_w)


def _in_base(env: ManagerBasedEnv, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    """A world pose in the robot base frame as position (3) and :func:`rot6d` orientation (6)."""
    pos, quat = _in_base_quat(env, pos_w, quat_w)
    return torch.cat((pos, rot6d(quat)), dim=-1)


def tcp_pose_b(env: ManagerBasedEnv) -> torch.Tensor:
    """TCP pose in the base frame from forward kinematics, shape ``(N, 9)``."""
    return _in_base(env, *tcp_w(env))


def valve_pose_b(env: ManagerBasedEnv) -> torch.Tensor:
    """True valve root pose in the base frame (privileged), shape ``(N, 9)``."""
    valve = valve_asset(env)
    return _in_base(env, valve.data.root_pos_w.torch, valve.data.root_quat_w.torch)


def grasp_target_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Moving lever grasp target in the base frame (privileged), shape ``(N, 9)``."""
    return _in_base(env, *grasp_target_w(env))


def tcp_to_grasp_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Grasp target position minus TCP position, in the base frame [m] (privileged)."""
    return grasp_target_b(env)[:, :3] - tcp_pose_b(env)[:, :3]


def valve_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Opening progress and joint rate [rad/s] (privileged), shape ``(N, 2)``."""
    return torch.stack((valve_progress(env), valve_rate(env)), dim=-1)


def pad_valve_force(env: ManagerBasedEnv, sensor_names: tuple[str, ...] = ("finger_contact", "jaw_contact")):
    """Valve-filtered contact force magnitude on each pad [N] (privileged)."""
    forces = []
    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]
        matrix = sensor.data.force_matrix_w.torch.reshape(env.num_envs, -1, 3)
        forces.append(torch.norm(matrix.sum(dim=1), dim=-1))
    return torch.stack(forces, dim=-1)


def pad_force_split(
    env: ManagerBasedEnv, sensor_names: tuple[str, str] = ("finger_contact", "jaw_contact")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Valve contact force on the (left, right) pads split into squeeze and lateral parts, each ``(N, 2)`` [N].

    Squeeze is the part along the closing axis (inner finger to inner finger),
    which the gripper's motor sets; lateral is the rest, e.g. the arm pushing
    the lever through the fingers.
    """
    robot: BaseArticulation = env.scene["robot"]
    left, right = (robot.find_bodies(name)[0][0] for name in ("left_inner_finger", "right_inner_finger"))
    axis = robot.data.body_pos_w.torch[:, right] - robot.data.body_pos_w.torch[:, left]
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
    squeeze, lateral = [], []
    for name, sign in zip(sensor_names, (1.0, -1.0)):
        sensor: ContactSensor = env.scene[name]
        force = sensor.data.force_matrix_w.torch.reshape(env.num_envs, -1, 3).sum(dim=1)
        along = (force * axis).sum(dim=-1, keepdim=True)
        squeeze.append(sign * along[:, 0])
        lateral.append((force - along * axis).norm(dim=-1))
    return torch.stack(squeeze, dim=-1), torch.stack(lateral, dim=-1)


def episode_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time since the episode started, as a fraction of the episode length, ``(N, 1)`` (deployable).

    The robot knows when it started the task. The teacher's references (when to
    close, where the valve should be) are indexed by this time, so without it
    the student has to guess the phase from 5 steps of history.
    """
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.expert_bank import reference_step

    # Counted from the start of the task, also when an episode starts mid-trajectory.
    return (reference_step(env).float() / env.max_episode_length).unsqueeze(-1)


def gripper_object_detected(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    stopped_below: float = 0.745,
    moved_above: float = 0.3,
    max_speed: float = 0.05,
) -> torch.Tensor:
    """The gripper's "object detected" status, ``(N, 1)`` (deployable; no force sensing).

    Like the Robotiq's gOBJ: commanded closed, and the fingers have come to
    rest (``max_speed`` [rad/s]) short of the closed target, between
    ``moved_above`` and ``stopped_below`` [rad]. With the force-limited
    gripper (``env.py``) the finger joint closes onto its 0.785 rad limit on
    empty air and stops at ~0.703 rad on the lever. ``asset_cfg`` selects the
    finger joint.
    """
    robot: BaseArticulation = env.scene[asset_cfg.name]
    finger = robot.data.joint_pos.torch[:, asset_cfg.joint_ids[0]]
    speed = robot.data.joint_vel.torch[:, asset_cfg.joint_ids[0]].abs()
    closing = env.action_manager.get_term("gripper_action").raw_actions[:, 0] < 0.0
    detected = closing & (speed < max_speed) & (finger > moved_above) & (finger < stopped_below)
    return detected.float().unsqueeze(-1)


class registered_valve_b(ManagerTermBase):
    """The valve as a one-off board registration reports it (deployable).

    Registration happens once, at the start of the episode (an AprilTag or a
    touch-probe of the board, plus one look at the lever), so its result is
    latched at reset:

    * the valve root pose with a registration error (``bias_pos`` [m],
      ``bias_rot`` [rad] about each axis), plus ``jitter_pos`` [m] of white
      noise per step;
    * the lever's opening progress at registration, with ``angle_noise``
      [rad] of error. It is *not* updated afterwards: the student has to track
      the lever from its own proprioception once it holds it.

    Returns position (3), :func:`rot6d` orientation (6) and initial progress (1).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._pose = torch.zeros(env.num_envs, 9, device=env.device)
        self._progress = torch.zeros(env.num_envs, 1, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # The observation manager resets after the reset events, so the valve
        # root and joint state read here are the new episode's.
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None or isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, device=self.device)
        )
        if ids.numel() == 0:
            return
        params = self.cfg.params
        n = len(ids)
        valve = valve_asset(self._env)
        pos, quat = _in_base_quat(
            self._env, valve.data.root_pos_w.torch[ids], valve.data.root_quat_w.torch[ids], env_ids=ids
        )
        bias_pos = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * float(params.get("bias_pos", 0.0))
        euler = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * float(params.get("bias_rot", 0.0))
        bias_quat = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self._pose[ids, :3] = pos + bias_pos
        self._pose[ids, 3:] = rot6d(math_utils.quat_mul(bias_quat, quat))
        angle = valve_angle(self._env)[ids]
        angle = angle + (torch.rand(n, device=self.device) * 2.0 - 1.0) * float(params.get("angle_noise", 0.0))
        self._progress[ids, 0] = _progress_of(self._env, angle)

    def __call__(
        self,
        env: ManagerBasedEnv,
        bias_pos: float = 0.0,
        bias_rot: float = 0.0,
        jitter_pos: float = 0.0,
        angle_noise: float = 0.0,
    ) -> torch.Tensor:
        pose = self._pose.clone()
        pose[:, :3] += torch.randn_like(pose[:, :3]) * jitter_pos
        return torch.cat((pose, self._progress), dim=-1)


##
# Rewards
##


def reach_tcp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """``1 - tanh(d / std)`` on the TCP-to-grasp distance."""
    dist, _ = tcp_grasp_error(env)
    return 1.0 - torch.tanh(dist / std)


def align_tcp(env: ManagerBasedRLEnv, std: float, gate_dist: float) -> torch.Tensor:
    """Orientation alignment to the grasp, counted only within ``gate_dist`` [m]."""
    dist, ang = tcp_grasp_error(env)
    return (1.0 - torch.tanh(ang / std)) * (dist < gate_dist).float()


def lever_held(env: ManagerBasedEnv, dist_threshold: float, ang_threshold: float) -> torch.Tensor:
    """True while the gripper is closed at the expert grasp pose on the lever.

    Both position and orientation are required: a closed gripper near the
    lever at the wrong orientation is a fist pushing the lever, which the
    first teacher run learned to do (0.99 s opening, 0% grasps). With
    :attr:`ValveTaskCfg.hold_by_contact` both pads must touch the valve instead
    (the thresholds are then unused). With :attr:`ValveTaskCfg.push`, either pad touching it.
    """
    task = valve_task(env)
    if task.hold_by_contact or task.push:
        from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.expert_bank import CONTACT_FORCE_N

        touching = pad_valve_force(env) > CONTACT_FORCE_N
        if task.push:
            return touching.any(dim=-1)
        return touching.all(dim=-1) & _gripper_closed(env).bool()
    dist, ang = tcp_grasp_error(env)
    return (dist < dist_threshold) & (ang < ang_threshold) & _gripper_closed(env).bool()


def grasp_lever(env: ManagerBasedRLEnv, dist_threshold: float, ang_threshold: float) -> torch.Tensor:
    """1 while the gripper is closed on the lever at the grasp pose."""
    return lever_held(env, dist_threshold, ang_threshold).float()


def align_held(env: ManagerBasedRLEnv, std: float, dist_threshold: float, ang_threshold: float) -> torch.Tensor:
    """Tight orientation alignment [rad] while the lever is held.

    Teacher v3 held the lever at 0.34 rad, just inside the 0.35 rad hold gate,
    so its distilled student fell out of the gate on small imitation errors.
    A narrow kernel while holding pulls the grasp to the expert orientation.
    """
    _, ang = tcp_grasp_error(env)
    return (1.0 - torch.tanh(ang / std)) * lever_held(env, dist_threshold, ang_threshold).float()


def close_far_penalty(env: ManagerBasedRLEnv, dist_threshold: float) -> torch.Tensor:
    """1 while the gripper is closed away from the lever (closing on air)."""
    dist, _ = tcp_grasp_error(env)
    return (dist > dist_threshold).float() * _gripper_closed(env)


def valve_unheld_motion(env: ManagerBasedRLEnv, dist_threshold: float, ang_threshold: float) -> torch.Tensor:
    """Squared valve joint rate [rad^2/s^2] while the lever is *not* held.

    Teacher v2 struck the lever on its approach, flinging it half open at
    ~11 rad/s before grasping. Any valve motion before the grasp is an impact.
    """
    return valve_rate(env).square() * (~lever_held(env, dist_threshold, ang_threshold)).float()


def pad_force_excess(env: ManagerBasedRLEnv, max_force: float) -> torch.Tensor:
    """Squared pad-force excess over ``max_force`` [N], summed over both pads."""
    return (pad_valve_force(env) - max_force).clamp(min=0.0).square().sum(dim=-1)


##
# Terminations
##


def valve_open_success(env: ManagerBasedRLEnv, threshold_rad: float) -> torch.Tensor:
    """Success: the valve is within ``threshold_rad`` [rad] of fully open."""
    return _is_open(env, valve_angle(env), threshold_rad)


def released_and_clear(
    env: ManagerBasedEnv, threshold_rad: float, min_distance: float = 0.06, max_force: float = 1.0
) -> torch.Tensor:
    """The valve is open and the arm has let go and backed off.

    Open within ``threshold_rad`` [rad], gripper commanded open, no pad force
    above ``max_force`` [N] from the valve, and the TCP at least
    ``min_distance`` [m] from the grasp point (the expert retreats ~10 cm to
    its approach point). A pushed lever (:attr:`ValveTaskCfg.push`) is let go of with the hand still
    closed.
    """
    dist, _ = tcp_grasp_error(env)
    let_go = torch.ones_like(dist, dtype=torch.bool) if valve_task(env).push else _gripper_closed(env) == 0.0
    return (
        valve_open_success(env, threshold_rad)
        & let_go
        & (pad_valve_force(env).max(dim=-1).values < max_force)
        & (dist >= min_distance)
    )


def opening_credited(env: ManagerBasedEnv, held: torch.Tensor, ever_held: torch.Tensor) -> torch.Tensor:
    """Whether reaching open now counts as the task's: the valve is held now (:func:`lever_held`) or, for a
    pushed lever (:attr:`ValveTaskCfg.push`), was touched earlier in the episode (``ever_held``)."""
    return ever_held if valve_task(env).push else held


def invalid_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_joint_vel: float = 50.0) -> torch.Tensor:
    """Terminate on non-finite state or runaway joint rates [rad/s] (solver blow-ups)."""
    robot: BaseArticulation = env.scene["robot"]
    valve = valve_asset(env)
    arm_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    vel = torch.cat((arm_vel, valve.data.joint_vel.torch), dim=-1)
    nonfinite = (~torch.isfinite(robot.data.joint_pos.torch)).any(dim=-1) | (~torch.isfinite(vel)).any(dim=-1)
    return nonfinite | (vel.abs() > max_joint_vel).any(dim=-1)


##
# Turning-rate command
##


class ValveTurnRateCommand(CommandTerm):
    """Commanded valve turning speed and the reference angle it generates.

    Each episode samples a turning rate [rad/s] from ``rate_range``, or takes
    the expert's when the episode replays an expert bank that varies it. The
    reference angle starts at the valve's reset angle and stays there until the
    lever is first held (:func:`lever_held`), so it waits for the grasp instead
    of running away during the reach. From then on it advances toward open at
    that rate whether or not the lever is still held: an earlier version paused
    it whenever the grip drifted out of the hold gate, which froze all tracking
    reward and left the teacher stuck chattering the lever. Rewards track the
    reference, which removes the incentive to open as early as possible and
    makes the turning speed an explicit, operator-set input.

    ``command`` is ``(rate [rad/s], reference progress in [0, 1])``.
    """

    cfg: ValveTurnRateCommandCfg

    def __init__(self, cfg: ValveTurnRateCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.rate = torch.zeros(self.num_envs, device=self.device)
        self.ref_angle = torch.zeros(self.num_envs, device=self.device)
        self.engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.engaged_time = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_error_rad"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        ref_progress = _progress_of(self._env, self.ref_angle)
        return torch.stack((self.rate, ref_progress), dim=-1)

    @property
    def reference_done(self) -> torch.Tensor:
        """True once the reference has reached fully open."""
        return _is_open(self._env, self.ref_angle, 1.0e-6)

    def _update_metrics(self):
        self.metrics["tracking_error_rad"] = (valve_angle(self._env) - self.ref_angle).abs()

    def _resample_command(self, env_ids: Sequence[int] | slice):
        # The command manager resets after the reset events, so the valve angle
        # and expert-bank trajectory read here are the new episode's.
        self.rate[env_ids] = torch.empty_like(self.rate[env_ids]).uniform_(*self.cfg.rate_range)
        bank_term = getattr(self._env, "expert_bank_term", None)
        if (bank_term is not None and bank_term.bank.turn_rate is not None
                and not bank_term.cfg.params.get("standard", False)):
            # The expert's own turning speed: the references follow it.
            self.rate[env_ids] = bank_term.bank.turn_rate[bank_term.index[env_ids]]
        self.ref_angle[env_ids] = valve_angle(self._env)[env_ids]
        self.engaged[env_ids] = False
        self.engaged_time[env_ids] = 0.0

    @property
    def turning(self) -> torch.Tensor:
        """True while the reference is advancing (engaged and not yet open)."""
        return self.engaged & ~self.reference_done

    def _update_command(self):
        self.engaged |= lever_held(self._env, self.cfg.hold_dist, self.cfg.hold_ang)
        self.engaged_time += self._env.step_dt * self.engaged.float()
        # The reference starts once the gripper has had time to close.
        moving = self.engaged_time > self.cfg.engage_delay_s
        task = valve_task(self._env)
        sign = _opening_sign(self._env)
        self.ref_angle += sign * self.rate * self._env.step_dt * moving.float()
        # Clamped at open (negative rotation opens the HiveBoard valves).
        if sign < 0:
            self.ref_angle.clamp_(min=task.open_rad)
        else:
            self.ref_angle.clamp_(max=task.open_rad)


@configclass
class ValveTurnRateCommandCfg(CommandTermCfg):
    """Configuration for :class:`ValveTurnRateCommand`."""

    class_type: type = ValveTurnRateCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    rate_range: tuple[float, float] = (0.25, 0.8)
    """Turning rate sampled once per episode [rad/s]."""
    hold_dist: float = 0.03
    """TCP distance [m] within which the lever counts as held (see :func:`lever_held`)."""
    hold_ang: float = 0.35
    """TCP orientation error [rad] within which the lever counts as held."""
    engage_delay_s: float = 0.0
    """Time after the first hold before the reference starts turning [s]. The
    cuRobo expert idles ~0.5 s while its gripper closes; the policy is asked
    to turn right away instead (the turn's joint reference is indexed by valve
    angle, so the expert's pause does not appear in it)."""


def _turn_command(env: ManagerBasedEnv, command_name: str) -> ValveTurnRateCommand:
    return env.command_manager.get_term(command_name)


def turn_rate_command(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Commanded turning rate [rad/s], shape ``(N, 1)`` (deployable: the operator sets it)."""
    return _turn_command(env, command_name).rate.unsqueeze(-1)


def valve_reference_state(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Reference progress and tracking error [rad] (privileged), shape ``(N, 2)``."""
    term = _turn_command(env, command_name)
    error = valve_angle(env) - term.ref_angle
    return torch.stack((term.command[:, 1], error), dim=-1)


def track_valve_reference(
    env: ManagerBasedRLEnv, std: float, dist_threshold: float, ang_threshold: float, command_name: str = "valve_turn"
) -> torch.Tensor:
    """``exp(-(q - q_ref)^2 / std^2)`` while the lever is held; ``std`` [rad]."""
    error = valve_angle(env) - _turn_command(env, command_name).ref_angle
    held = lever_held(env, dist_threshold, ang_threshold).float()
    return torch.exp(-(error / std).square()) * held


def valve_opened_on_schedule(
    env: ManagerBasedRLEnv,
    threshold_rad: float,
    dist_threshold: float,
    ang_threshold: float,
    command_name: str = "valve_turn",
) -> torch.Tensor:
    """1 while the valve is open, held, and the reference has also reached open.

    Opening ahead of the reference earns nothing extra, so there is no reward
    for turning faster than commanded.
    """
    opened = _is_open(env, valve_angle(env), threshold_rad)
    on_schedule = _turn_command(env, command_name).reference_done
    return (opened & on_schedule & lever_held(env, dist_threshold, ang_threshold)).float()


def valve_rate_excess(env: ManagerBasedRLEnv, factor: float, command_name: str = "valve_turn") -> torch.Tensor:
    """Squared valve rate [rad/s] above ``factor`` times the commanded rate."""
    limit = factor * _turn_command(env, command_name).rate
    return (valve_rate(env).abs() - limit).clamp(min=0.0).square()


def valve_rate_deviation(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Squared deviation [rad^2/s^2] of the valve rate from the commanded rate while turning.

    Tracking the reference angle alone let the teacher chatter the lever at
    20 Hz around the reference (mean |rate| ~1 rad/s for a 0.3 rad/s command);
    this asks for a steady turn at the commanded speed.
    """
    term = _turn_command(env, command_name)
    return (valve_rate(env) - _opening_sign(env) * term.rate).square() * term.turning.float()
