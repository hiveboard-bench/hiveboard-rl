import math
import os
from dataclasses import MISSING, is_dataclass

import torch
import warp as wp
import yaml
from hydra.utils import get_object, instantiate
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.actuators.newton.adapter import read_group_parameter, write_group_parameter
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    ManagerTermBase,
    SceneEntityCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_hiveboard.assets import make_ee_frame
from isaaclab_hiveboard.mdp.actions import RateLimitedBinaryJointPositionActionCfg
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    ScrewJointCouplingCfg,
    register_screw_joint_mimic,
)
from isaaclab_hiveboard.mdp.events import (
    apply_articulation_gravcomp,
    set_contact_stiffness,
    set_joint_limit_damping,
)

from isaaclab_tasks.core.cabinet import mdp as base_mdp
from isaaclab_tasks.utils import PresetCfg

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONTACT_FORCE_N = 1.0
IDLE_STEP_RAD = 1.0e-3
DYNAMICS = ("friction", "damping", "spring", "breakaway", "armature")
BREAKAWAY_WIDTH_RAD = 0.15
SPRING_STATIC_RAD = 0.4
ARM_COMMANDS = ("increment", "absolute_integrated", "absolute", "relative_integrated", "relative")


@clone
def spawn_movable_panel(prim_path, cfg, translation=None, orientation=None):
    from pxr import UsdPhysics

    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd_file

    prim = spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)
    return prim


def reset_panel(env, env_ids, pos, rot):
    ids = all_ids(env, env_ids)
    pose = mechanism_asset(env).data.root_link_pose_w[ids]
    offset_pos = pose.new_tensor(pos).expand(len(ids), -1)
    offset_rot = pose.new_tensor(rot).expand(len(ids), -1)
    pos, rot = math_utils.combine_frame_transforms(
        pose[:, :3], pose[:, 3:7], offset_pos, offset_rot
    )
    pose = torch.cat((pos, rot), dim=-1)
    panel = env.scene["honeycomb"]
    panel.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    panel.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids
    )


@configclass
class MechanismCfg:
    asset_name: str = MISSING
    joint_name: str = MISSING
    closed_rad: float = MISSING
    open_rad: float = MISSING
    grasp_frame: str = MISSING
    closed_end_stop: bool = MISSING
    coupled_joint_name: str | None = None
    coupled_ratio: float = 0.0
    coupled_offset: float = 0.0
    mode: str = MISSING
    success_tolerance: float = MISSING
    hold_dist: float = 0.03
    hold_ang: float = 0.35
    screw: ScrewJointCouplingCfg | None = None


@configclass
class RobotCfg:
    name: str = MISSING
    arm_joint_names: tuple[str, ...] = MISSING
    hand_joint_names: tuple[str, ...] = MISSING
    finger_joint_names: tuple[str, ...] = MISSING
    finger_body_names: tuple[str, str] = MISSING
    tcp_body: str = MISSING
    tcp_pos: tuple[float, float, float] = MISSING
    tcp_rot: tuple[float, float, float, float] = MISSING
    home: tuple[float, ...] = MISSING
    actuator_groups: tuple[str, ...] = MISSING
    implicit: bool = MISSING
    diagnostic_joint: str = MISSING


def mechanism_asset(env):
    return env.scene[env.cfg.mechanism.asset_name]


def mechanism_joint_id(env):
    task = env.cfg.mechanism
    return env.scene[task.asset_name].find_joints(task.joint_name)[0][0]


def write_mechanism_position(env, env_ids, angle, rate=None):
    task = env.cfg.mechanism
    mechanism = mechanism_asset(env)
    rate = torch.zeros_like(angle) if rate is None else rate
    joints = [mechanism_joint_id(env)]
    pos, vel = angle[:, None], rate[:, None]
    if task.coupled_joint_name is not None:
        joints.append(mechanism.find_joints(task.coupled_joint_name)[0][0])
        pos = torch.cat((pos, task.coupled_offset + task.coupled_ratio * pos), dim=-1)
        vel = torch.cat((vel, task.coupled_ratio * vel), dim=-1)
    mechanism.write_joint_position_to_sim_index(
        position=pos.contiguous(), joint_ids=joints, env_ids=env_ids
    )
    mechanism.write_joint_velocity_to_sim_index(
        velocity=vel.contiguous(), joint_ids=joints, env_ids=env_ids
    )


def all_ids(env, env_ids):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def sync_coupled_joint(env, env_ids):
    ids = all_ids(env, env_ids)
    if env.cfg.mechanism.coupled_joint_name is not None:
        write_mechanism_position(env, ids, mechanism_position(env)[ids].clone())


def mechanism_rate(env):
    return mechanism_asset(env).data.joint_vel.torch[:, mechanism_joint_id(env)]


def _progress_of(env, angle):
    task = env.cfg.mechanism
    return (angle - task.closed_rad) / (task.open_rad - task.closed_rad)


def _opening_sign(env):
    task = env.cfg.mechanism
    return 1.0 if task.open_rad > task.closed_rad else -1.0


def _is_open(env, angle, threshold_rad):
    return _opening_sign(env) * (env.cfg.mechanism.open_rad - angle) <= threshold_rad


class IntegratedJointPositionAction(JointAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        assert cfg.command in ARM_COMMANDS, cfg.command
        assert cfg.max_position_error is None or 0 < cfg.max_position_error < math.inf
        self._fresh = False
        self._filtered = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._target = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._lower = limits[..., 0].clone()
        self._upper = limits[..., 1].clone()

    def increment_toward(self, target):
        return ((target - self._target) / self.cfg.scale).clamp(-1.0, 1.0)

    def measured(self):
        return self._asset.data.joint_pos.torch[:, self._joint_ids]

    def apply_actions(self):
        if self.cfg.command in ("absolute", "relative"):
            if self._fresh:
                torch.clamp(self.processed_actions, self._lower, self._upper, out=self._target)
                self._fresh = False
            self._asset.set_joint_position_target_index(
                target=self._target, joint_ids=self._joint_ids
            )
            return
        if self._fresh:
            alpha = self.cfg.smoothing
            self._filtered.mul_(1.0 - alpha).add_(self.processed_actions, alpha=alpha)
            self._target += self._filtered
            if self.cfg.max_position_error is not None:
                measured = self.measured()
                limit = self.cfg.max_position_error
                torch.clamp(self._target, measured - limit, measured + limit, out=self._target)
            torch.clamp(self._target, self._lower, self._upper, out=self._target)
            self._fresh = False
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def process_actions(self, actions):
        super().process_actions(actions)
        target = self._raw_actions
        if self.cfg.command.startswith("relative"):
            target = target + self.measured()
        if self.cfg.command in ("absolute", "relative"):
            self._processed_actions = target.clone()
        elif self.cfg.command in ("absolute_integrated", "relative_integrated"):
            self._processed_actions = self.increment_toward(target) * self.cfg.scale
        self._fresh = True

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None or isinstance(env_ids, slice) else env_ids
        self._target[ids] = self._asset.data.joint_pos.torch[ids][:, self._joint_ids]
        self._filtered[ids] = 0.0
        self._fresh = False


@configclass
class IntegratedJointPositionActionCfg(JointActionCfg):
    class_type: type = IntegratedJointPositionAction
    smoothing: float = 1.0
    max_position_error: float | None = None
    command: str = "increment"


def silence_solver_overflow_warnings(env, env_ids):
    NewtonManager._solver.mjw_model.opt.warn_overflow = False


def reset_joints_from_postures(env, env_ids, postures, position_range, asset_cfg):
    asset = env.scene[asset_cfg.name]
    ids = all_ids(env, env_ids)
    table = torch.tensor(list(postures.values()), device=env.device)
    pick = torch.randint(len(table), (len(ids),), device=env.device)
    noise = torch.empty(len(ids), table.shape[1], device=env.device).uniform_(*position_range)
    limits = asset.data.joint_pos_limits.torch[ids][:, asset_cfg.joint_ids]
    q = (table[pick] + noise).clamp(min=limits[..., 0], max=limits[..., 1])
    asset.write_joint_position_to_sim_index(position=q, joint_ids=asset_cfg.joint_ids, env_ids=ids)
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(q), joint_ids=asset_cfg.joint_ids, env_ids=ids
    )


def _body_frame_w(env, asset_name, body_name, offset_pos, offset_rot):
    asset = env.scene[asset_name]
    body = asset.find_bodies(body_name)[0][0]
    pos, quat = asset.data.body_pos_w.torch[:, body], asset.data.body_quat_w.torch[:, body]
    off_pos = torch.tensor(offset_pos, device=env.device, dtype=pos.dtype).expand_as(pos)
    off_rot = torch.tensor(offset_rot, device=env.device, dtype=quat.dtype).expand_as(quat)
    return math_utils.combine_frame_transforms(pos, quat, off_pos, off_rot)


def grasp_target_w(env):
    task = env.cfg.mechanism
    frame_cfg = next(
        f for f in env.scene.cfg.target_frame.target_frames if f.name == task.grasp_frame
    )
    body = frame_cfg.prim_path.rsplit("/", 1)[-1]
    goal_pos, goal_quat = _body_frame_w(
        env, task.asset_name, body, frame_cfg.offset.pos, frame_cfg.offset.rot
    )
    flip = goal_quat.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(goal_quat)
    flipped = math_utils.quat_mul(goal_quat, flip)
    _, tcp_quat = tcp_w(env)
    error = math_utils.quat_error_magnitude
    nearer = error(tcp_quat, flipped) < error(tcp_quat, goal_quat)
    return goal_pos, torch.where(nearer[:, None], flipped, goal_quat)


def tcp_w(env):
    cfg = env.cfg.robot
    return _body_frame_w(env, "robot", cfg.tcp_body, cfg.tcp_pos, cfg.tcp_rot)


def tcp_grasp_error(env):
    tcp_pos, tcp_quat = tcp_w(env)
    goal_pos, goal_quat = grasp_target_w(env)
    dist = torch.norm(tcp_pos - goal_pos, dim=-1)
    return dist, math_utils.quat_error_magnitude(tcp_quat, goal_quat)


def mechanism_position(env):
    return mechanism_asset(env).data.joint_pos.torch[:, mechanism_joint_id(env)]


def mechanism_progress(env):
    return _progress_of(env, mechanism_position(env)).clamp(0.0, 1.0)


def _gripper_closed(env):
    return (env.action_manager.get_term("gripper_action").raw_actions[:, 0] < 0.0).float()


def rot6d(quat):
    matrix = math_utils.matrix_from_quat(quat)
    return matrix[..., :2].transpose(-1, -2).reshape(*quat.shape[:-1], 6)


def _in_base_quat(env, pos_w, quat_w, env_ids=slice(None)):
    data = env.scene["robot"].data
    base_pos, base_quat = data.root_pos_w.torch[env_ids], data.root_quat_w.torch[env_ids]
    return math_utils.subtract_frame_transforms(base_pos, base_quat, pos_w, quat_w)


def _in_base(env, pos_w, quat_w):
    pos, quat = _in_base_quat(env, pos_w, quat_w)
    return torch.cat((pos, rot6d(quat)), dim=-1)


def tcp_pose_b(env):
    return _in_base(env, *tcp_w(env))


def mechanism_pose_b(env):
    mechanism = mechanism_asset(env)
    return _in_base(env, mechanism.data.root_pos_w.torch, mechanism.data.root_quat_w.torch)


def grasp_target_b(env):
    return _in_base(env, *grasp_target_w(env))


def tcp_to_grasp_b(env):
    return grasp_target_b(env)[:, :3] - tcp_pose_b(env)[:, :3]


def mechanism_state(env):
    return torch.stack((mechanism_progress(env), mechanism_rate(env)), dim=-1)


def pad_force(env, sensor_names=("finger_contact", "jaw_contact")):
    forces = []
    for name in sensor_names:
        matrix = env.scene[name].data.force_matrix_w.torch.reshape(env.num_envs, -1, 3)
        forces.append(torch.norm(matrix.sum(dim=1), dim=-1))
    return torch.stack(forces, dim=-1)


def pad_force_split(env, sensor_names=("finger_contact", "jaw_contact")):
    robot = env.scene["robot"]
    left, right = (robot.find_bodies(name)[0][0] for name in env.cfg.robot.finger_body_names)
    axis = robot.data.body_pos_w.torch[:, right] - robot.data.body_pos_w.torch[:, left]
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
    squeeze, lateral = [], []
    for name, sign in zip(sensor_names, (1.0, -1.0)):
        force = env.scene[name].data.force_matrix_w.torch.reshape(env.num_envs, -1, 3).sum(dim=1)
        along = (force * axis).sum(dim=-1, keepdim=True)
        squeeze.append(sign * along[:, 0])
        lateral.append((force - along * axis).norm(dim=-1))
    return torch.stack(squeeze, dim=-1), torch.stack(lateral, dim=-1)


def episode_time(env):
    return (reference_step(env).float() / env.max_episode_length).unsqueeze(-1)


def gripper_object_detected(env, asset_cfg, stopped_below, moved_above, max_speed):
    robot = env.scene[asset_cfg.name]
    finger = robot.data.joint_pos.torch[:, asset_cfg.joint_ids[0]]
    speed = robot.data.joint_vel.torch[:, asset_cfg.joint_ids[0]].abs()
    closing = env.action_manager.get_term("gripper_action").raw_actions[:, 0] < 0.0
    detected = closing & (speed < max_speed) & (finger > moved_above) & (finger < stopped_below)
    return detected.float().unsqueeze(-1)


class registered_mechanism_b(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._pose = torch.zeros(env.num_envs, 9, device=env.device)
        self._progress = torch.zeros(env.num_envs, 1, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device)
        if ids.numel() == 0:
            return
        params = self.cfg.params
        n = len(ids)
        root = mechanism_asset(self._env).data
        pos, quat = _in_base_quat(
            self._env, root.root_pos_w.torch[ids], root.root_quat_w.torch[ids], env_ids=ids
        )
        bias_pos = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * params["bias_pos"]
        euler = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * params["bias_rot"]
        bias_quat = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self._pose[ids, :3] = pos + bias_pos
        self._pose[ids, 3:] = rot6d(math_utils.quat_mul(bias_quat, quat))
        noise = (torch.rand(n, device=self.device) * 2.0 - 1.0) * params["angle_noise"]
        self._progress[ids, 0] = _progress_of(self._env, mechanism_position(self._env)[ids] + noise)

    def __call__(self, env, bias_pos, bias_rot, jitter_pos, angle_noise):
        pose = self._pose.clone()
        pose[:, :3] += torch.randn_like(pose[:, :3]) * jitter_pos
        return torch.cat((pose, self._progress), dim=-1)


def held(env):
    task = env.cfg.mechanism
    if task.mode != "grasp":
        touching = pad_force(env) > CONTACT_FORCE_N
        if task.mode == "push":
            return touching.any(dim=-1)
        return touching.all(dim=-1) & _gripper_closed(env).bool()
    dist, ang = tcp_grasp_error(env)
    return (dist < task.hold_dist) & (ang < task.hold_ang) & _gripper_closed(env).bool()


def pad_force_excess(env, max_force):
    return (pad_force(env) - max_force).clamp(min=0.0).square().sum(dim=-1)


def succeeded(env):
    return _is_open(env, mechanism_position(env), env.cfg.mechanism.success_tolerance)


def released_and_clear(env, min_distance=0.06, max_force=1.0):
    dist, _ = tcp_grasp_error(env)
    if env.cfg.mechanism.mode == "push":
        let_go = torch.ones_like(dist, dtype=torch.bool)
    else:
        let_go = _gripper_closed(env) == 0.0
    return (
        succeeded(env)
        & let_go
        & (pad_force(env).max(dim=-1).values < max_force)
        & (dist >= min_distance)
    )


def opening_credited(env, held, ever_held):
    return ever_held if env.cfg.mechanism.mode == "push" else held


def invalid_state(env, asset_cfg, max_joint_vel):
    robot = env.scene["robot"]
    arm_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    vel = torch.cat((arm_vel, mechanism_asset(env).data.joint_vel.torch), dim=-1)
    nonfinite = ~torch.isfinite(robot.data.joint_pos.torch).all(dim=-1)
    nonfinite |= ~torch.isfinite(vel).all(dim=-1)
    return nonfinite | (vel.abs() > max_joint_vel).any(dim=-1)


class ReferenceCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.rate = torch.zeros(self.num_envs, device=self.device)
        self.ref_angle = torch.zeros(self.num_envs, device=self.device)
        self.engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.engaged_time = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_error_rad"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return torch.stack((self.rate, _progress_of(self._env, self.ref_angle)), dim=-1)

    @property
    def reference_done(self):
        return _is_open(self._env, self.ref_angle, 1.0e-6)

    def _update_metrics(self):
        self.metrics["tracking_error_rad"] = (mechanism_position(self._env) - self.ref_angle).abs()

    def _resample_command(self, env_ids):
        self.rate[env_ids] = torch.empty_like(self.rate[env_ids]).uniform_(*self.cfg.rate_range)
        bank_term = self._env.expert_bank_term
        if not bank_term.cfg.params["standard"]:
            self.rate[env_ids] = bank_term.bank.turn_rate[bank_term.index[env_ids]]
        self.ref_angle[env_ids] = mechanism_position(self._env)[env_ids]
        self.engaged[env_ids] = False
        self.engaged_time[env_ids] = 0.0

    def _update_command(self):
        self.engaged |= held(self._env)
        self.engaged_time += self._env.step_dt * self.engaged.float()
        moving = self.engaged_time > self.cfg.engage_delay_s
        sign = _opening_sign(self._env)
        self.ref_angle += sign * self.rate * self._env.step_dt * moving.float()
        if sign < 0:
            self.ref_angle.clamp_(min=self._env.cfg.mechanism.open_rad)
        else:
            self.ref_angle.clamp_(max=self._env.cfg.mechanism.open_rad)


@configclass
class ReferenceCommandCfg(CommandTermCfg):
    class_type: type = ReferenceCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    rate_range: tuple[float, float] = MISSING
    engage_delay_s: float = MISSING


def reference_rate(env):
    return env.command_manager.get_term("reference").rate.unsqueeze(-1)


def reference_state(env):
    term = env.command_manager.get_term("reference")
    error = mechanism_position(env) - term.ref_angle
    return torch.stack((term.command[:, 1], error), dim=-1)


def rate_excess(env, factor):
    limit = factor * env.command_manager.get_term("reference").rate
    return (mechanism_rate(env).abs() - limit).clamp(min=0.0).square()


class randomize_dynamics(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        task = env.cfg.mechanism
        ranges = cfg.params["ranges"]
        assert task.closed_end_stop or (
            ranges["spring"][1] == ranges["breakaway"][1] == cfg.params["stuck_prob"] == 0.0
        ), f"'{task.asset_name}' has no end stop at closed for a spring or seat torque"
        self.mechanism = env.scene[task.asset_name]
        self.joint_ids = self.mechanism.find_joints(task.joint_name)[0]
        self.closed_rad = task.closed_rad
        self.params = torch.zeros(env.num_envs, len(DYNAMICS), device=env.device)
        stuck_breakaway = cfg.params["stuck_breakaway"]
        scale = [
            max(ranges[name][1], stuck_breakaway[1] if name == "breakaway" else 0.0) or 1.0
            for name in DYNAMICS
        ]
        self.scale = torch.tensor(scale, device=env.device)
        env.dynamics_term = self

    def __call__(self, env, env_ids, ranges, stuck_prob, stuck_breakaway):
        ids = all_ids(env, env_ids)
        values = torch.empty(len(ids), len(DYNAMICS), device=env.device)
        for i, name in enumerate(DYNAMICS):
            values[:, i].uniform_(*ranges[name])
        values[:, 2] = torch.minimum(values[:, 2], values[:, 0] / SPRING_STATIC_RAD)
        stuck = torch.rand(len(ids), device=env.device) < stuck_prob
        if stuck.any():
            n = int(stuck.sum())
            values[stuck, 3] = torch.empty(n, device=env.device).uniform_(*stuck_breakaway)
            closed = torch.full((n, 1), self.closed_rad, device=env.device)
            self.mechanism.write_joint_position_to_sim_index(
                position=closed, joint_ids=self.joint_ids, env_ids=ids[stuck]
            )
            self.mechanism.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(closed), joint_ids=self.joint_ids, env_ids=ids[stuck]
            )
        self.write(ids, values)

    def write(self, env_ids, values):
        self.params[env_ids] = values
        self.mechanism.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=values[:, 0:1].contiguous(),
            joint_viscous_friction_coeff=values[:, 1:2].contiguous(),
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )
        self.mechanism.write_joint_armature_to_sim_index(
            armature=values[:, 4:5].contiguous(), joint_ids=self.joint_ids, env_ids=env_ids
        )


def dynamics_obs(env):
    return env.dynamics_term.params / env.dynamics_term.scale


class LoadAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        task = env.cfg.mechanism
        assert task.asset_name == cfg.asset_name, (cfg.asset_name, task.asset_name)
        self._joint_ids = self._asset.find_joints(task.joint_name)[0]
        self._closed_rad = task.closed_rad
        self._opening_sign = _opening_sign(env)
        self._empty = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._empty

    @property
    def processed_actions(self):
        return self._empty

    def process_actions(self, actions):
        pass

    def apply_actions(self):
        term = self._env.dynamics_term
        angle = self._asset.data.joint_pos.torch[:, self._joint_ids[0]]
        spring, breakaway, sign = term.params[:, 2], term.params[:, 3], self._opening_sign
        opening = (sign * (angle - self._closed_rad)).clamp(min=0.0)
        width = (1.0 - opening / BREAKAWAY_WIDTH_RAD).clamp(min=0.0)
        torque = -sign * (spring * opening + breakaway * width)
        self._asset.set_joint_effort_target_index(target=torque[:, None], joint_ids=self._joint_ids)


@configclass
class LoadActionCfg(ActionTermCfg):
    class_type: type = LoadAction
    asset_name: str = MISSING


def delayed_actuators(robot, groups, max_delay_s, physics_dt):
    max_delay = math.ceil(round(max_delay_s / physics_dt, 6))
    actuators = dict(robot.actuators)
    for name in groups:
        assert isinstance(actuators[name], IdealPDActuatorCfg), name
        fields = actuators[name].to_dict()
        for key in ("class_type", "min_delay", "max_delay"):
            fields.pop(key, None)
        actuators[name] = DelayedPDActuatorCfg(**fields, min_delay=0, max_delay=max_delay)
    return robot.replace(actuators=actuators)


def randomize_actuator_delay(
    env, env_ids, delay_range_s, groups, asset_cfg=SceneEntityCfg("robot")
):
    asset = env.scene[asset_cfg.name]
    ids = all_ids(env, env_ids)
    low = math.floor(round(delay_range_s[0] / env.physics_dt, 6))
    high = math.ceil(round(delay_range_s[1] / env.physics_dt, 6))
    steps = torch.randint(low, high + 1, (len(ids),), device=env.device)
    for name in groups:
        num_joints = read_group_parameter(asset.actuators, name, "delay", "delay_steps").shape[1]
        delay = steps.clamp(max=asset.cfg.actuators[name].max_delay)
        values = delay[:, None].expand(-1, num_joints)
        write_group_parameter(asset.actuators, name, "delay", "delay_steps", values, env_ids=ids)


class ExpertBank:
    def __init__(self, path, device, task, robot, dt):
        bank = torch.load(path, map_location="cpu", weights_only=False)
        meta = bank["meta"]
        assert meta["task"] == task, (path, meta["task"], task)
        assert meta["robot"] == robot.name, path
        assert list(meta["arm_joint_names"]) == list(robot.arm_joint_names), path
        assert list(meta["gripper_joint_names"]) == list(robot.hand_joint_names), path
        assert abs(float(meta["dt"]) - dt) <= 1e-7, (path, meta["dt"], dt)
        self.meta = meta
        self.size = len(bank["arm_q"])
        q = bank["arm_q"].float()
        phases = list(meta["phases"])
        phase = bank["phase_id"]
        gripping = (phase == phases.index("grip")) | (phase == phases.index("actuate"))
        assert bool(gripping.any(dim=1).all()), path
        grasp = gripping.float().argmax(dim=1)
        open_step = (bank["t_open"] / meta["dt"]).round().long().clamp(max=q.shape[1] - 1)
        moving = (q[:, 1:] - q[:, :-1]).abs().amax(dim=-1) > IDLE_STEP_RAD
        last_move = q.shape[1] - 1 - moving.flip(dims=[1]).float().argmax(dim=1)
        target = bank["arm_target"].float()
        paths = []
        for i in range(self.size):
            path = target[i, : int(grasp[i]) + 1]
            moving = (path[1:] - path[:-1]).abs().amax(dim=-1) > IDLE_STEP_RAD
            paths.append(torch.cat((path[:1], path[1:][moving])))
        horizon = max(len(path) for path in paths)
        self.reach = torch.stack(
            [torch.cat((p, p[-1:].expand(horizon - len(p), -1))) for p in paths]
        )
        self.grasp_step = torch.tensor([len(p) - 1 for p in paths])
        self.arm_target = target
        self.gripper_cmd = bank["gripper_cmd"].float()
        self.q = q
        self.position = bank["mechanism_position"].float()
        self.gripper_q = bank["gripper_q"].float()
        self.gripper_closed = bank["gripper_cmd"] < 0
        self.contact = (bank["pad_force"].float() > CONTACT_FORCE_N).float()
        self.gripper_joint_pos = bank["gripper_joint_pos"].float()
        self.dt = float(meta["dt"])
        self.idle_shift = grasp - self.grasp_step
        self.open_step = open_step
        self.retreat_step = torch.maximum(last_move, open_step)
        self.arm_q0 = bank["arm_q0"].float()
        self.position0 = bank["mechanism_position0"].float()
        self.pose_env = bank["mechanism_pose_env"].float()
        assert list(meta["dynamics"]) == list(DYNAMICS), path
        self.dynamics = bank["dynamics"].float()
        column = list(meta["expert_diversity"]).index("turn_rate")
        self.turn_rate = bank["expert_diversity"][:, column].float()
        for name, value in list(vars(self).items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))

    def _bank_step(self, idx, step):
        shifted = torch.where(step >= self.grasp_step[idx], step + self.idle_shift[idx], step)
        return shifted.clamp(max=self.q.shape[1] - 1)

    def mechanism_reference(self, idx, step):
        return self.position[idx, self._bank_step(idx, step)]

    def gripper_reference(self, idx, step):
        return self.gripper_q[idx, self._bank_step(idx, step)]

    def gripper_closed_reference(self, idx, step):
        return self.gripper_closed[idx, self._bank_step(idx, step)]

    def contact_reference(self, idx, step):
        return self.contact[idx, self._bank_step(idx, step)]

    def joint_reference(self, idx, step):
        reach = self.reach[idx, step.clamp(max=self.reach.shape[1] - 1)]
        turn = self.q[idx, self._bank_step(idx, step)]
        return torch.where((step >= self.grasp_step[idx])[:, None], turn, reach)

    def velocity_reference(self, idx, step):
        previous = (step - 1).clamp(min=0)
        following = step + 1
        delta = self.joint_reference(idx, following) - self.joint_reference(idx, previous)
        velocity = delta / ((following - previous).float()[:, None] * self.dt)
        held = (step >= self.grasp_step[idx]) & (self._bank_step(idx, step) >= self.q.shape[1] - 1)
        return torch.where(held[:, None], torch.zeros_like(velocity), velocity)


class reset_from_expert_bank(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.cfg.robot
        self.bank = ExpertBank(
            cfg.params["path"], env.device, cfg.params["task"], self.robot, env.step_dt
        )
        self.index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.start_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        robot = env.scene["robot"]
        hand, arm = list(self.robot.hand_joint_names), list(self.robot.arm_joint_names)
        self._hand_ids = robot.find_joints(hand, preserve_order=True)[0]
        self._arm_ids = robot.find_joints(arm, preserve_order=True)[0]
        self._finger_id = robot.find_joints(env.cfg.actions.gripper_action.joint_names)[0][0]
        self._mechanism_name = env.cfg.mechanism.asset_name
        env.expert_bank_term = self
        print(f"[INFO] Expert bank: {self.bank.size} trajectories from {cfg.params['path']}")

    def __call__(self, env, env_ids, path, task, mid_start_prob, standard):
        ids = all_ids(env, env_ids)
        if standard:
            self.index[ids] = 0
            self.start_step[ids] = 0
            home = torch.tensor(self.robot.home, device=env.device).expand(len(ids), -1)
            env.scene["robot"].write_joint_position_to_sim_index(
                position=home, joint_ids=self._arm_ids, env_ids=ids
            )
            closed = torch.full((len(ids),), env.cfg.mechanism.closed_rad, device=env.device)
            write_mechanism_position(env, ids, closed)
            return
        draw = torch.randint(0, self.bank.size, (len(ids),), device=env.device)
        self.write_start_state(env, ids, draw)
        if mid_start_prob > 0.0:
            mid = torch.rand(len(ids), device=env.device) < mid_start_prob
            if torch.any(mid):
                last = self.bank.retreat_step[draw[mid]] - self.bank.idle_shift[draw[mid]]
                step = (torch.rand(int(mid.sum()), device=env.device) * (last + 1).float()).long()
                self.write_mid_state(env, ids[mid], draw[mid], step)

    def write_start_state(self, env, ids, draw):
        self.index[ids] = draw
        self.start_step[ids] = 0
        robot = env.scene["robot"]
        mechanism = env.scene[self._mechanism_name]
        pose = self.bank.pose_env[draw].clone()
        pose[:, :3] += env.scene.env_origins[ids]
        mechanism.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        mechanism.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids
        )
        write_mechanism_position(env, ids, self.bank.position0[draw])
        env.dynamics_term.write(ids, self.bank.dynamics[draw])
        robot.write_joint_position_to_sim_index(
            position=self.bank.arm_q0[draw], joint_ids=self._arm_ids, env_ids=ids
        )
        robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros(len(ids), len(self._arm_ids), device=env.device),
            joint_ids=self._arm_ids,
            env_ids=ids,
        )

    def write_mid_state(self, env, ids, draw, step):
        bank = self.bank
        self.start_step[ids] = step
        b = bank._bank_step(draw, step)
        prev, nxt = (b - 1).clamp(min=0), (b + 1).clamp(max=bank.q.shape[1] - 1)
        span = ((nxt - prev).clamp(min=1) * bank.dt)[:, None]
        robot = env.scene["robot"]
        robot.write_joint_position_to_sim_index(
            position=bank.q[draw, b], joint_ids=self._arm_ids, env_ids=ids
        )
        robot.write_joint_velocity_to_sim_index(
            velocity=(bank.q[draw, nxt] - bank.q[draw, prev]) / span,
            joint_ids=self._arm_ids,
            env_ids=ids,
        )
        robot.write_joint_position_to_sim_index(
            position=bank.gripper_joint_pos[draw, b], joint_ids=self._hand_ids, env_ids=ids
        )
        robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros(len(ids), len(self._hand_ids), device=env.device),
            joint_ids=self._hand_ids,
            env_ids=ids,
        )
        rate = (bank.position[draw, nxt] - bank.position[draw, prev]) / span[:, 0]
        write_mechanism_position(env, ids, bank.position[draw, b], rate)


class reset_from_expert_bank_in_order(reset_from_expert_bank):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        gen = torch.Generator().manual_seed(cfg.params["seed"])
        self.order = torch.randperm(self.bank.size, generator=gen).to(env.device)
        self.cursor = 0

    def __call__(self, env, env_ids, path, task, mid_start_prob, standard, seed):
        ids = all_ids(env, env_ids)
        picks = (self.cursor + torch.arange(len(ids), device=env.device)) % len(self.order)
        self.cursor += len(ids)
        self.write_start_state(env, ids, self.order[picks])


def reference_step(env):
    return env.episode_length_buf + env.expert_bank_term.start_step


def expert_action(env):
    term = env.expert_bank_term
    bank = term.bank
    step = bank._bank_step(term.index, term.start_step) + env.episode_length_buf
    step = step.clamp(max=bank.arm_target.shape[1] - 1)
    target = bank.arm_target[term.index, step]
    arm = env.action_manager.get_term("arm_action")
    if arm.cfg.command == "increment":
        target = arm.increment_toward(target)
    elif arm.cfg.command.startswith("relative"):
        target = target - arm.measured()
    return torch.cat((target, bank.gripper_cmd[term.index, step, None]), dim=-1)


def expert_joint_error(env):
    term = env.expert_bank_term
    joint_pos = env.scene["robot"].data.joint_pos.torch[:, term._arm_ids]
    return joint_pos - term.bank.joint_reference(term.index, reference_step(env))


def track_expert_joints(env, std):
    return torch.exp(-expert_joint_error(env).square().sum(dim=-1) / std**2)


def track_expert_velocity(env, std):
    term = env.expert_bank_term
    reference = term.bank.velocity_reference(term.index, reference_step(env))
    measured = env.scene["robot"].data.joint_vel.torch[:, term._arm_ids]
    return torch.exp(-(measured - reference).square().mean(dim=-1) / std**2)


def expert_mechanism_error(env):
    term = env.expert_bank_term
    reference = term.bank.mechanism_reference(term.index, reference_step(env))
    return (mechanism_position(env) - reference).unsqueeze(-1)


def track_expert_mechanism(env, std):
    return torch.exp(-expert_mechanism_error(env)[:, 0].square() / std**2)


def expert_gripper_reference(env, follow_release, follow_reach):
    term = env.expert_bank_term
    step = reference_step(env)
    if follow_reach:
        return term.bank.gripper_closed_reference(term.index, step).float().unsqueeze(-1)
    closed = step >= term.bank.grasp_step[term.index]
    if follow_release:
        closed &= term.bank.gripper_closed_reference(term.index, step)
    return closed.float().unsqueeze(-1)


def track_expert_gripper(env, std):
    term = env.expert_bank_term
    finger = env.scene["robot"].data.joint_pos.torch[:, term._finger_id]
    error = finger - term.bank.gripper_reference(term.index, reference_step(env))
    return torch.exp(-error.square() / std**2)


def track_expert_contact(env):
    term = env.expert_bank_term
    policy = (pad_force(env) > CONTACT_FORCE_N).float()
    expert = term.bank.contact_reference(term.index, reference_step(env))
    return (policy == expert).float().mean(dim=-1)


def expert_joint_drift(env, max_error):
    return torch.linalg.vector_norm(expert_joint_error(env), dim=-1) > max_error


def expert_mechanism_lag(env, max_error):
    return expert_mechanism_error(env)[:, 0].abs() > max_error


@wp.kernel
def _delay_targets(
    desired: wp.array2d(dtype=wp.float32),
    backend: wp.array2d(dtype=wp.float32),
    arm: wp.array(dtype=wp.int32),
    mapped: wp.array(dtype=wp.int32),
    history: wp.array3d(dtype=wp.float32),
    cursor: wp.array2d(dtype=wp.int32),
    steps: wp.array(dtype=wp.int32),
    length: int,
):
    e, j = wp.tid()
    c = cursor[e, j]
    history[e, c, j] = desired[e, arm[j]]
    read = (c - steps[e] + length) % length
    backend[e, mapped[j]] = history[e, read, j]
    cursor[e, j] = (c + 1) % length


class TargetDelay(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        joints = list(env.cfg.robot.arm_joint_names)
        self.arm_ids = self.robot.find_joints(joints, preserve_order=True)[0]
        self.length = math.ceil(0.04 / env.physics_dt) + 1
        self.history = torch.zeros(env.num_envs, self.length, len(self.arm_ids), device=env.device)
        self.cursor = torch.zeros(
            env.num_envs, len(self.arm_ids), device=env.device, dtype=torch.int32
        )
        self.steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        arm = torch.tensor(self.arm_ids, dtype=torch.int32, device=env.device)
        if self.robot.data.has_joint_ordering:
            mapping = wp.to_torch(self.robot._joint_user_to_backend_map())
            mapped = mapping[arm.long()].to(torch.int32).contiguous()
        else:
            mapped = arm.clone()
        self._owners = (arm, mapped)
        self._arm, self._mapped = (wp.from_torch(v, dtype=wp.int32) for v in self._owners)
        self._history = wp.from_torch(self.history)
        self._cursor = wp.from_torch(self.cursor, dtype=wp.int32)
        self._steps = wp.from_torch(self.steps, dtype=wp.int32)
        self._device = env.device
        NewtonManager.register_post_actuator_callback(self._apply)

    def _apply(self):
        data = self.robot.data
        wp.launch(
            _delay_targets,
            dim=(self.history.shape[0], len(self.arm_ids)),
            inputs=[
                self.robot.actuators._joint_pos_target,
                data._sim_bind_joint_position_target,
                self._arm,
                self._mapped,
                self._history,
                self._cursor,
                self._steps,
                self.length,
            ],
            device=self._device,
        )
        if data._joint_targets_need_conversion:
            from isaaclab_newton.assets.articulation.joint_coordinates import (
                scatter_joint_coordinates,
            )

            scatter_joint_coordinates(
                data._joint_coord_map,
                data._sim_bind_joint_position_target,
                data._sim_bind_joint_target_coords,
                self.robot._ALL_ENV_MASK,
            )

    def __call__(self, env, env_ids, delay_range_s):
        low, high = delay_range_s
        assert 0 <= low <= high <= 0.04, delay_range_s
        ids = slice(None) if env_ids is None else env_ids
        self.steps[ids] = torch.randint(
            math.floor(low / env.physics_dt),
            math.ceil(high / env.physics_dt) + 1,
            (self.steps[ids].numel(),),
            device=env.device,
            dtype=torch.int32,
        )
        self.history[ids] = self.robot.data.joint_pos.torch[ids][:, self.arm_ids][:, None, :]
        self.cursor[ids] = 0


@configclass
class PhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    default = newton_mjwarp


@configclass
class ActionsCfg:
    arm_action = IntegratedJointPositionActionCfg(asset_name="robot", preserve_order=True)
    gripper_action = RateLimitedBinaryJointPositionActionCfg(asset_name="robot")
    load = LoadActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        gripper_pos = ObsTerm(func=base_mdp.joint_pos)
        tcp_pose = ObsTerm(func=tcp_pose_b)
        registered_mechanism = ObsTerm(func=registered_mechanism_b)
        last_action = ObsTerm(func=base_mdp.last_action)
        rate = ObsTerm(func=reference_rate)
        gripper_object = ObsTerm(func=gripper_object_detected)
        episode_time = ObsTerm(func=episode_time)
        enable_corruption = True
        concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        arm_joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        arm_joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        gripper_pos = ObsTerm(func=base_mdp.joint_pos)
        tcp_pose = ObsTerm(func=tcp_pose_b)
        mechanism_pose = ObsTerm(func=mechanism_pose_b)
        grasp_target = ObsTerm(func=grasp_target_b)
        tcp_to_grasp = ObsTerm(func=tcp_to_grasp_b)
        mechanism_state = ObsTerm(func=mechanism_state)
        pad_force = ObsTerm(func=pad_force)
        last_action = ObsTerm(func=base_mdp.last_action)
        rate = ObsTerm(func=reference_rate)
        reference = ObsTerm(func=reference_state)
        expert_joint_error = ObsTerm(func=expert_joint_error)
        expert_gripper = ObsTerm(func=expert_gripper_reference)
        expert_mechanism_error = ObsTerm(func=expert_mechanism_error)
        dynamics = ObsTerm(func=dynamics_obs)
        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


@configclass
class EventCfg:
    quiet_solver = EventTerm(func=silence_solver_overflow_warnings, mode="startup")
    robot_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    mechanism_gravcomp = EventTerm(func=apply_articulation_gravcomp, mode="startup")
    robot_physics_material = EventTerm(func=base_mdp.randomize_rigid_body_material, mode="startup")
    mechanism_physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material, mode="startup"
    )
    contacts = EventTerm(func=set_contact_stiffness, mode="startup")
    reset_all = EventTerm(
        func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
    )
    arm_delay = None
    reset_mechanism_root = None
    reset_mechanism_joint = None
    reset_arm = None
    dynamics = EventTerm(func=randomize_dynamics, mode="reset")
    reset_from_bank = EventTerm(func=reset_from_expert_bank, mode="reset")
    reset_panel = None
    end_stops = None
    target_delay = None


@configclass
class CommandsCfg:
    reference = ReferenceCommandCfg()


@configclass
class RewardsCfg:
    track_expert_coarse = RewTerm(func=track_expert_joints)
    track_expert = RewTerm(func=track_expert_joints)
    track_expert_velocity = RewTerm(func=track_expert_velocity)
    track_expert_gripper = RewTerm(func=track_expert_gripper)
    track_expert_contact = RewTerm(func=track_expert_contact)
    track_expert_mechanism = RewTerm(func=track_expert_mechanism)
    track_expert_mechanism_coarse = track_expert_mechanism.replace()
    action_rate = RewTerm(func=base_mdp.action_rate_l2)
    action_magnitude = RewTerm(func=base_mdp.action_l2)
    arm_joint_vel = RewTerm(func=base_mdp.joint_vel_l2)
    pad_force = RewTerm(func=pad_force_excess)
    overspeed = RewTerm(func=rate_excess)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    invalid = DoneTerm(func=invalid_state)
    expert_drift = DoneTerm(func=expert_joint_drift)
    expert_mechanism_lag = DoneTerm(func=expert_mechanism_lag)


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    task: str = MISSING
    robot: RobotCfg = MISSING
    mechanism: MechanismCfg = MISSING
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())
    viewer: ViewerCfg = ViewerCfg()


def load(name):
    with open(os.path.join(REPO_ROOT, "configs", f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def merge(obj, data):
    for key, value in data.items():
        current = obj.get(key) if isinstance(obj, dict) else getattr(obj, key)
        if isinstance(value, dict) and "_target_" in value:
            value = instantiate(value, _convert_="all")
        elif isinstance(value, dict) and (isinstance(current, dict) or is_dataclass(current)):
            merge(current, value)
            continue
        if isinstance(obj, dict):
            obj[key] = value
        else:
            setattr(obj, key, value)


def bank_path(robot, task):
    return os.path.join(REPO_ROOT, "logs", "expert_bank", f"{robot}_{task}.pt")


def as_pose(pos, rot):
    return torch.tensor([pos], dtype=torch.float64), torch.tensor([rot], dtype=torch.float64)


def compose(robot_name, task_name, mode):
    robot, task = load(f"robots/{robot_name}"), load(f"tasks/{task_name}")
    asset = load(f"assets/{task['asset']}")
    layers = (load("train"), robot)
    ee = get_object(robot["ee"])
    offset = ee.tcp_offset
    name = task["asset"]
    mechanism = asset["mechanism"]
    interaction = asset["interaction"].copy()
    grip_effort_scale = interaction.pop("grip_effort_scale", 1.0)
    physics = {
        "asset_name": name,
        "joint_name": "RevoluteJoint",
        "closed_end_stop": mechanism["closed_end_stop"],
    }
    screw = None
    if "screw" in asset:
        coupling = asset["screw"].copy()
        axial_reference = coupling.pop("axial_reference")
        screw = ScrewJointCouplingCfg(
            asset_name=name,
            **coupling,
            viscous_friction=0.0,
            coulomb_friction=0.0,
            stiction=0.0,
            end_stop_base_damping=0.0,
            end_stop_scale=0.0,
            end_stop_activation_distance=0.0005,
            use_native_mimic_constraint=True,
        )
        physics.update(
            joint_name=screw.revolute_joint_name,
            coupled_joint_name=screw.prismatic_joint_name,
            coupled_ratio=screw.pitch_m_per_revolution / math.tau,
            coupled_offset=axial_reference,
        )
    cfg = EnvCfg(
        task=task_name,
        robot=RobotCfg(
            **robot["robot"],
            name=robot_name,
            home=robot["postures"]["home"],
            actuator_groups=robot["actuator_groups"],
            finger_body_names=[pad.rsplit("/", 1)[-1] for pad in robot["pads"]],
            tcp_body=ee.body_name,
            tcp_pos=offset.pos,
            tcp_rot=offset.rot,
        ),
        mechanism=MechanismCfg(
            **physics,
            **interaction,
            closed_rad=task["goal"]["start"],
            open_rad=task["goal"]["target"],
            success_tolerance=task["goal"]["tolerance"],
            screw=screw,
        ),
        scene=InteractiveSceneCfg(num_envs=1, env_spacing=2.5),
        episode_length_s=task["episode_length_s"],
    )
    scene = cfg.scene
    scene.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        collision_group=-1,
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    scene.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    setattr(
        scene,
        physics["asset_name"],
        ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/" + asset["prim"],
            spawn=sim_utils.UsdFileCfg(
                usd_path=get_object("isaaclab_hiveboard.assets." + asset["usd"]),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                **asset["mounting"],
                joint_pos={
                    joint: (
                        physics["coupled_offset"]
                        if screw and joint == screw.prismatic_joint_name
                        else task["goal"]["start"]
                        if joint == physics["joint_name"]
                        else 0.0
                    )
                    for joint in mechanism["joints"]
                },
                joint_vel={".*": 0.0},
            ),
            actuators={
                joint: ImplicitActuatorCfg(
                    joint_names_expr=[joint],
                    **{
                        "stiffness": 0.0,
                        "damping": 0.0,
                        "effort_limit_sim": 2.0,
                        **values,
                    },
                )
                for joint, values in mechanism["joints"].items()
            },
        ),
    )
    panel = asset["panel"]
    scene.honeycomb = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Honeycomb",
        collision_group=-1,
        spawn=sim_utils.UsdFileCfg(
            func=spawn_movable_panel,
            usd_path=get_object("isaaclab_hiveboard.assets.HONEYCOMB_USD"),
            scale=(0.001, 0.001, 0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    frames = asset["frames"]
    frame_path = "{ENV_REGEX_NS}/" + asset["prim"] + "/" + frames["path"]
    scene.target_frame = FrameTransformerCfg(
        prim_path=frame_path,
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=frame_path,
                name=frame,
                offset=OffsetCfg(pos=value["pos"], rot=value.get("rot", frames["rot"])),
            )
            for frame, value in frames["targets"].items()
        ],
    )
    cfg.events.dynamics.params = {
        "ranges": {key: [0.0, 0.0] for key in DYNAMICS},
        "stuck_prob": 0.0,
        "stuck_breakaway": [0.0, 0.0],
    }
    scene, name = cfg.scene, cfg.mechanism.asset_name
    scene.robot = get_object(robot["articulation"]).replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene.ee_frame = make_ee_frame(ee)
    for sensor, pad in zip(("finger_contact", "jaw_contact"), robot["pads"]):
        contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + pad,
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/" + asset["prim"] + "/.*"],
        )
        setattr(scene, sensor, contact)
    state = getattr(scene, name).init_state
    pos, rot = math_utils.combine_frame_transforms(
        *as_pose(robot["board"]["pos"], robot["board"]["rot"]), *as_pose(state.pos, state.rot)
    )
    state.pos, state.rot = tuple(pos[0].tolist()), tuple(rot[0].tolist())
    pos, rot = math_utils.combine_frame_transforms(pos, rot, *as_pose(panel["pos"], panel["rot"]))
    scene.honeycomb.init_state = RigidObjectCfg.InitialStateCfg(
        pos=tuple(pos[0].tolist()), rot=tuple(rot[0].tolist())
    )
    cfg.events.reset_panel = EventTerm(func=reset_panel, mode="reset", params=panel)
    if cfg.mechanism.screw:
        register_screw_joint_mimic(
            cfg.mechanism.screw,
            revolute_pos_at_reset=cfg.mechanism.closed_rad,
            prismatic_pos_at_reset=cfg.mechanism.coupled_offset,
        )
    for layer in layers:
        merge(cfg, layer.get("env", {}))
    physical = asset["physics"]
    merge(cfg.events.dynamics.params, physical["dynamics"])
    for material in ("robot", "mechanism"):
        merge(
            getattr(cfg.events, material + "_physics_material").params,
            physical.get(material + "_material", {}),
        )
    for solver in (cfg.sim.physics.newton_mjwarp, cfg.sim.physics.default):
        merge(solver.solver_cfg, physical.get("solver", {}))
    merge(cfg.viewer, {"asset_name": name, **asset.get("viewer", {})})
    if "end_stop_damping" in mechanism:
        cfg.events.end_stops = EventTerm(
            func=set_joint_limit_damping,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(name, joint_names=[physics["joint_name"]]),
                "kd": mechanism["end_stop_damping"],
            },
        )
    if cfg.mechanism.mode == "contact_grasp":
        cfg.observations.teacher.expert_gripper.params["follow_release"] = True
        cfg.terminations.expert_drift.params["max_error"] = 0.5
        cfg.terminations.expert_mechanism_lag.params["max_error"] = 0.5
    elif cfg.mechanism.mode == "push":
        cfg.observations.teacher.expert_gripper.params["follow_reach"] = True
        cfg.rewards.overspeed = None
        cfg.terminations.expert_mechanism_lag.params["max_error"] = 0.6
    arm = SceneEntityCfg("robot", joint_names=list(cfg.robot.arm_joint_names), preserve_order=True)
    hand = SceneEntityCfg(
        "robot", joint_names=list(cfg.robot.finger_joint_names), preserve_order=True
    )
    obs, events = cfg.observations, cfg.events
    for term in (
        obs.policy.arm_joint_pos,
        obs.teacher.arm_joint_pos,
        obs.teacher.arm_joint_vel,
        cfg.rewards.arm_joint_vel,
        cfg.terminations.invalid,
    ):
        term.params["asset_cfg"] = arm
    for term in (obs.policy.gripper_pos, obs.teacher.gripper_pos, obs.policy.gripper_object):
        term.params["asset_cfg"] = hand
    cfg.actions.arm_action.joint_names = list(cfg.robot.arm_joint_names)
    cfg.actions.load.asset_name = name
    events.robot_physics_material.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=list(cfg.robot.finger_body_names)
    )
    events.mechanism_gravcomp.params["asset_cfg"] = SceneEntityCfg(name)
    events.mechanism_physics_material.params["asset_cfg"] = SceneEntityCfg(name)
    events.contacts.params["shape_regex"] = f"{robot['contact_shapes']}|/{asset['prim']}/"
    events.reset_from_bank.params.update(path=bank_path(robot_name, task_name), task=task_name)
    actuators = dict(scene.robot.actuators)
    for group, effort in robot["grip_effort"].items():
        actuators[group] = actuators[group].replace(effort_limit=effort * grip_effort_scale)
    scene.robot = delayed_actuators(
        scene.robot.replace(actuators=actuators),
        robot["delayed_groups"],
        robot["delay_s"],
        cfg.sim.dt,
    )
    for layer in layers:
        merge(cfg, layer.get(mode, {}))
    return cfg


def runner_cfg(robot, task):
    cfg = RslRlOnPolicyRunnerCfg(
        experiment_name=f"{robot}_{task}",
        actor=RslRlMLPModelCfg(distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg()),
        critic=RslRlMLPModelCfg(),
        algorithm=RslRlPpoAlgorithmCfg(),
    )
    merge(cfg, load("train")["agent"])
    return cfg
