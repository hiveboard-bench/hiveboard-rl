import math

import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

import isaaclab.utils.math as math_utils
from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.actuators.newton.adapter import read_group_parameter, write_group_parameter
from isaaclab.assets import RigidObjectCfg
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
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_hiveboard.assets.anymal.bench import (
    ANYMAL_ARM_JOINT_NAMES,
    ANYMAL_NEWTON_GRIPPER_OPEN,
    NEWTON_GRIPPER_JOINT_NAMES,
)
from isaaclab_hiveboard.mdp.actions import RateLimitedBinaryJointPositionActionCfg
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import register_screw_joint_mimic
from isaaclab_hiveboard.mdp.events import apply_articulation_gravcomp, set_contact_stiffness
from isaaclab_hiveboard.tasks.anymal.ball_valve.configs.scene import BallValveSceneCfg
from isaaclab_hiveboard.tasks.anymal.circuit_breaker.configs.events import CircuitBreakerEventCfg
from isaaclab_hiveboard.tasks.anymal.circuit_breaker.configs.scene import CircuitBreakerSceneCfg
from isaaclab_hiveboard.tasks.anymal.m30_thread.env import AnymalM30ThreadSceneCfg
from isaaclab_hiveboard.tasks.anymal.screw import screw_commands
from isaaclab_hiveboard.tasks.anymal.small_valve.configs.scene import SmallValveSceneCfg
from isaaclab_hiveboard.tasks.franka.ball_valve.env import FrankaBallValveEnvCfg
from isaaclab_hiveboard.tasks.scenes.threads import M30_SPEC
from isaaclab_hiveboard.tasks.viewer import use_play_viewer

from isaaclab_tasks.core.cabinet import mdp as base_mdp
from isaaclab_tasks.utils import PresetCfg

from isaaclab_hiveboard_rl import specs

CONTACT_FORCE_N = 1.0
IDLE_STEP_RAD = 1.0e-3
ARM_DELAY_RANGE_S = (0.0, 0.04)
VALVE_DYNAMICS = ("friction", "damping", "spring", "breakaway", "armature")
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


def reset_panel_from_valve(env, env_ids):
    if env.cfg.valve_task.asset_name != "ball_valve":
        return
    valve = env.scene["ball_valve"]
    panel = env.scene["honeycomb"]
    ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is not None:
        ids = ids[env_ids]
    pose = valve.data.root_link_pose_w[ids].clone()
    offset = torch.tensor((-0.04, 0.0, 0.0), device=env.device).expand(len(ids), -1)
    pose[:, :3] += math_utils.quat_apply(pose[:, 3:7], offset)
    panel.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    panel.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids
    )


@configclass
class ValveTaskCfg:
    asset_name: str = "ball_valve"
    joint_name: str = "RevoluteJoint"
    closed_rad: float = 0.0
    open_rad: float = -math.pi / 2
    grasp_frame: str = "lever_pivot"
    grasp_offset_pos: tuple[float, float, float] = (0.0, -0.0244, 0.0)
    grasp_offset_quat: tuple[float, float, float, float] = (0.7071068, 0.0, 0.0, 0.7071068)
    closed_end_stop: bool = True
    coupled_joint_name: str | None = None
    coupled_ratio: float = 0.0
    coupled_offset: float = 0.0
    hold_by_contact: bool = False
    push: bool = False


def valve_asset(env):
    return env.scene[env.cfg.valve_task.asset_name]


def valve_joint_id(env):
    task = env.cfg.valve_task
    return env.scene[task.asset_name].find_joints(task.joint_name)[0][0]


def write_valve_angle(env, env_ids, angle, rate=None):
    task = env.cfg.valve_task
    valve = valve_asset(env)
    rate = torch.zeros_like(angle) if rate is None else rate
    joints = [valve_joint_id(env)]
    pos, vel = angle[:, None], rate[:, None]
    if task.coupled_joint_name is not None:
        joints.append(valve.find_joints(task.coupled_joint_name)[0][0])
        pos = torch.cat((pos, task.coupled_offset + task.coupled_ratio * pos), dim=-1)
        vel = torch.cat((vel, task.coupled_ratio * vel), dim=-1)
    valve.write_joint_position_to_sim_index(
        position=pos.contiguous(), joint_ids=joints, env_ids=env_ids
    )
    valve.write_joint_velocity_to_sim_index(
        velocity=vel.contiguous(), joint_ids=joints, env_ids=env_ids
    )


def all_ids(env, env_ids):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def sync_coupled_joint(env, env_ids):
    ids = all_ids(env, env_ids)
    if env.cfg.valve_task.coupled_joint_name is not None:
        write_valve_angle(env, ids, valve_angle(env)[ids].clone())


def valve_rate(env):
    return valve_asset(env).data.joint_vel.torch[:, valve_joint_id(env)]


def _progress_of(env, angle):
    task = env.cfg.valve_task
    return (angle - task.closed_rad) / (task.open_rad - task.closed_rad)


def _opening_sign(env):
    task = env.cfg.valve_task
    return 1.0 if task.open_rad > task.closed_rad else -1.0


def _is_open(env, angle, threshold_rad):
    return _opening_sign(env) * (env.cfg.valve_task.open_rad - angle) <= threshold_rad


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


def reset_joints_from_postures(
    env,
    env_ids,
    postures,
    position_range,
    asset_cfg,
    flip_prob=0.0,
    flip_joints=None,
    flip_margin=0.05,
):
    asset = env.scene[asset_cfg.name]
    ids = all_ids(env, env_ids)
    table = torch.tensor(list(postures.values()), device=env.device)
    pick = torch.randint(len(table), (len(ids),), device=env.device)
    noise = torch.empty(len(ids), table.shape[1], device=env.device).uniform_(*position_range)
    q = table[pick] + noise
    limits = asset.data.joint_pos_limits.torch[ids][:, asset_cfg.joint_ids]
    if flip_prob > 0.0 and flip_joints is not None:
        roll1, pitch, roll2 = flip_joints
        flip = torch.rand(len(ids), device=env.device) < flip_prob
        way = torch.where(torch.rand(len(ids), device=env.device) < 0.5, 1.0, -1.0)
        low, high = limits[..., 0] + flip_margin, limits[..., 1] - flip_margin
        r2 = q[:, roll2] - torch.pi
        q[:, roll1] = torch.where(flip, q[:, roll1] + way * torch.pi, q[:, roll1])
        q[:, pitch] = torch.where(flip, -q[:, pitch], q[:, pitch])
        q[:, roll2] = torch.where(flip, r2, q[:, roll2])
        q = torch.minimum(torch.maximum(q, low), high)
    q = q.clamp(min=limits[..., 0], max=limits[..., 1])
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


def _lever_frame_w(env, frame_name=None):
    task = env.cfg.valve_task
    frame_cfg = next(
        f
        for f in env.scene.cfg.target_frame.target_frames
        if f.name == (frame_name or task.grasp_frame)
    )
    body = frame_cfg.prim_path.rsplit("/", 1)[-1]
    return _body_frame_w(env, task.asset_name, body, frame_cfg.offset.pos, frame_cfg.offset.rot)


def grasp_target_w(env):
    pos, quat = _lever_frame_w(env)
    task = env.cfg.valve_task
    off_pos = torch.tensor(task.grasp_offset_pos, device=env.device).expand_as(pos)
    off_quat = torch.tensor(task.grasp_offset_quat, device=env.device).expand_as(quat)
    goal_pos, goal_quat = math_utils.combine_frame_transforms(pos, quat, off_pos, off_quat)
    flip = goal_quat.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(goal_quat)
    flipped = math_utils.quat_mul(goal_quat, flip)
    _, tcp_quat = tcp_w(env)
    error = math_utils.quat_error_magnitude
    nearer = error(tcp_quat, flipped) < error(tcp_quat, goal_quat)
    return goal_pos, torch.where(nearer[:, None], flipped, goal_quat)


def tcp_w(env):
    cfg = env.cfg.rl_robot
    return _body_frame_w(env, "robot", cfg.tcp_body, cfg.tcp_pos, cfg.tcp_rot)


def tcp_grasp_error(env):
    tcp_pos, tcp_quat = tcp_w(env)
    goal_pos, goal_quat = grasp_target_w(env)
    dist = torch.norm(tcp_pos - goal_pos, dim=-1)
    return dist, math_utils.quat_error_magnitude(tcp_quat, goal_quat)


def valve_angle(env):
    return valve_asset(env).data.joint_pos.torch[:, valve_joint_id(env)]


def valve_progress(env):
    return _progress_of(env, valve_angle(env)).clamp(0.0, 1.0)


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


def valve_pose_b(env):
    valve = valve_asset(env)
    return _in_base(env, valve.data.root_pos_w.torch, valve.data.root_quat_w.torch)


def grasp_target_b(env):
    return _in_base(env, *grasp_target_w(env))


def tcp_to_grasp_b(env):
    return grasp_target_b(env)[:, :3] - tcp_pose_b(env)[:, :3]


def valve_state(env):
    return torch.stack((valve_progress(env), valve_rate(env)), dim=-1)


def pad_valve_force(env, sensor_names=("finger_contact", "jaw_contact")):
    forces = []
    for name in sensor_names:
        matrix = env.scene[name].data.force_matrix_w.torch.reshape(env.num_envs, -1, 3)
        forces.append(torch.norm(matrix.sum(dim=1), dim=-1))
    return torch.stack(forces, dim=-1)


def pad_force_split(env, sensor_names=("finger_contact", "jaw_contact")):
    robot = env.scene["robot"]
    left, right = (robot.find_bodies(name)[0][0] for name in env.cfg.rl_robot.finger_body_names)
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


def gripper_object_detected(env, asset_cfg, stopped_below=0.745, moved_above=0.3, max_speed=0.05):
    robot = env.scene[asset_cfg.name]
    finger = robot.data.joint_pos.torch[:, asset_cfg.joint_ids[0]]
    speed = robot.data.joint_vel.torch[:, asset_cfg.joint_ids[0]].abs()
    closing = env.action_manager.get_term("gripper_action").raw_actions[:, 0] < 0.0
    detected = closing & (speed < max_speed) & (finger > moved_above) & (finger < stopped_below)
    return detected.float().unsqueeze(-1)


class registered_valve_b(ManagerTermBase):
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
        root = valve_asset(self._env).data
        pos, quat = _in_base_quat(
            self._env, root.root_pos_w.torch[ids], root.root_quat_w.torch[ids], env_ids=ids
        )
        bias_pos = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * params["bias_pos"]
        euler = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * params["bias_rot"]
        bias_quat = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self._pose[ids, :3] = pos + bias_pos
        self._pose[ids, 3:] = rot6d(math_utils.quat_mul(bias_quat, quat))
        noise = (torch.rand(n, device=self.device) * 2.0 - 1.0) * params["angle_noise"]
        self._progress[ids, 0] = _progress_of(self._env, valve_angle(self._env)[ids] + noise)

    def __call__(self, env, bias_pos=0.0, bias_rot=0.0, jitter_pos=0.0, angle_noise=0.0):
        pose = self._pose.clone()
        pose[:, :3] += torch.randn_like(pose[:, :3]) * jitter_pos
        return torch.cat((pose, self._progress), dim=-1)


def lever_held(env, dist_threshold, ang_threshold):
    task = env.cfg.valve_task
    if task.hold_by_contact or task.push:
        touching = pad_valve_force(env) > CONTACT_FORCE_N
        if task.push:
            return touching.any(dim=-1)
        return touching.all(dim=-1) & _gripper_closed(env).bool()
    dist, ang = tcp_grasp_error(env)
    return (dist < dist_threshold) & (ang < ang_threshold) & _gripper_closed(env).bool()


def pad_force_excess(env, max_force):
    return (pad_valve_force(env) - max_force).clamp(min=0.0).square().sum(dim=-1)


def valve_open_success(env, threshold_rad):
    return _is_open(env, valve_angle(env), threshold_rad)


def released_and_clear(env, threshold_rad, min_distance=0.06, max_force=1.0):
    dist, _ = tcp_grasp_error(env)
    if env.cfg.valve_task.push:
        let_go = torch.ones_like(dist, dtype=torch.bool)
    else:
        let_go = _gripper_closed(env) == 0.0
    return (
        valve_open_success(env, threshold_rad)
        & let_go
        & (pad_valve_force(env).max(dim=-1).values < max_force)
        & (dist >= min_distance)
    )


def opening_credited(env, held, ever_held):
    return ever_held if env.cfg.valve_task.push else held


def invalid_state(env, asset_cfg, max_joint_vel=50.0):
    robot = env.scene["robot"]
    arm_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    vel = torch.cat((arm_vel, valve_asset(env).data.joint_vel.torch), dim=-1)
    nonfinite = ~torch.isfinite(robot.data.joint_pos.torch).all(dim=-1)
    nonfinite |= ~torch.isfinite(vel).all(dim=-1)
    return nonfinite | (vel.abs() > max_joint_vel).any(dim=-1)


class ValveTurnRateCommand(CommandTerm):
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
        self.metrics["tracking_error_rad"] = (valve_angle(self._env) - self.ref_angle).abs()

    def _resample_command(self, env_ids):
        self.rate[env_ids] = torch.empty_like(self.rate[env_ids]).uniform_(*self.cfg.rate_range)
        bank_term = self._env.expert_bank_term
        standard = bank_term.cfg.params.get("standard", False)
        if bank_term.bank.turn_rate is not None and not standard:
            self.rate[env_ids] = bank_term.bank.turn_rate[bank_term.index[env_ids]]
        self.ref_angle[env_ids] = valve_angle(self._env)[env_ids]
        self.engaged[env_ids] = False
        self.engaged_time[env_ids] = 0.0

    def _update_command(self):
        self.engaged |= lever_held(self._env, self.cfg.hold_dist, self.cfg.hold_ang)
        self.engaged_time += self._env.step_dt * self.engaged.float()
        moving = self.engaged_time > self.cfg.engage_delay_s
        sign = _opening_sign(self._env)
        self.ref_angle += sign * self.rate * self._env.step_dt * moving.float()
        if sign < 0:
            self.ref_angle.clamp_(min=self._env.cfg.valve_task.open_rad)
        else:
            self.ref_angle.clamp_(max=self._env.cfg.valve_task.open_rad)


@configclass
class ValveTurnRateCommandCfg(CommandTermCfg):
    class_type: type = ValveTurnRateCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    rate_range: tuple[float, float] = (0.25, 0.8)
    hold_dist: float = 0.03
    hold_ang: float = 0.35
    engage_delay_s: float = 0.0


def turn_rate_command(env, command_name="valve_turn"):
    return env.command_manager.get_term(command_name).rate.unsqueeze(-1)


def valve_reference_state(env, command_name="valve_turn"):
    term = env.command_manager.get_term(command_name)
    error = valve_angle(env) - term.ref_angle
    return torch.stack((term.command[:, 1], error), dim=-1)


def valve_rate_excess(env, factor, command_name="valve_turn"):
    limit = factor * env.command_manager.get_term(command_name).rate
    return (valve_rate(env).abs() - limit).clamp(min=0.0).square()


class randomize_valve_dynamics(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        task = env.cfg.valve_task
        ranges = cfg.params["ranges"]
        assert task.closed_end_stop or (
            ranges["spring"][1] == ranges["breakaway"][1] == cfg.params["stuck_prob"] == 0.0
        ), f"'{task.asset_name}' has no end stop at closed for a spring or seat torque"
        self.valve = env.scene[task.asset_name]
        self.joint_ids = self.valve.find_joints(task.joint_name)[0]
        self.closed_rad = task.closed_rad
        self.params = torch.zeros(env.num_envs, len(VALVE_DYNAMICS), device=env.device)
        stuck_breakaway = cfg.params["stuck_breakaway"]
        scale = [
            max(ranges[name][1], stuck_breakaway[1] if name == "breakaway" else 0.0) or 1.0
            for name in VALVE_DYNAMICS
        ]
        self.scale = torch.tensor(scale, device=env.device)
        env.valve_dynamics_term = self

    def __call__(self, env, env_ids, ranges, stuck_prob, stuck_breakaway):
        ids = all_ids(env, env_ids)
        values = torch.empty(len(ids), len(VALVE_DYNAMICS), device=env.device)
        for i, name in enumerate(VALVE_DYNAMICS):
            values[:, i].uniform_(*ranges[name])
        values[:, 2] = torch.minimum(values[:, 2], values[:, 0] / SPRING_STATIC_RAD)
        stuck = torch.rand(len(ids), device=env.device) < stuck_prob
        if stuck.any():
            n = int(stuck.sum())
            values[stuck, 3] = torch.empty(n, device=env.device).uniform_(*stuck_breakaway)
            closed = torch.full((n, 1), self.closed_rad, device=env.device)
            self.valve.write_joint_position_to_sim_index(
                position=closed, joint_ids=self.joint_ids, env_ids=ids[stuck]
            )
            self.valve.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(closed), joint_ids=self.joint_ids, env_ids=ids[stuck]
            )
        self.write(ids, values)

    def write(self, env_ids, values):
        self.params[env_ids] = values
        self.valve.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=values[:, 0:1].contiguous(),
            joint_viscous_friction_coeff=values[:, 1:2].contiguous(),
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )
        self.valve.write_joint_armature_to_sim_index(
            armature=values[:, 4:5].contiguous(), joint_ids=self.joint_ids, env_ids=env_ids
        )


def randomize_valve_dynamics_cfg(ranges, stuck_prob, stuck_breakaway):
    return EventTerm(
        func=randomize_valve_dynamics,
        mode="reset",
        params={
            "ranges": dict(ranges),
            "stuck_prob": stuck_prob,
            "stuck_breakaway": stuck_breakaway,
        },
    )


def valve_dynamics_obs(env):
    return env.valve_dynamics_term.params / env.valve_dynamics_term.scale


class ValveLoadAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        task = env.cfg.valve_task
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
        term = getattr(self._env, "valve_dynamics_term", None)
        if term is None:
            return
        angle = self._asset.data.joint_pos.torch[:, self._joint_ids[0]]
        spring, breakaway, sign = term.params[:, 2], term.params[:, 3], self._opening_sign
        opening = (sign * (angle - self._closed_rad)).clamp(min=0.0)
        width = (1.0 - opening / BREAKAWAY_WIDTH_RAD).clamp(min=0.0)
        torque = -sign * (spring * opening + breakaway * width)
        self._asset.set_joint_effort_target_index(target=torque[:, None], joint_ids=self._joint_ids)


@configclass
class ValveLoadActionCfg(ActionTermCfg):
    class_type: type = ValveLoadAction
    asset_name: str = "ball_valve"


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


def randomize_actuator_delay_cfg(delay_range_s, groups):
    return EventTerm(
        func=randomize_actuator_delay,
        mode="reset",
        params={"delay_range_s": delay_range_s, "groups": groups},
    )


class ExpertBank:
    def __init__(self, path, device, task, robot, dt):
        bank = torch.load(path, map_location="cpu", weights_only=False)
        meta = bank["meta"]
        assert task is None or meta.get("task") == task, (path, meta.get("task"), task)
        assert list(meta["arm_joint_names"]) == list(robot.arm_joint_names), path
        assert meta.get("robot", robot.name) == robot.name, path
        assert list(meta.get("gripper_joint_names", [])) == list(robot.hand_joint_names), path
        assert abs(float(meta["dt"]) - dt) <= 1e-7, (path, meta["dt"], dt)
        self.meta = meta
        self.size = len(bank["arm_q"])
        q = bank["arm_q"].float()
        if "phase_id" in bank:
            phases = list(meta["phases"])
            phase = bank["phase_id"]
            gripping = (phase == phases.index("grip")) | (phase == phases.index("actuate"))
            assert bool(gripping.any(dim=1).all()), path
            grasp = gripping.float().argmax(dim=1)
        else:
            grasp = (bank["gripper_cmd"] < 0).float().argmax(dim=1)
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
        self.valve = bank["valve_angle"].float()
        self.gripper_q = bank["gripper_q"].float()
        self.gripper_closed = bank["gripper_cmd"] < 0
        self.contact = (bank["pad_force"].float() > CONTACT_FORCE_N).float()
        self.gripper_joint_pos = bank["gripper_joint_pos"].float()
        self.dt = float(meta["dt"])
        self.idle_shift = grasp - self.grasp_step
        self.open_step = open_step
        self.retreat_step = torch.maximum(last_move, open_step)
        self.arm_q0 = bank["arm_q0"].float()
        self.valve_angle0 = bank["valve_angle0"].float()
        self.valve_pose_env = bank["valve_pose_env"].float()
        self.valve_dynamics = None
        if "valve_dynamics" in bank:
            assert list(meta["valve_dynamics"]) == list(VALVE_DYNAMICS), path
            self.valve_dynamics = bank["valve_dynamics"].float()
        self.turn_rate = None
        if "expert_diversity" in bank:
            column = list(meta["expert_diversity"]).index("turn_rate")
            self.turn_rate = bank["expert_diversity"][:, column].float()
        for name, value in list(vars(self).items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))

    def _bank_step(self, idx, step):
        shifted = torch.where(step >= self.grasp_step[idx], step + self.idle_shift[idx], step)
        return shifted.clamp(max=self.q.shape[1] - 1)

    def valve_reference(self, idx, step):
        return self.valve[idx, self._bank_step(idx, step)]

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
        self.robot = env.cfg.rl_robot
        self.bank = ExpertBank(
            cfg.params["path"], env.device, cfg.params.get("expert_task"), self.robot, env.step_dt
        )
        self.index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.start_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        robot = env.scene["robot"]
        hand, arm = list(self.robot.hand_joint_names), list(self.robot.arm_joint_names)
        self._hand_ids = robot.find_joints(hand, preserve_order=True)[0]
        self._arm_ids = robot.find_joints(arm, preserve_order=True)[0]
        self._finger_id = robot.find_joints(env.cfg.actions.gripper_action.joint_names)[0][0]
        self._valve_name = env.cfg.valve_task.asset_name
        env.expert_bank_term = self
        print(f"[INFO] Expert bank: {self.bank.size} trajectories from {cfg.params['path']}")

    def __call__(self, env, env_ids, path, mid_start_prob=0.0, expert_task=None, standard=False):
        ids = all_ids(env, env_ids)
        if standard:
            self.index[ids] = 0
            self.start_step[ids] = 0
            home = torch.tensor(self.robot.home, device=env.device).expand(len(ids), -1)
            env.scene["robot"].write_joint_position_to_sim_index(
                position=home, joint_ids=self._arm_ids, env_ids=ids
            )
            closed = torch.full((len(ids),), env.cfg.valve_task.closed_rad, device=env.device)
            write_valve_angle(env, ids, closed)
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
        valve = env.scene[self._valve_name]
        pose = self.bank.valve_pose_env[draw].clone()
        pose[:, :3] += env.scene.env_origins[ids]
        valve.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        valve.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids
        )
        write_valve_angle(env, ids, self.bank.valve_angle0[draw])
        if self.bank.valve_dynamics is not None:
            env.valve_dynamics_term.write(ids, self.bank.valve_dynamics[draw])
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
        rate = (bank.valve[draw, nxt] - bank.valve[draw, prev]) / span[:, 0]
        write_valve_angle(env, ids, bank.valve[draw, b], rate)


class reset_from_expert_bank_in_order(reset_from_expert_bank):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        order = cfg.params.get("order")
        if order is None:
            gen = torch.Generator().manual_seed(int(cfg.params.get("seed", 0)))
            order = torch.randperm(self.bank.size, generator=gen)
        self.order = torch.as_tensor(order, dtype=torch.long, device=env.device)
        self.cursor = 0

    def __call__(
        self,
        env,
        env_ids,
        path,
        mid_start_prob=0.0,
        expert_task=None,
        standard=False,
        order=None,
        seed=0,
    ):
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


def expert_joint_error(env, command_name="valve_turn"):
    term = env.expert_bank_term
    joint_pos = env.scene["robot"].data.joint_pos.torch[:, term._arm_ids]
    return joint_pos - term.bank.joint_reference(term.index, reference_step(env))


def track_expert_joints(env, std, command_name="valve_turn"):
    error = expert_joint_error(env, command_name)
    return torch.exp(-error.square().sum(dim=-1) / std**2)


def track_expert_velocity(env, std):
    term = env.expert_bank_term
    reference = term.bank.velocity_reference(term.index, reference_step(env))
    measured = env.scene["robot"].data.joint_vel.torch[:, term._arm_ids]
    return torch.exp(-(measured - reference).square().mean(dim=-1) / std**2)


def expert_valve_error(env):
    term = env.expert_bank_term
    reference = term.bank.valve_reference(term.index, reference_step(env))
    return (valve_angle(env) - reference).unsqueeze(-1)


def track_expert_valve(env, std, gate_on_contact=False):
    reward = torch.exp(-expert_valve_error(env)[:, 0].square() / std**2)
    if not gate_on_contact:
        return reward
    term = env.expert_bank_term
    contact = term.bank.contact_reference(term.index, reference_step(env))
    expert_grips = contact.min(dim=-1).values > 0
    grips = (pad_valve_force(env) > CONTACT_FORCE_N).all(dim=-1)
    return torch.where(expert_grips & ~grips, torch.zeros_like(reward), reward)


def expert_gripper_reference(
    env, command_name="valve_turn", follow_release=False, follow_reach=False
):
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
    policy = (pad_valve_force(env) > CONTACT_FORCE_N).float()
    expert = term.bank.contact_reference(term.index, reference_step(env))
    return (policy == expert).float().mean(dim=-1)


def expert_joint_drift(env, max_error, command_name="valve_turn"):
    return torch.linalg.vector_norm(expert_joint_error(env, command_name), dim=-1) > max_error


def expert_valve_lag(env, max_error):
    return expert_valve_error(env)[:, 0].abs() > max_error


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
        joints = list(env.cfg.rl_robot.arm_joint_names)
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

    def __call__(self, env, env_ids, delay_range_s=(0.0, 0.04)):
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


ARM = SceneEntityCfg("robot", joint_names=list(ANYMAL_ARM_JOINT_NAMES), preserve_order=True)
GRIPPER = SceneEntityCfg("robot", joint_names=list(NEWTON_GRIPPER_JOINT_NAMES), preserve_order=True)


@configclass
class BallValveRLPhysicsCfg(PresetCfg):
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
        use_cuda_graph=False,
    )
    default = newton_mjwarp


@configclass
class ActionsCfg:
    arm_action = IntegratedJointPositionActionCfg(
        asset_name="robot", joint_names=list(ANYMAL_ARM_JOINT_NAMES), preserve_order=True, scale=0.1
    )
    gripper_action = RateLimitedBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(NEWTON_GRIPPER_JOINT_NAMES),
        open_command_expr=dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_OPEN)),
        close_command_expr={NEWTON_GRIPPER_JOINT_NAMES[0]: 1.5},
        close_speed=2.0,
    )
    valve_load = ValveLoadActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": ARM},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        gripper_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": GRIPPER},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        tcp_pose = ObsTerm(func=tcp_pose_b)
        registered_valve = ObsTerm(
            func=registered_valve_b,
            params={"bias_pos": 0.01, "bias_rot": 0.035, "jitter_pos": 0.002, "angle_noise": 0.035},
        )
        last_action = ObsTerm(func=base_mdp.last_action)
        turn_rate = ObsTerm(func=turn_rate_command, params={"command_name": "valve_turn"})
        gripper_object = ObsTerm(func=gripper_object_detected, params={"asset_cfg": GRIPPER})
        episode_time = ObsTerm(func=episode_time)
        enable_corruption = True
        concatenate_terms = True
        history_length = 5

    @configclass
    class TeacherCfg(ObsGroup):
        arm_joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": ARM})
        arm_joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": ARM}, scale=0.1)
        gripper_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": GRIPPER})
        tcp_pose = ObsTerm(func=tcp_pose_b)
        valve_pose = ObsTerm(func=valve_pose_b)
        grasp_target = ObsTerm(func=grasp_target_b)
        tcp_to_grasp = ObsTerm(func=tcp_to_grasp_b, scale=5.0)
        valve_state = ObsTerm(func=valve_state)
        pad_force = ObsTerm(func=pad_valve_force, scale=0.02, clip=(0.0, 10.0))
        last_action = ObsTerm(func=base_mdp.last_action)
        turn_rate = ObsTerm(func=turn_rate_command, params={"command_name": "valve_turn"})
        valve_reference = ObsTerm(func=valve_reference_state, params={"command_name": "valve_turn"})
        expert_joint_error = ObsTerm(func=expert_joint_error, params={"command_name": "valve_turn"})
        expert_gripper = ObsTerm(
            func=expert_gripper_reference, params={"command_name": "valve_turn"}
        )
        expert_valve_error = ObsTerm(func=expert_valve_error, scale=5.0)
        valve_dynamics = ObsTerm(func=valve_dynamics_obs)
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
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_inner_finger", "right_inner_finger"]
            ),
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
    gripper_valve_contacts = EventTerm(
        func=set_contact_stiffness,
        mode="startup",
        params={
            "shape_regex": "/robotiq_2f_140/|/Valve/",
            "ke": 4.0e4,
            "kd": 400.0,
            "solimp": (0.95, 0.99, 0.001),
        },
    )
    reset_all = EventTerm(
        func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
    )
    arm_delay = randomize_actuator_delay_cfg(
        ARM_DELAY_RANGE_S, specs.ROBOTS["anymal"].actuator_groups
    )
    reset_valve_root = None
    reset_valve_joint = None
    reset_arm = None
    valve_dynamics = None
    reset_from_bank = None
    reset_panel = EventTerm(func=reset_panel_from_valve, mode="reset")


@configclass
class CommandsCfg:
    valve_turn = ValveTurnRateCommandCfg(rate_range=(0.3, 0.3), hold_dist=0.03, hold_ang=0.35)


@configclass
class RewardsCfg:
    track_expert_coarse = RewTerm(func=track_expert_joints, weight=2.0, params={"std": 0.5})
    track_expert = RewTerm(func=track_expert_joints, weight=3.0, params={"std": 0.15})
    track_expert_velocity = RewTerm(func=track_expert_velocity, weight=0.0, params={"std": 0.5})
    track_expert_gripper = RewTerm(func=track_expert_gripper, weight=3.0, params={"std": 0.1})
    track_expert_contact = RewTerm(func=track_expert_contact, weight=3.0)
    track_expert_valve = RewTerm(
        func=track_expert_valve, weight=6.0, params={"std": 0.1, "gate_on_contact": False}
    )
    track_expert_valve_coarse = track_expert_valve.replace(
        weight=5.0, params={"std": 0.5, "gate_on_contact": False}
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.5)
    action_magnitude = RewTerm(func=base_mdp.action_l2, weight=-0.02)
    arm_joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-1.0e-3, params={"asset_cfg": ARM})
    pad_force = RewTerm(func=pad_force_excess, weight=-1.0e-3, params={"max_force": 150.0})
    valve_overspeed = RewTerm(func=valve_rate_excess, weight=-1.0, params={"factor": 1.5})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    invalid = DoneTerm(func=invalid_state, params={"asset_cfg": ARM})
    expert_drift = DoneTerm(func=expert_joint_drift, params={"max_error": 0.3})
    expert_valve_lag = DoneTerm(func=expert_valve_lag, params={"max_error": 0.3})


@configclass
class BallValveRLSceneCfg(BallValveSceneCfg):
    honeycomb = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Honeycomb",
        spawn=BallValveSceneCfg().honeycomb.spawn.replace(func=spawn_movable_panel),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.04, 0.06, 0.804), rot=(0.0, 0.0, 1.0, 0.0)
        ),
        collision_group=-1,
    )


@configclass
class AnymalBallValveRLEnvCfg(ManagerBasedRLEnvCfg):
    robot: str = "anymal"
    tool: str = "ball_valve"
    rl_robot: specs.RobotCfg = specs.RobotCfg()
    scene: BallValveRLSceneCfg = BallValveRLSceneCfg(num_envs=1024, env_spacing=3.0)
    valve_task: ValveTaskCfg = ValveTaskCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, render_interval=10, physics=BallValveRLPhysicsCfg()
    )
    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_body",
        asset_name="ball_valve",
        body_name="alavanca_pivot",
        eye=(-1.5, 1.5, 0.5),
        lookat=(0.0, 0.0, 0.0),
    )
    decimation = 10
    episode_length_s = 17.0


def make_env_cfg(robot, tool, play=False):
    spec = specs.spec(robot, tool)
    cfg = AnymalBallValveRLEnvCfg(robot=robot, tool=tool, rl_robot=specs.ROBOTS[robot])
    if robot == "franka":
        expert = FrankaBallValveEnvCfg()
        cfg.scene = expert.scene
        cfg.scene.num_envs = 1024
        cfg.scene.env_spacing = 3.0
        cfg.scene.honeycomb = BallValveRLSceneCfg().honeycomb
        cfg.decimation = 33
        cfg.sim.dt = 0.05 / cfg.decimation
        cfg.sim.render_interval = cfg.decimation
        cfg.viewer = expert.viewer
        joints = list(cfg.rl_robot.arm_joint_names)
        arm = SceneEntityCfg("robot", joint_names=joints, preserve_order=True)
        fingers = list(cfg.rl_robot.finger_joint_names)
        hand = SceneEntityCfg("robot", joint_names=fingers, preserve_order=True)
        cfg.actions.arm_action.joint_names = joints
        cfg.actions.gripper_action = expert.actions.gripper_action
        for group in (cfg.observations.policy, cfg.observations.teacher):
            group.arm_joint_pos.params["asset_cfg"] = arm
            group.gripper_pos.params["asset_cfg"] = hand
        cfg.observations.teacher.arm_joint_vel.params["asset_cfg"] = arm
        cfg.observations.policy.gripper_pos.noise = Unoise(n_min=-0.0002, n_max=0.0002)
        cfg.observations.policy.gripper_object.params = {
            "asset_cfg": hand,
            "moved_above": 0.0105,
            "stopped_below": 0.035,
            "max_speed": 0.001,
        }
        cfg.rewards.track_expert_gripper.params["std"] = 0.002
        cfg.rewards.arm_joint_vel.params["asset_cfg"] = arm
        cfg.terminations.invalid.params["asset_cfg"] = arm
        cfg.events.robot_physics_material.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=list(cfg.rl_robot.finger_body_names)
        )
        cfg.events.gripper_valve_contacts = expert.events.finger_contacts
        cfg.events.arm_delay = None
        cfg.valve_task = ValveTaskCfg(
            grasp_offset_pos=(0.0, 0.0, 0.0), grasp_offset_quat=(1.0, 0.0, 0.0, 0.0)
        )
    if tool == "small_valve":
        cfg.scene = SmallValveSceneCfg(num_envs=1024, env_spacing=3.0)
        cfg.valve_task = ValveTaskCfg(
            asset_name="small_valve",
            grasp_frame="rotate_frame",
            grasp_offset_pos=(0.0, 0.0, 0.0),
            grasp_offset_quat=(0.0, 0.0, 0.0, 1.0),
            closed_end_stop=False,
        )
        cfg.viewer.body_name = "eixo_trans"
    if tool == "m30_thread":
        cfg.scene = AnymalM30ThreadSceneCfg(num_envs=1024, env_spacing=3.0)
        cfg.valve_task = ValveTaskCfg(
            asset_name="thread",
            open_rad=-2.0 * math.pi * M30_SPEC.turns,
            grasp_frame="grasp",
            grasp_offset_pos=(-0.02, 0.0, 0.0),
            grasp_offset_quat=(0.0, 0.0, 0.0, 1.0),
            closed_end_stop=False,
            coupled_joint_name="PrismaticJoint",
            coupled_ratio=M30_SPEC.pitch / (2.0 * math.pi),
            coupled_offset=M30_SPEC.start,
            hold_by_contact=True,
        )
        cfg.episode_length_s = 57.0
        cfg.scene.thread.actuators["advance"].stiffness = 0.0
        cfg.scene.thread.actuators["advance"].damping = 0.0
        register_screw_joint_mimic(
            screw_commands("thread", M30_SPEC).pose_command.screw_coupling,
            revolute_pos_at_reset=0.0,
            prismatic_pos_at_reset=M30_SPEC.start,
        )
        cfg.events.gripper_valve_contacts.params["shape_regex"] = "/robotiq_2f_140/|/Thread/"
        cfg.observations.teacher.expert_gripper.params["follow_release"] = True
        cfg.terminations.expert_drift.params["max_error"] = 0.5
        cfg.terminations.expert_valve_lag.params["max_error"] = 0.5
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (0.45, 0.65, 1.0)
        cfg.viewer.lookat = (1.0, 0.0, 0.7)
    if tool == "circuit_breaker":
        cfg.scene = CircuitBreakerSceneCfg(num_envs=1024, env_spacing=3.0)
        cfg.valve_task = ValveTaskCfg(
            asset_name="circuit_breaker",
            closed_rad=math.pi / 6,
            open_rad=-math.pi / 6,
            grasp_frame="rotate_frame",
            grasp_offset_pos=(0.0, 0.0, 0.0),
            grasp_offset_quat=(0.0, 0.0, 0.0, 1.0),
            push=True,
        )
        cfg.episode_length_s = 9.0
        cfg.scene.circuit_breaker.init_state.joint_pos = {"RevoluteJoint": math.pi / 6}
        cfg.events.gripper_valve_contacts.params["shape_regex"] = (
            "/robotiq_2f_140/|/CircuitBreaker/"
        )
        cfg.events.breaker_end_stops = CircuitBreakerEventCfg().breaker_end_stops
        cfg.observations.teacher.expert_gripper.params["follow_reach"] = True
        cfg.rewards.valve_overspeed = None
        cfg.terminations.expert_valve_lag.params["max_error"] = 0.6
        cfg.viewer.body_name = "lever_pivot"
    if robot == "anymal":
        gripper = cfg.scene.robot.actuators["gripper"].replace(effort_limit=spec["grip_torque_nm"])
        cfg.scene.robot = delayed_actuators(
            cfg.scene.robot.replace(actuators={**cfg.scene.robot.actuators, "gripper": gripper}),
            cfg.rl_robot.actuator_groups,
            ARM_DELAY_RANGE_S[1],
            cfg.sim.dt,
        )
        asset = SceneEntityCfg(cfg.valve_task.asset_name)
        cfg.events.valve_gravcomp.params["asset_cfg"] = asset
        cfg.events.valve_physics_material.params["asset_cfg"] = asset
        cfg.actions.valve_load.asset_name = asset.name
        cfg.viewer.asset_name = asset.name
    cfg.events.valve_dynamics = randomize_valve_dynamics_cfg(
        spec["valve_dynamics_ranges"], spec["stuck_prob"], spec["stuck_breakaway_range"]
    )
    cfg.events.reset_from_bank = EventTerm(
        func=reset_from_expert_bank,
        mode="reset",
        params={
            "path": specs.bank_path(robot, tool),
            "mid_start_prob": 0.5,
            "expert_task": spec["expert_task"],
        },
    )
    if robot == "franka":
        cfg.events.target_delay = EventTerm(
            func=TargetDelay, mode="reset", params={"delay_range_s": (0.0, 0.04)}
        )
    if play:
        cfg.scene.num_envs = 16
        cfg.events.robot_physics_material = None
        cfg.events.valve_physics_material = None
        cfg.observations.policy.enable_corruption = False
        cfg.events.reset_from_bank.params["mid_start_prob"] = 0.0
        cfg.terminations.expert_drift = None
        cfg.terminations.expert_valve_lag = None
        cfg.commands.valve_turn.rate_range = (0.3, 0.3)
        use_play_viewer(cfg)
    return cfg


@configclass
class AnymalBallValveStudentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 100
    experiment_name = "anymal_ball_valve_student_ppo"
    logger = "wandb"
    wandb_project = "isaaclab-hiveboard"
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["teacher"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
