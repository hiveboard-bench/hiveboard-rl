import math
import os
from dataclasses import MISSING, is_dataclass

import torch
import warp as wp
import yaml
from hydra.utils import instantiate
from isaaclab_newton.physics import NewtonManager

import isaaclab.utils.math as math_utils
from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.actuators.newton.adapter import read_group_parameter, write_group_parameter
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_hiveboard import tasks_simple as board
from isaaclab_hiveboard.tasks_simple import (
    all_ids,
    mechanism_asset,
    mechanism_joint_id,
    mechanism_position,
    write_mechanism_position,
)

from isaaclab_tasks.core.cabinet import mdp as base_mdp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONTACT_FORCE_N = 1.0
IDLE_STEP_RAD = 1.0e-3
HOLD_DIST = 0.03
HOLD_ANG = 0.35
ARM_COMMANDS = ("increment", "absolute_integrated", "absolute", "relative_integrated", "relative")


def mechanism_rate(env):
    if env.cfg.task.asset.joint is None:
        root, body = mechanism_asset(env).data, env.scene["manipuland"].data
        delta = body.root_pos_w.torch - root.root_pos_w.torch
        velocity = (
            body.root_link_lin_vel_w.torch - root.root_link_lin_vel_w.torch
            - torch.cross(root.root_ang_vel_w.torch, delta, dim=-1)
        )
        return math_utils.quat_apply_inverse(root.root_quat_w.torch, velocity)[:, 0]
    return mechanism_asset(env).data.joint_vel.torch[:, mechanism_joint_id(env)]


def _progress_of(env, angle):
    task = env.cfg.task
    return (angle - task.start) / (task.target - task.start)


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


def grasp_target_w(env):
    frame = env.scene["target_frame"].data
    index = frame.target_frame_names.index(env.cfg.task.asset.grasp_frame)
    goal_pos = frame.target_pos_w.torch[:, index]
    goal_quat = frame.target_quat_w.torch[:, index]
    flip = goal_quat.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(goal_quat)
    flipped = math_utils.quat_mul(goal_quat, flip)
    _, tcp_quat = tcp_w(env)
    error = math_utils.quat_error_magnitude
    nearer = error(tcp_quat, flipped) < error(tcp_quat, goal_quat)
    return goal_pos, torch.where(nearer[:, None], flipped, goal_quat)


def tcp_w(env):
    ee, robot = env.cfg.robot.ee, env.scene["robot"]
    body = robot.find_bodies(ee.body_name)[0][0]
    pos, quat = robot.data.body_pos_w.torch[:, body], robot.data.body_quat_w.torch[:, body]
    off_pos = torch.tensor(ee.tcp_offset.pos, device=env.device, dtype=pos.dtype).expand_as(pos)
    off_rot = torch.tensor(ee.tcp_offset.rot, device=env.device, dtype=quat.dtype).expand_as(quat)
    return math_utils.combine_frame_transforms(pos, quat, off_pos, off_rot)


def tcp_grasp_error(env):
    tcp_pos, tcp_quat = tcp_w(env)
    goal_pos, goal_quat = grasp_target_w(env)
    dist = torch.norm(tcp_pos - goal_pos, dim=-1)
    return dist, math_utils.quat_error_magnitude(tcp_quat, goal_quat)


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
    left, right = (robot.find_bodies(p.rsplit("/", 1)[-1])[0][0] for p in env.cfg.robot.pads)
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
    mode = env.cfg.task.asset.mode
    if mode != "grasp":
        touching = pad_force(env) > CONTACT_FORCE_N
        if mode == "push":
            return touching.any(dim=-1)
        return touching.all(dim=-1) & _gripper_closed(env).bool()
    dist, ang = tcp_grasp_error(env)
    return (dist < HOLD_DIST) & (ang < HOLD_ANG) & _gripper_closed(env).bool()


def pad_force_excess(env, max_force):
    return (pad_force(env) - max_force).clamp(min=0.0).square().sum(dim=-1)


def _at_goal(env):
    task = env.cfg.task
    return board.opening_sign(env) * (task.target - mechanism_position(env)) <= task.tolerance


def succeeded(env):
    task = env.cfg.task
    reached = _at_goal(env)
    if task.persistent_goal:
        return reached
    env.goal_reached |= reached
    return env.goal_reached


def released_and_clear(env, min_distance=0.06, max_force=1.0):
    dist, _ = tcp_grasp_error(env)
    if env.cfg.task.asset.mode == "push":
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
    contact = ever_held if env.cfg.task.asset.mode == "push" else held
    return contact if env.cfg.task.persistent_goal else contact & _at_goal(env)


def invalid_state(env, asset_cfg, max_joint_vel):
    robot = env.scene["robot"]
    arm_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    vel = torch.cat((arm_vel, mechanism_asset(env).data.joint_vel.torch), dim=-1)
    if env.cfg.task.asset.manipuland_usd is not None:
        body = env.scene["manipuland"].data
        vel = torch.cat((vel, body.root_vel_w.torch), dim=-1)
    nonfinite = ~torch.isfinite(robot.data.joint_pos.torch).all(dim=-1)
    if env.cfg.task.asset.manipuland_usd is not None:
        body = env.scene["manipuland"].data
        nonfinite |= ~torch.isfinite(body.root_pose_w.torch).all(dim=-1)
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
        sign = board.opening_sign(self._env)
        self.ref_angle += sign * self.rate * self._env.step_dt * moving.float()
        if sign < 0:
            self.ref_angle.clamp_(min=self._env.cfg.task.target)
        else:
            self.ref_angle.clamp_(max=self._env.cfg.task.target)


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


def dynamics_obs(env):
    return env.dynamics_term.params / env.dynamics_term.scale


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
        if "manipuland_pose_env0" in bank:
            for name in (
                "manipuland_pose_env0", "manipuland_velocity0",
                "manipuland_pose_env", "manipuland_velocity",
            ):
                setattr(self, name, bank[name].float())
        if "goal_reached" in bank:
            self.goal_reached = bank["goal_reached"].bool()
        assert list(meta["dynamics"]) == list(board.DYNAMICS), path
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

    def gripper_closed_reference(self, idx, step):
        return self.gripper_closed[idx, self._bank_step(idx, step)]

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
        env.expert_bank_term = self
        print(f"[INFO] Expert bank: {self.bank.size} trajectories from {cfg.params['path']}")

    def __call__(self, env, env_ids, path, task, mid_start_prob, standard):
        ids = all_ids(env, env_ids)
        if standard:
            self.index[ids] = 0
            self.start_step[ids] = 0
            home = torch.tensor(self.robot.postures["home"], device=env.device)
            home = home.expand(len(ids), -1)
            env.scene["robot"].write_joint_position_to_sim_index(
                position=home, joint_ids=self._arm_ids, env_ids=ids
            )
            closed = torch.full((len(ids),), env.cfg.task.start, device=env.device)
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
        mechanism = mechanism_asset(env)
        pose = self.bank.pose_env[draw].clone()
        pose[:, :3] += env.scene.env_origins[ids]
        mechanism.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        mechanism.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids
        )
        write_mechanism_position(env, ids, self.bank.position0[draw])
        env.dynamics_term.write(ids, self.bank.dynamics[draw])
        if env.cfg.task.asset.manipuland_usd is not None:
            pose = self.bank.manipuland_pose_env0[draw].clone()
            pose[:, :3] += env.scene.env_origins[ids]
            env.scene["manipuland"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.scene["manipuland"].write_root_velocity_to_sim_index(
                root_velocity=self.bank.manipuland_velocity0[draw], env_ids=ids
            )
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
        if env.cfg.task.asset.manipuland_usd is not None:
            pose = bank.manipuland_pose_env[draw, b].clone()
            pose[:, :3] += env.scene.env_origins[ids]
            env.scene["manipuland"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.scene["manipuland"].write_root_velocity_to_sim_index(
                root_velocity=bank.manipuland_velocity[draw, b], env_ids=ids
            )
        if not env.cfg.task.persistent_goal:
            env.goal_reached[ids] = bank.goal_reached[draw, b]


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
    step = term.bank._bank_step(term.index, reference_step(env))
    error = finger - term.bank.gripper_q[term.index, step]
    return torch.exp(-error.square() / std**2)


def track_expert_contact(env):
    term = env.expert_bank_term
    policy = (pad_force(env) > CONTACT_FORCE_N).float()
    step = term.bank._bank_step(term.index, reference_step(env))
    expert = term.bank.contact[term.index, step]
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
class ActionsCfg(board.ActionsCfg):
    arm_action = IntegratedJointPositionActionCfg(asset_name="robot", preserve_order=True)


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
    quiet_solver = None
    robot_gravcomp = None
    mechanism_gravcomp = None
    robot_physics_material = None
    mechanism_physics_material = None
    manipuland_physics_material = None
    contacts = None
    reset_all = None
    arm_delay = None
    reset_mechanism_root = None
    reset_mechanism_joint = None
    reset_manipuland = None
    reset_arm = None
    dynamics = None
    reset_goal = None
    reset_from_bank = EventTerm(func=reset_from_expert_bank, mode="reset")
    reset_panel = None
    end_stops = None
    target_delay = None


@configclass
class CommandsCfg(board.CommandsCfg):
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
class EnvCfg(board.HiveBoardEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()


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


def compose(robot_name, task_name, mode):
    robot = load(f"robots/{robot_name}")
    layers = (load("train"), robot)
    cfg = board.build(EnvCfg(), robot_name, task_name)
    for layer in layers:
        merge(cfg, layer.get("env", {}))
    asset = cfg.task.asset
    merge(cfg.viewer, {"asset_name": asset.name, **asset.viewer})
    obs, events = cfg.observations, cfg.events
    if asset.grip_width is not None and cfg.robot.gripper_width_scale is not None:
        obs.policy.gripper_object.params["moved_above"] += (
            max(cfg.robot.gripper_close.values())
            - max(board.ROBOTS[robot_name].gripper_close.values())
        )
    if asset.mode == "contact_grasp":
        obs.teacher.expert_gripper.params["follow_release"] = True
        cfg.terminations.expert_drift.params["max_error"] = 0.5
        cfg.terminations.expert_mechanism_lag.params["max_error"] = 0.5
    elif asset.mode == "push":
        obs.teacher.expert_gripper.params["follow_reach"] = True
        cfg.rewards.overspeed = None
        cfg.terminations.expert_mechanism_lag.params["max_error"] = 0.6
    arm = SceneEntityCfg("robot", joint_names=list(cfg.robot.arm_joint_names), preserve_order=True)
    hand = SceneEntityCfg(
        "robot", joint_names=list(cfg.robot.finger_joint_names), preserve_order=True
    )
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
    events.reset_from_bank.params.update(path=bank_path(robot_name, task_name), task=task_name)
    cfg.scene.robot = delayed_actuators(
        cfg.scene.robot, robot["delayed_groups"], robot["delay_s"], cfg.sim.dt
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
