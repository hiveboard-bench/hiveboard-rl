# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset from, and track, precomputed cuRobo expert trajectories.

The bank (``scripts/rl/build_expert_bank.py``) holds successful scripted-expert
rollouts from the RL task's own reset distribution. Each episode is reset to the
initial state of one bank trajectory, which then provides the arm-joint
reference for that episode:

* **Reach** (before the lever is first held): the expert's *planned* joint
  path (cuRobo's joint targets) indexed by episode time, up to the grasp. Steps
  where the plan does not move - the scripted expert settling onto each
  segment's end pose before starting the next - are dropped, so the reference
  moves at the plan's speeds with no pauses.
* **Turn** (from the expert's grasp on): the expert's measured joints, also
  by episode time, held at the pose where it opened the valve (the scripted
  expert lets go afterwards). An earlier version evaluated the expert's joints
  at the valve's *actual* angle, so a lever held still kept the arm exactly on
  the reference and joint tracking paid for not turning (teachers v17/v18).

Before the grasp, episode time is the reach's idle-free timeline; the turn
and the valve reference are shifted by the idle steps dropped from the reach.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import BaseArticulation
from isaaclab.managers import EventTermCfg, ManagerTermBase

from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_JOINT_NAMES

from .valve_dynamics import VALVE_DYNAMICS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

#: Every joint of the Robotiq linkage. The bank stores all of them so an
#: episode can start mid-grasp without the solver snapping the linkage.
GRIPPER_LINKAGE_JOINTS = [
    "finger_joint",
    "left_outer_finger_joint",
    "left_inner_finger_joint",
    "left_inner_finger_pad_joint",
    "right_outer_knuckle_joint",
    "right_outer_finger_joint",
    "right_inner_finger_joint",
    "right_inner_finger_pad_joint",
]
#: Pad force [N] above which a finger pad counts as touching the lever. The
#: expert's pads read 0 N before the grasp and 8-14 N while turning.
CONTACT_FORCE_N = 1.0
#: Largest joint-target change [rad] of a step counted as idle in the reach.
IDLE_STEP_RAD = 1.0e-3


class ExpertBank:
    """Expert references precomputed per trajectory, on the simulation device."""

    def __init__(self, path: str, device: str, valve_open_rad: float, task: str | None = None):
        bank = torch.load(path, map_location="cpu", weights_only=False)
        meta = bank["meta"]
        if task is not None and meta.get("task") != task:
            raise ValueError(f"{path} was recorded on {meta.get('task')}, not {task}")
        if list(meta["arm_joint_names"]) != list(ANYMAL_ARM_JOINT_NAMES):
            raise ValueError(f"Bank joint order {meta['arm_joint_names']} differs from {ANYMAL_ARM_JOINT_NAMES}")
        self.meta = meta
        self.size = len(bank["arm_q"])
        q = bank["arm_q"].float()  # (N, T, 6)
        valve = bank["valve_angle"].float()  # (N, T)
        # Grasp step: first step of the grip, or of the actuation when it grips
        # by itself (the ball valve's turn closes the gripper). Banks without
        # named phases: first closed-gripper step, the same step for the ball
        # valve. Open step: first step at open.
        if "phase_id" in bank:
            phases = list(meta["phases"])
            gripping = (bank["phase_id"] == phases.index("grip")) | (bank["phase_id"] == phases.index("actuate"))
            if not bool(gripping.any(dim=1).all()):
                raise ValueError(f"{path} has trajectories without a grip or actuate phase")
            grasp = gripping.float().argmax(dim=1)
        else:
            grasp = (bank["gripper_cmd"] < 0).float().argmax(dim=1)
        open_step = (bank["t_open"] / meta["dt"]).round().long().clamp(max=q.shape[1] - 1)
        # Retreat step: when the arm has backed off and stopped (last step it moved).
        moving = (q[:, 1:] - q[:, :-1]).abs().amax(dim=-1) > IDLE_STEP_RAD
        last_move = q.shape[1] - 1 - moving.flip(dims=[1]).float().argmax(dim=1)
        retreat_step = torch.maximum(last_move, open_step)

        # Reach: planned joints up to the grasp without the idle steps, then
        # held at the grasp configuration.
        target = bank["arm_target"].float()
        paths = []
        for i in range(self.size):
            path = target[i, : int(grasp[i]) + 1]
            moving = (path[1:] - path[:-1]).abs().amax(dim=-1) > IDLE_STEP_RAD
            paths.append(torch.cat((path[:1], path[1:][moving])))
        horizon = max(len(path) for path in paths)
        self.reach = torch.stack([torch.cat((p, p[-1:].expand(horizon - len(p), -1))) for p in paths])  # (N, H, 6)
        self.grasp_step = torch.tensor([len(p) - 1 for p in paths])

        # The expert's own commands by time, for behaviour cloning (expert_action).
        self.arm_target = target
        self.gripper_cmd = bank["gripper_cmd"].float()

        # Turn and valve: the expert's measured joints and valve angle by time.
        # Episode steps past the reach's grasp map to the bank's own steps by
        # the idle steps the reach dropped.
        self.q = q
        self.valve = valve
        self.gripper_q = bank["gripper_q"].float()
        # The expert's gripper command, closed where negative (a regrasping expert opens and closes again).
        self.gripper_closed = bank["gripper_cmd"] < 0
        if "pad_force" not in bank:
            raise ValueError(f"{path} has no pad_force; rebuild it with scripts/rl/build_expert_bank.py")
        # Per-pad contact with the lever, (N, T, 2).
        self.contact = (bank["pad_force"].float() > CONTACT_FORCE_N).float()
        if list(meta.get("gripper_joint_names", [])) != GRIPPER_LINKAGE_JOINTS:
            raise ValueError(f"{path} has no gripper_joint_pos; rebuild it with scripts/rl/build_expert_bank.py")
        # The whole Robotiq linkage, for starting episodes mid-grasp, (N, T, 8).
        self.gripper_joint_pos = bank["gripper_joint_pos"].float()
        self.dt = float(meta["dt"])
        self.idle_shift = grasp - self.grasp_step
        self.open_step = open_step
        self.retreat_step = retreat_step
        self.valve_open_rad = valve_open_rad
        self.arm_q0 = bank["arm_q0"].float()
        self.valve_angle0 = bank["valve_angle0"].float()
        self.valve_pose_env = bank["valve_pose_env"].float()
        # Per-trajectory valve dynamics and expert variation (valve_dynamics.py,
        # expert_diversity.py); older banks have neither.
        self.valve_dynamics = bank["valve_dynamics"].float() if "valve_dynamics" in bank else None
        if self.valve_dynamics is not None and list(meta["valve_dynamics"]) != list(VALVE_DYNAMICS):
            raise ValueError(f"Bank valve dynamics {meta['valve_dynamics']} differ from {VALVE_DYNAMICS}")
        self.turn_rate = None
        if "expert_diversity" in bank:
            self.turn_rate = bank["expert_diversity"][:, list(meta["expert_diversity"]).index("turn_rate")].float()
        for name in (
            "reach",
            "grasp_step",
            "arm_target",
            "gripper_cmd",
            "q",
            "valve",
            "gripper_q",
            "gripper_closed",
            "contact",
            "gripper_joint_pos",
            "idle_shift",
            "open_step",
            "retreat_step",
            "arm_q0",
            "valve_angle0",
            "valve_pose_env",
            "valve_dynamics",
            "turn_rate",
        ):
            if getattr(self, name) is not None:
                setattr(self, name, getattr(self, name).to(device))

    def reach_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Expert joints at episode ``step`` (held at the grasp afterwards), ``(len(idx), 6)``."""
        return self.reach[idx, step.clamp(max=self.reach.shape[1] - 1)]

    def _bank_step(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """The bank's own step for episode ``step``, held at the recording's last step.

        Past the open step the expert lets go of the lever and backs the arm
        off to the approach point, so the references after opening are that
        release and retreat, then the arm standing clear.
        """
        shifted = torch.where(step >= self.grasp_step[idx], step + self.idle_shift[idx], step)
        return shifted.clamp(max=self.q.shape[1] - 1)

    def valve_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Expert valve angle at episode ``step``, ``(len(idx),)`` [rad]."""
        return self.valve[idx, self._bank_step(idx, step)]

    def gripper_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Expert finger joint position at episode ``step``, ``(len(idx),)`` [rad]."""
        return self.gripper_q[idx, self._bank_step(idx, step)]

    def gripper_closed_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Whether the expert commands its gripper closed at episode ``step``, ``(len(idx),)``."""
        return self.gripper_closed[idx, self._bank_step(idx, step)]

    def contact_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Expert per-pad lever contact (1/0) at episode ``step``, ``(len(idx), 2)``."""
        return self.contact[idx, self._bank_step(idx, step)]

    def turn_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Expert measured joints at episode ``step``, ``(len(idx), 6)``."""
        return self.q[idx, self._bank_step(idx, step)]

    def joint_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Joint reference on the same idle-free reach and recorded turn timeline."""
        reach = self.reach_reference(idx, step)
        turn = self.turn_reference(idx, step)
        return torch.where((step >= self.grasp_step[idx])[:, None], turn, reach)

    def velocity_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Central-difference reference velocity [rad/s], zero after the recording.

        Differentiate the actual position-reward timeline, including removed
        reach idle steps. At episode step zero use a forward difference.
        """
        previous = (step - 1).clamp(min=0)
        following = step + 1
        velocity = (self.joint_reference(idx, following) - self.joint_reference(idx, previous)) / (
            (following - previous).float()[:, None] * self.dt
        )
        held = (step >= self.grasp_step[idx]) & (self._bank_step(idx, step) >= self.q.shape[1] - 1)
        return torch.where(held[:, None], torch.zeros_like(velocity), velocity)


class reset_from_expert_bank(ManagerTermBase):
    """Reset event: start each episode on a random bank trajectory.

    Writes the valve root pose, the valve angle and the arm joints of the drawn
    trajectory (the scene's other joints keep the preceding default reset) and
    remembers the index, which the reference terms read. Replaces the valve
    pose, valve angle and arm reset randomization, which the bank already
    samples from the same distribution. When the bank recorded them, the
    trajectory's valve dynamics replace the ones the ``valve_dynamics`` event
    sampled (it must run first), and the turn-rate command takes the expert's
    turning speed (:class:`ValveTurnRateCommand`), so the references stay
    feasible and the observed speed matches them.

    With probability ``mid_start_prob`` the episode starts at a random step of
    the trajectory up to the end of the expert's retreat instead of its
    beginning (reference state initialization): arm, whole gripper linkage and
    valve as the expert had them, e.g. already gripping and turning, or
    letting go. ``start_step``
    holds that step; :func:`reference_step` adds it to the episode time, so
    every time-indexed reference starts there too. Students trained from the
    beginning only (v9-v11, PPO v1) never discovered closing the gripper.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp

        valve_task = mdp.valve_task(env)
        self.bank = ExpertBank(cfg.params["path"], env.device, valve_task.open_rad, cfg.params.get("expert_task"))
        self.index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.start_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        robot: BaseArticulation = env.scene["robot"]
        self._hand_ids = robot.find_joints(GRIPPER_LINKAGE_JOINTS, preserve_order=True)[0]
        self._arm_ids = robot.find_joints(list(ANYMAL_ARM_JOINT_NAMES), preserve_order=True)[0]
        # The joint the bank recorded as gripper_q (build_expert_bank.py).
        self._finger_id = robot.find_joints(env.cfg.actions.gripper_action.joint_names)[0][0]
        self._valve_name = valve_task.asset_name
        # Reference terms find the bank here.
        env.expert_bank_term = self
        print(f"[INFO] Expert bank: {self.bank.size} trajectories from {cfg.params['path']}")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        path: str,
        mid_start_prob: float = 0.0,
        expert_task: str | None = None,
        standard: bool = False,
    ) -> None:
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
        if standard:
            # Keep reference bookkeeping, but retain the preceding authored scene reset.
            # Recording is evaluated from home, rather than a sampled expert start.
            self.index[ids] = 0
            self.start_step[ids] = 0
            from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp
            from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.env import ARM_POSTURES

            robot = env.scene["robot"]
            home = torch.tensor(ARM_POSTURES["home"], device=env.device).expand(len(ids), -1)
            robot.write_joint_position_to_sim_index(position=home, joint_ids=self._arm_ids, env_ids=ids)
            mdp.write_valve_angle(env, ids, torch.full((len(ids),), env.cfg.valve_task.closed_rad, device=env.device))
            return
        draw = torch.randint(0, self.bank.size, (len(ids),), device=env.device)
        self.write_start_state(env, ids, draw)
        if mid_start_prob > 0.0:
            mid = torch.rand(len(ids), device=env.device) < mid_start_prob
            if torch.any(mid):
                last = self.bank.retreat_step[draw[mid]] - self.bank.idle_shift[draw[mid]]
                step = (torch.rand(int(mid.sum()), device=env.device) * (last + 1).float()).long()
                self.write_mid_state(env, ids[mid], draw[mid], step)

    def write_start_state(self, env: ManagerBasedEnv, ids: torch.Tensor, draw: torch.Tensor) -> None:
        """Put environments ``ids`` at the initial state of bank trajectories ``draw``."""
        self.index[ids] = draw
        self.start_step[ids] = 0
        robot: BaseArticulation = env.scene["robot"]
        valve: BaseArticulation = env.scene[self._valve_name]
        pose = self.bank.valve_pose_env[draw].clone()
        pose[:, :3] += env.scene.env_origins[ids]
        valve.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        valve.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
        from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp

        mdp.write_valve_angle(env, ids, self.bank.valve_angle0[draw])
        if self.bank.valve_dynamics is not None:
            env.valve_dynamics_term.write(ids, self.bank.valve_dynamics[draw])
        robot.write_joint_position_to_sim_index(position=self.bank.arm_q0[draw], joint_ids=self._arm_ids, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros(len(ids), len(self._arm_ids), device=env.device), joint_ids=self._arm_ids, env_ids=ids
        )

    def write_mid_state(self, env: ManagerBasedEnv, ids: torch.Tensor, draw: torch.Tensor, step: torch.Tensor) -> None:
        """Put environments ``ids`` at episode ``step`` of bank trajectories ``draw`` (after the start state)."""
        bank = self.bank
        self.start_step[ids] = step
        b = bank._bank_step(draw, step)
        prev, nxt = (b - 1).clamp(min=0), (b + 1).clamp(max=bank.q.shape[1] - 1)
        span = ((nxt - prev).clamp(min=1) * bank.dt)[:, None]
        robot: BaseArticulation = env.scene["robot"]
        robot.write_joint_position_to_sim_index(position=bank.q[draw, b], joint_ids=self._arm_ids, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(
            velocity=(bank.q[draw, nxt] - bank.q[draw, prev]) / span, joint_ids=self._arm_ids, env_ids=ids
        )
        robot.write_joint_position_to_sim_index(
            position=bank.gripper_joint_pos[draw, b], joint_ids=self._hand_ids, env_ids=ids
        )
        robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros(len(ids), len(self._hand_ids), device=env.device),
            joint_ids=self._hand_ids,
            env_ids=ids,
        )
        from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp

        mdp.write_valve_angle(env, ids, bank.valve[draw, b], (bank.valve[draw, nxt] - bank.valve[draw, prev]) / span[:, 0])

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass


class reset_from_expert_bank_in_order(reset_from_expert_bank):
    """:class:`reset_from_expert_bank` drawing whole trajectories in a fixed order, cycling through it.

    ``order`` (a param) lists the trajectory indices; by default it is a ``seed``-ed permutation of the
    whole bank, so every trajectory is drawn once before any repeats. Episodes start at the beginning
    of their trajectory (``mid_start_prob`` is ignored). Used to replay the bank and to record it.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        order = cfg.params.get("order")
        if order is None:
            gen = torch.Generator().manual_seed(int(cfg.params.get("seed", 0)))
            order = torch.randperm(self.bank.size, generator=gen)
        else:
            order = torch.as_tensor(list(order), dtype=torch.long)
            if bad := [int(i) for i in order if not 0 <= i < self.bank.size]:
                raise ValueError(f"Trajectory indices {bad} out of range; the bank has {self.bank.size}.")
        self.order = order.to(env.device)
        self.cursor = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        path: str,
        mid_start_prob: float = 0.0,
        expert_task: str | None = None,
        standard: bool = False,
        order: Sequence[int] | None = None,
        seed: int = 0,
    ) -> None:
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
        picks = self.order[(self.cursor + torch.arange(len(ids), device=env.device)) % len(self.order)]
        self.cursor += len(ids)
        self.write_start_state(env, ids, picks)


def reference_step(env: ManagerBasedEnv) -> torch.Tensor:
    """Step along the expert trajectory: the episode step plus where the episode started on it."""
    term = _bank_term(env)
    return env.episode_length_buf + term.start_step


def _bank_term(env: ManagerBasedEnv) -> reset_from_expert_bank:
    term = getattr(env, "expert_bank_term", None)
    if term is None:
        raise RuntimeError("Expert-bank terms need the reset_from_expert_bank event.")
    return term


def expert_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The bank expert's action for the current step, in the arm action's command, ``(N, 7)``.

    The expert's recorded joint targets and gripper command (-1 closes) by episode time, from where
    the episode started on its trajectory, on the bank's own timeline (idle steps included: this is
    what the expert sent, not the idle-free tracking reference). For the ``increment`` command the
    target becomes the clipped increment from the arm's integrated target, so off the expert's path
    it steers back toward the expert's target at this time; for the ``relative`` ones, the target
    minus the measured joints, which the action adds back.
    """
    term = _bank_term(env)
    bank = term.bank
    step = (bank._bank_step(term.index, term.start_step) + env.episode_length_buf).clamp(
        max=bank.arm_target.shape[1] - 1
    )
    target = bank.arm_target[term.index, step]
    arm = env.action_manager.get_term("arm_action")
    if arm.cfg.command == "increment":
        target = arm.increment_toward(target)
    elif arm.cfg.command.startswith("relative"):
        target = target - arm.measured()
    return torch.cat((target, bank.gripper_cmd[term.index, step, None]), dim=-1)


def expert_joint_reference(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Expert arm-joint reference at the current episode time, ``(N, 6)`` [rad].

    The reach until the expert's grasp step, then the expert's turn (see the
    module docstring). ``command_name`` is kept for the callers' signature.
    """
    term = _bank_term(env)
    idx, step = term.index, reference_step(env)
    return term.bank.joint_reference(idx, step)


def expert_joint_error(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Arm joints minus the expert reference, ``(N, 6)`` [rad] (privileged observation)."""
    robot: BaseArticulation = env.scene["robot"]
    arm = _bank_term(env)._arm_ids
    return robot.data.joint_pos.torch[:, arm] - expert_joint_reference(env, command_name)


def track_expert_joints(env: ManagerBasedRLEnv, std: float, command_name: str = "valve_turn") -> torch.Tensor:
    """``exp(-||q - q_expert||^2 / std^2)`` over the six arm joints; ``std`` [rad]."""
    error = expert_joint_error(env, command_name)
    return torch.exp(-error.square().sum(dim=-1) / std**2)


def track_expert_velocity(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Match measured arm velocity to the expert, using mean squared error.

    ``std`` [rad/s] sets the per-joint RMS error scale. This rewards stopping
    after retreat as well as matching the reach, turn and release speeds.
    """
    term = _bank_term(env)
    reference = term.bank.velocity_reference(term.index, reference_step(env))
    measured = env.scene["robot"].data.joint_vel.torch[:, term._arm_ids]
    return torch.exp(-(measured - reference).square().mean(dim=-1) / std**2)


def expert_valve_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Valve angle minus the expert's at the same episode time, ``(N, 1)`` [rad] (privileged observation)."""
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp

    term = _bank_term(env)
    return (mdp.valve_angle(env) - term.bank.valve_reference(term.index, reference_step(env))).unsqueeze(-1)


def track_expert_valve(env: ManagerBasedRLEnv, std: float, gate_on_contact: bool = False) -> torch.Tensor:
    """``exp(-(θ - θ_expert(t))^2 / std^2)``: the valve follows the expert's by episode time; ``std`` [rad].

    Before the expert grasps, this pays for leaving the valve where it is;
    after, for turning it on the expert's schedule; once the expert has
    opened, for keeping it open.

    With ``gate_on_contact``, it pays while the expert grips the lever (both
    pads in contact) only if both of the policy's pads press on it too (over
    ``CONTACT_FORCE_N``). Ungated, BC-initialized PPO learned to push the lever
    open: 87% of BallValve episodes opened it, 41% with a grasp. The gate reads
    measured pad forces, not the TCP pose and gripper command that teacher
    v19 met without gripping (see RewardsCfg).
    """
    reward = torch.exp(-expert_valve_error(env)[:, 0].square() / std**2)
    if not gate_on_contact:
        return reward
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp

    term = _bank_term(env)
    expert_grips = term.bank.contact_reference(term.index, reference_step(env)).min(dim=-1).values > 0
    grips = (mdp.pad_valve_force(env) > CONTACT_FORCE_N).all(dim=-1)
    return torch.where(expert_grips & ~grips, torch.zeros_like(reward), reward)


def reset_from_expert_bank_cfg(path: str, mid_start_prob: float = 0.0, expert_task: str | None = None) -> EventTermCfg:
    """Reset event drawing each episode's start from the bank at ``path`` (see :class:`reset_from_expert_bank`).

    ``expert_task``, when given, is the scripted task the bank must have been recorded on.
    """
    params = {"path": path, "mid_start_prob": mid_start_prob}
    if expert_task is not None:
        params["expert_task"] = expert_task
    return EventTermCfg(func=reset_from_expert_bank, mode="reset", params=params)


def expert_gripper_reference(
    env: ManagerBasedRLEnv, command_name: str = "valve_turn", follow_release: bool = False, follow_reach: bool = False
) -> torch.Tensor:
    """1 where the expert's gripper is closed, ``(N, 1)`` (privileged observation).

    Closed from the expert's grasp step on the reach timeline (see
    :class:`ExpertBank`). With ``follow_release`` it follows the expert's
    gripper command after that step, so it opens again where the expert lets
    go (a regrasping expert, or the release before the retreat). With
    ``follow_reach`` it follows the command during the reach as well, for an
    expert that closes its hand on the way (a fist that pushes a lever).
    ``command_name`` is kept for the callers' signature.
    """
    term = _bank_term(env)
    step = reference_step(env)
    if follow_reach:
        return term.bank.gripper_closed_reference(term.index, step).float().unsqueeze(-1)
    closed = step >= term.bank.grasp_step[term.index]
    if follow_release:
        closed &= term.bank.gripper_closed_reference(term.index, step)
    return closed.float().unsqueeze(-1)


def track_expert_gripper(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """``exp(-(q_finger - q_expert(t))^2 / std^2)``: the fingers follow the expert's by episode time; ``std`` [rad].

    Tracks where the fingers are, not the open/close command: scoring the
    command's sign let teacher v21 cage the lever with half-closed fingers
    (0.2-0.54 rad against the expert's 0.68 on the lever) and reopen one step
    in seven while still earning 6/7 of the reward.
    """
    term = _bank_term(env)
    robot: BaseArticulation = env.scene["robot"]
    finger = robot.data.joint_pos.torch[:, term._finger_id]
    error = finger - term.bank.gripper_reference(term.index, reference_step(env))
    return torch.exp(-error.square() / std**2)


def track_expert_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fraction of the two finger pads whose lever contact matches the expert's at this episode time.

    Pays for gripping the lever when the expert does (both pads over
    ``CONTACT_FORCE_N``) and for not touching it before. Nothing else pays
    for contact itself, while contact can disturb the tracked joints and valve.
    """
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp

    term = _bank_term(env)
    policy = (mdp.pad_valve_force(env) > CONTACT_FORCE_N).float()
    expert = term.bank.contact_reference(term.index, reference_step(env))
    return (policy == expert).float().mean(dim=-1)


def expert_joint_drift(env: ManagerBasedRLEnv, max_error: float, command_name: str = "valve_turn") -> torch.Tensor:
    """Termination: the arm is more than ``max_error`` [rad] (joint-space norm) from the expert reference."""
    return torch.linalg.vector_norm(expert_joint_error(env, command_name), dim=-1) > max_error


def expert_valve_lag(env: ManagerBasedRLEnv, max_error: float) -> torch.Tensor:
    """Termination: the valve is more than ``max_error`` [rad] off the expert's valve angle at this episode time.

    Behind catches holding the lever without turning it (teacher v17), ahead
    catches whipping it round (teacher v16); before the grasp it catches
    knocking the lever.
    """
    return expert_valve_error(env)[:, 0].abs() > max_error
