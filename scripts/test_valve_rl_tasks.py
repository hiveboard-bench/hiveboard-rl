"""RL valve tasks: the ball-valve, small-valve, M30-thread and circuit-breaker configurations and the valve terms' conventions.

Config-only checks (no simulation)::

    uv run --with pytest python -m pytest scripts/test_valve_rl_tasks.py
"""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401
import pytest
import torch
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import valve_dynamics

_TASKS = {
    "Isaac-HiveBoard-Anymal-BallValve-RL-v0": ("ball_valve", "Isaac-HiveBoard-Anymal-BallValve-v0"),
    "Isaac-HiveBoard-Anymal-SmallValve-RL-v0": ("small_valve", "Isaac-HiveBoard-Anymal-SmallValve-v0"),
    "Isaac-HiveBoard-Anymal-M30Thread-RL-v0": ("thread", "Isaac-HiveBoard-Anymal-M30Thread-v0"),
    "Isaac-HiveBoard-Anymal-CircuitBreaker-RL-v0": ("circuit_breaker", "Isaac-HiveBoard-Anymal-CircuitBreaker-v0"),
}


def _cfg(task: str):
    module, name = gym.spec(task).kwargs["env_cfg_entry_point"].split(":")
    return importlib.import_module(module), getattr(importlib.import_module(module), name)()


@pytest.mark.parametrize("task", list(_TASKS))
@pytest.mark.parametrize("play", [False, True])
def test_every_valve_reference_names_the_task_valve(task, play):
    module, cfg = _cfg(task.replace("-RL-v0", "-RL-Play-v0") if play else task)
    asset, expert_task = _TASKS[task]
    assert cfg.valve_task.asset_name == asset
    assert getattr(cfg.scene, asset) is not None
    assert cfg.events.valve_gravcomp.params["asset_cfg"].name == asset
    assert cfg.events.valve_physics_material is None or cfg.events.valve_physics_material.params["asset_cfg"].name == asset
    assert cfg.actions.valve_load.asset_name == asset
    assert cfg.viewer.asset_name == asset
    assert cfg.events.reset_from_bank.params["expert_task"] == expert_task == module.EXPERT_TASK
    # The grasp frame exists in the scene and sits on a body of the valve.
    frames = {f.name: f for f in cfg.scene.target_frame.target_frames}
    assert cfg.valve_task.grasp_frame in frames
    assert frames[cfg.valve_task.grasp_frame].prim_path.startswith(getattr(cfg.scene, asset).prim_path + "/")
    # The bank builder's settings.
    for name in ("VALVE_POSE_RANGE", "VALVE_ANGLE_RANGE", "ARM_POSTURES", "VALVE_DYNAMICS_RANGES",
                 "EXPERT_DIVERSITY_RANGES", "EXPERT_GRASP_AXES", "EXPERT_OVERSHOOT", "EXPERT_GRIP_S", "BANK_PREFIX"):
        assert hasattr(module, name), name
    assert cfg.events.reset_from_bank.params["mid_start_prob"] == (0.0 if play else 0.5)


def test_ball_valve_dynamics_unchanged():
    _, cfg = _cfg("Isaac-HiveBoard-Anymal-BallValve-RL-v0")
    params = cfg.events.valve_dynamics.params
    assert params["ranges"] == valve_dynamics.VALVE_DYNAMICS_RANGES
    scale = valve_dynamics.dynamics_scale(params["ranges"], params["stuck_breakaway"])
    assert scale == (2.0, 0.5, 1.0, 5.0, 0.02)


def test_integrated_target_cannot_wind_up_against_blocked_joint():
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.mdp import IntegratedJointPositionAction

    action = object.__new__(IntegratedJointPositionAction)
    measured = torch.tensor([[0.0, 0.95]])
    action.cfg = SimpleNamespace(command="increment", smoothing=1.0, max_position_error=0.1)
    action._joint_ids = [0, 1]
    action._target = measured.clone()
    action._filtered = torch.zeros_like(measured)
    action._processed_actions = torch.full_like(measured, 0.1)
    action._lower = torch.full_like(measured, -1.0)
    action._upper = torch.full_like(measured, 1.0)
    writes = []
    action._asset = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=measured)),
        set_joint_position_target_index=lambda **kwargs: writes.append(kwargs['target'].clone()))
    for _ in range(100):
        action._fresh = True
        action.apply_actions()
        action.apply_actions()  # Physics substeps must not integrate twice.
    assert torch.allclose(action._target, torch.tensor([[0.1, 1.0]]))
    assert torch.equal(writes[-1], writes[-2])
    # Reverse direction immediately: there must be no hidden accumulated target.
    action._processed_actions.fill_(-0.1)
    action._fresh = True
    action.apply_actions()
    assert torch.allclose(action._target, torch.tensor([[0.0, 0.9]]))


def _arm_action(command: str, target: list[float], measured: list[float] | None = None):
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.mdp import IntegratedJointPositionAction

    action = object.__new__(IntegratedJointPositionAction)
    action.cfg = SimpleNamespace(command=command, scale=0.1, smoothing=1.0, max_position_error=None, clip=None)
    action._scale, action._offset = 0.1, 0.0
    action._joint_ids = [0, 1]
    action._target = torch.tensor([target])
    action._raw_actions = torch.zeros_like(action._target)
    action._filtered = torch.zeros_like(action._target)
    action._lower = torch.full_like(action._target, -1.0)
    action._upper = torch.full_like(action._target, 1.0)
    joint_pos = SimpleNamespace(torch=torch.tensor([measured if measured is not None else target]))
    action._asset = SimpleNamespace(
        set_joint_position_target_index=lambda **kwargs: None, data=SimpleNamespace(joint_pos=joint_pos)
    )
    return action


def test_absolute_integrated_command_runs_the_increment_toward_it():
    absolute = _arm_action("absolute_integrated", [0.0, 0.0])
    increment = _arm_action("increment", [0.0, 0.0])
    goal = torch.tensor([[0.05, 0.5]])
    for _ in range(3):
        label = increment.increment_toward(goal)
        increment.process_actions(label)
        increment.apply_actions()
        absolute.process_actions(goal)
        absolute.apply_actions()
        assert torch.allclose(absolute._target, increment._target)
    # One step reaches the near joint; the far one moves at the clipped 0.1 rad per step.
    assert torch.allclose(absolute._target, torch.tensor([[0.05, 0.3]]))


def test_absolute_command_sets_the_target_within_the_joint_limits():
    action = _arm_action("absolute", [0.0, 0.0])
    action.process_actions(torch.tensor([[0.05, 2.0]]))
    action.apply_actions()
    assert torch.allclose(action._target, torch.tensor([[0.05, 1.0]]))


def test_relative_commands_add_the_measured_joints():
    measured = [0.2, -0.1]
    goal = torch.tensor([[0.25, 0.4]])
    for relative, absolute in [("relative", "absolute"), ("relative_integrated", "absolute_integrated")]:
        rel = _arm_action(relative, [0.2, -0.1], measured)
        ref = _arm_action(absolute, [0.2, -0.1], measured)
        rel.process_actions(goal - torch.tensor([measured]))
        rel.apply_actions()
        ref.process_actions(goal)
        ref.apply_actions()
        assert torch.allclose(rel._target, ref._target)
    # The plain one still clamps to the joint limits.
    action = _arm_action("relative", [0.0, 0.0], [0.5, 0.5])
    action.process_actions(torch.tensor([[0.1, 0.8]]))
    action.apply_actions()
    assert torch.allclose(action._target, torch.tensor([[0.6, 1.0]]))


def test_expert_action_replays_the_bank_commands_by_episode_time():
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import expert_bank

    bank = object.__new__(expert_bank.ExpertBank)
    bank.grasp_step = torch.tensor([5])
    bank.idle_shift = torch.tensor([0])
    bank.q = torch.zeros(1, 4, 2)
    bank.arm_target = torch.tensor([[[0.0, 0.0], [0.05, 0.0], [0.1, 0.5], [0.2, 0.5]]])
    bank.gripper_cmd = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    for command, target, expected in [
        ("absolute", [0.0, 0.0], [0.1, 0.5]),
        ("increment", [0.08, 0.0], [0.2, 1.0]),
        # _arm_action measures its target: relative labels are the bank target minus it.
        ("relative", [0.05, 0.2], [0.05, 0.3]),
    ]:
        env = SimpleNamespace(
            expert_bank_term=SimpleNamespace(bank=bank, index=torch.tensor([0]), start_step=torch.tensor([0])),
            episode_length_buf=torch.tensor([2]),
            action_manager=SimpleNamespace(get_term=lambda name, command=command, target=target: _arm_action(command, target)),
        )
        assert torch.allclose(expert_bank.expert_action(env), torch.tensor([[*expected, -1.0]]))
    # Past the recording the last command is held.
    env.episode_length_buf = torch.tensor([50])
    assert torch.allclose(expert_bank.expert_action(env)[:, -1], torch.tensor([-1.0]))


def test_valve_reward_gated_on_contact_pays_only_a_grip_while_the_expert_grips(monkeypatch):
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import expert_bank, mdp

    bank = object.__new__(expert_bank.ExpertBank)
    bank.grasp_step = torch.tensor([0, 0, 0, 0])
    bank.idle_shift = torch.zeros(4, dtype=torch.long)
    bank.valve = torch.zeros(4, 3)
    bank.q = torch.zeros(4, 3, 6)
    # Expert: no contact at step 0, both pads at step 1, one pad at step 2.
    bank.contact = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]).expand(4, 3, 2)
    env = SimpleNamespace(
        expert_bank_term=SimpleNamespace(bank=bank, index=torch.arange(4), start_step=torch.zeros(4, dtype=torch.long)),
        episode_length_buf=torch.tensor([0, 1, 1, 2]),
    )
    monkeypatch.setattr(mdp, "valve_angle", lambda env: torch.zeros(4))
    # Policy pads: none, both, one, none.
    forces = torch.tensor([[0.0, 0.0], [5.0, 5.0], [5.0, 0.0], [0.0, 0.0]])
    monkeypatch.setattr(mdp, "pad_valve_force", lambda env: forces)
    assert torch.equal(expert_bank.track_expert_valve(env, std=0.1), torch.ones(4))
    # Ungripped valve tracking pays only before and after the expert's two-pad grip.
    assert torch.equal(expert_bank.track_expert_valve(env, std=0.1, gate_on_contact=True), torch.tensor([1.0, 1.0, 0.0, 1.0]))


def test_error_limit_is_disabled():
    # TODO: Expect BallValve's larger bound once it is back (see AnymalBallValveRLEnvCfg.__post_init__).
    for task in _TASKS:
        _, cfg = _cfg(task)
        assert cfg.actions.arm_action.max_position_error is None


def test_expert_velocity_uses_idle_shift_and_stops_at_end():
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.expert_bank import ExpertBank

    bank = object.__new__(ExpertBank)
    bank.dt = 0.05
    bank.grasp_step = torch.tensor([2])
    bank.idle_shift = torch.tensor([2])
    bank.reach = torch.tensor([[[0.0], [0.05], [0.1]]])
    # Two idle samples removed from the recorded reach; turn begins at bank step 4.
    bank.q = torch.tensor([[[0.0], [0.0], [0.0], [0.05], [0.1], [0.15], [0.2]]])
    for step in (0, 1, 2, 3):
        assert bank.velocity_reference(torch.tensor([0]), torch.tensor([step])).item() == pytest.approx(1.0)
    for step in (4, 5, 100):
        assert bank.velocity_reference(torch.tensor([0]), torch.tensor([step])).item() == 0.0


@pytest.mark.parametrize("closed, opening_sign", [(0.0, -1.0), (0.0, 1.0), (0.3, -1.0)])
def test_closing_torque_turns_toward_closed(closed, opening_sign):
    spring, breakaway = torch.tensor([1.0]), torch.tensor([2.0])
    seated = valve_dynamics.closing_torque(torch.tensor([closed]), spring, breakaway, closed, opening_sign)
    part_open = torch.tensor([closed + opening_sign * 0.5])
    opened = valve_dynamics.closing_torque(part_open, spring, breakaway, closed, opening_sign)
    # Full seat torque at closed, spring only past the seat, both pushing against opening.
    assert seated.item() == pytest.approx(-opening_sign * 2.0)
    assert opened.item() == pytest.approx(-opening_sign * 0.5)
    # Past closed (the wrong way) only the seat torque, still toward closed.
    beyond = valve_dynamics.closing_torque(torch.tensor([closed - opening_sign * 0.2]), spring, breakaway, closed, opening_sign)
    assert beyond.item() == pytest.approx(-opening_sign * 2.0)


def test_small_valve_turns_a_quarter_turn_open_negative():
    _, cfg = _cfg("Isaac-HiveBoard-Anymal-SmallValve-RL-v0")
    assert cfg.valve_task.open_rad == pytest.approx(-math.pi / 2) and cfg.valve_task.closed_rad == 0.0


def test_m30_thread_runs_the_nut_down_two_turns():
    module, cfg = _cfg("Isaac-HiveBoard-Anymal-M30Thread-RL-v0")
    task = cfg.valve_task
    assert task.open_rad == pytest.approx(-4.0 * math.pi) and task.closed_rad == 0.0
    # The nut's travel is 7 mm at the start and 0 when seated.
    assert task.coupled_offset == pytest.approx(0.007)
    assert task.coupled_offset + task.coupled_ratio * task.open_rad == pytest.approx(0.0, abs=1e-9)
    assert task.hold_by_contact and not task.closed_end_stop
    # The thread's mimic constraint moves the nut, not a drive anchored at the start.
    assert cfg.scene.thread.actuators["advance"].stiffness == 0.0
    # Seated within the protocol's 1 mm.
    assert module.SUCCESS_TOLERANCE_RAD * task.coupled_ratio == pytest.approx(0.001)
    assert cfg.observations.teacher.expert_gripper.params["follow_release"]


def test_circuit_breaker_pushes_the_lever_up_from_its_down_stop():
    module, cfg = _cfg("Isaac-HiveBoard-Anymal-CircuitBreaker-RL-v0")
    task = cfg.valve_task
    assert task.push and task.closed_end_stop and not task.hold_by_contact
    # Down (+30 deg) to up (-30 deg), the joint's limits.
    assert task.closed_rad == pytest.approx(math.pi / 6) and task.open_rad == pytest.approx(-math.pi / 6)
    assert cfg.scene.circuit_breaker.init_state.joint_pos["RevoluteJoint"] == pytest.approx(task.closed_rad)
    # The fist closes during the reach, and the flick's speed is not penalized.
    assert cfg.observations.teacher.expert_gripper.params["follow_reach"]
    assert cfg.rewards.valve_overspeed is None
    # No spring: a released lever stays up.
    assert module.VALVE_DYNAMICS_RANGES["spring"] == (0.0, 0.0)
    # The end stops are damped, so a flicked lever does not bounce back down.
    from isaaclab_hiveboard.tasks.scenes.circuit_breaker import BREAKER_LIMIT_KD

    assert cfg.events.breaker_end_stops.params["kd"] == BREAKER_LIMIT_KD


def test_circuit_breaker_expert_pushes_without_idling():
    from isaaclab_hiveboard.mdp.commands.sequential_pose_command import GripperCommand

    module, _ = _cfg("Isaac-HiveBoard-Anymal-CircuitBreaker-RL-v0")
    m, c = gym.spec(module.EXPERT_TASK).kwargs["env_cfg_entry_point"].split(":")
    expert = getattr(importlib.import_module(m), c)()
    module.configure_expert(expert)
    commands = expert.commands.pose_command.commands
    # The only gripper segment left is the press that holds the lever up; the fist stays closed throughout.
    assert [cmd.phase for cmd in commands] == ["approach", "engage", "actuate", "actuate", "retreat", "retreat"]
    assert [type(cmd) is GripperCommand for cmd in commands] == [False, False, False, True, False, False]
    assert all(not getattr(cmd, "gripper_open", False) and not getattr(cmd, "open_gripper", False) for cmd in commands)
    # The push ends on the lever's angle near its up stop.
    push = commands[2]
    assert push.done_when_joint[:2] == ("circuit_breaker", "RevoluteJoint")
    assert push.done_when_joint[3] == pytest.approx(module.LEVER_UP_RAD + module.PUSH_DONE_RAD)
    assert expert.scene.circuit_breaker.init_state.joint_pos["RevoluteJoint"] == pytest.approx(module.LEVER_DOWN_RAD)


def test_critic_warmup_ppo_keeps_the_actor_then_starts_it_at_the_configured_rate():
    import types

    from isaaclab_rl.rsl_rl import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg
    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    from isaaclab.utils import to_dict

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    task, entry = "Isaac-HiveBoard-Anymal-BallValve-RL-v0", "rsl_rl_student_ppo_cfg_entry_point"
    cfg = to_dict(handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(task, entry), check_rsl_rl_version()))
    cfg["multi_gpu"] = None
    cfg["algorithm"]["critic_warmup_updates"] = 1
    cfg["algorithm"]["critic_warmup_learning_rate"] = 1.0e-3
    cfg["algorithm"]["learning_rate"] = 1.0e-5
    cfg["algorithm"]["freeze_actor_normalization"] = True
    n = 4
    obs = lambda: TensorDict({"policy": torch.randn(n, 180), "teacher": torch.randn(n, 70)}, batch_size=[n])  # noqa: E731
    alg = resolve_callable(cfg["algorithm"]["class_name"]).construct_algorithm(
        obs(), types.SimpleNamespace(num_envs=n, num_actions=7), cfg, "cpu"
    )
    flat = lambda module: torch.cat([p.detach().flatten().clone() for p in module.parameters()])  # noqa: E731
    # A loaded checkpoint's optimizer brings its own learning rate (PPO.load).
    alg._set_learning_rate(5.0e-4)

    def update() -> tuple[bool, bool, float]:
        actor, critic, lr = flat(alg.actor), flat(alg.critic), alg.learning_rate
        for _ in range(cfg["num_steps_per_env"]):
            o = obs()
            alg.act(o)
            alg.process_env_step(o, torch.randn(n), torch.zeros(n, dtype=torch.bool), {})
        alg.compute_returns(obs())
        alg.update()
        return not torch.equal(actor, flat(alg.actor)), not torch.equal(critic, flat(alg.critic)), lr

    actor_moved, critic_moved, _ = update()
    assert not actor_moved and critic_moved
    assert all(p.requires_grad for p in alg.actor.parameters())
    # After the warm-up the actor starts at the configured rate, not the checkpoint's.
    assert alg.learning_rate == 1.0e-5
    normalizer = alg._raw_actor.obs_normalizer
    mean, critic_mean = normalizer._mean.clone(), alg._raw_critic.obs_normalizer._mean.clone()
    actor_moved, critic_moved, _ = update()
    assert actor_moved and critic_moved
    # The actor's normalization stays as loaded; the critic's keeps learning.
    assert torch.equal(normalizer._mean, mean) and not torch.equal(alg._raw_critic.obs_normalizer._mean, critic_mean)
