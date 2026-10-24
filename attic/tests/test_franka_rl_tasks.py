"""FR3 contracts and shared-bank regressions; CPU tests require no simulation."""

from types import SimpleNamespace

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401
import pytest
import torch
import warp as wp
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.expert_bank import ExpertBank
from isaaclab_hiveboard_rl.tasks.franka.ball_valve_rl.env import ROBOT, FrankaBallValveRLEnvCfg
from isaaclab_hiveboard_rl.tasks.robot import RobotCfg


def _bank(tmp_path, robot=ROBOT):
    q = torch.arange(4, dtype=torch.float32)[None, :, None].expand(2, 4, len(robot.arm_joint_names)) * 0.01
    bank = {
        "meta": {
            "task": "expert",
            "dt": 0.05,
            "arm_joint_names": list(robot.arm_joint_names),
            "gripper_joint_names": list(robot.hand_joint_names),
        },
        "arm_q": q,
        "arm_target": q.clone(),
        "arm_q0": q[:, 0].clone(),
        "valve_angle": torch.zeros(2, 4),
        "valve_angle0": torch.zeros(2),
        "valve_pose_env": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).expand(2, -1),
        "gripper_q": torch.zeros(2, 4),
        "gripper_cmd": torch.tensor([[1.0, -1.0, -1.0, 1.0]]).expand(2, -1),
        "gripper_joint_pos": torch.zeros(2, 4, len(robot.hand_joint_names)),
        "pad_force": torch.zeros(2, 4, 2),
        "t_open": torch.full((2,), 0.1),
    }
    path = tmp_path / "bank.pt"
    torch.save(bank, path)
    return path, bank


def test_franka_configuration_and_registry():
    cfg = FrankaBallValveRLEnvCfg()
    assert cfg.rl_robot.name == "franka"
    assert len(cfg.actions.arm_action.joint_names) == 7
    assert cfg.actions.gripper_action.close_command_expr == {"fr3_finger_joint.*": 0.010}
    assert cfg.decimation == 33 and cfg.sim.dt * cfg.decimation == pytest.approx(0.05)
    assert cfg.events.arm_delay is None
    names = list(vars(cfg.events))
    assert names.index("target_delay") > names.index("reset_from_bank")
    assert all(type(a).__name__ == "ImplicitActuatorCfg" for a in cfg.scene.robot.actuators.values())
    assert cfg.observations.policy.history_length == 5
    assert cfg.observations.policy.gripper_pos.params["asset_cfg"].joint_names == list(ROBOT.finger_joint_names)
    for suffix in ("RL-v0", "RL-Play-v0"):
        spec = gym.spec(f"Isaac-HiveBoard-Franka-BallValve-{suffix}")
        assert spec.kwargs["rsl_rl_student_ppo_cfg_entry_point"] == spec.kwargs["rsl_rl_cfg_entry_point"]


@pytest.mark.parametrize("robot", [RobotCfg(), ROBOT])
def test_bank_dimensions_and_references(tmp_path, robot):
    path, raw = _bank(tmp_path, robot)
    bank = ExpertBank(str(path), "cpu", -1.57, "expert", robot, 0.05)
    idx, step = torch.tensor([0, 1]), torch.tensor([0, 3])
    assert bank.joint_reference(idx, step).shape == (2, len(robot.arm_joint_names))
    assert bank.gripper_joint_pos.shape[-1] == len(robot.hand_joint_names)
    assert torch.isfinite(bank.velocity_reference(idx, step)).all()
    assert not bank.velocity_reference(idx, torch.full((2,), 100)).any()
    with pytest.raises(ValueError, match="interval"):
        ExpertBank(str(path), "cpu", -1.57, "expert", robot, 0.075)
    with pytest.raises(ValueError, match="recorded on"):
        ExpertBank(str(path), "cpu", -1.57, "other", robot)
    raw["meta"]["arm_joint_names"] = list(reversed(robot.arm_joint_names))
    torch.save(raw, path)
    with pytest.raises(ValueError, match="joint order"):
        ExpertBank(str(path), "cpu", -1.57, "expert", robot)


def test_cross_robot_bank_rejected(tmp_path):
    path, _ = _bank(tmp_path)
    with pytest.raises(ValueError, match="joint order"):
        ExpertBank(str(path), "cpu", -1.57)


def test_bad_hand_shape_rejected(tmp_path):
    path, raw = _bank(tmp_path)
    raw["gripper_joint_pos"] = torch.zeros(2, 4, 8)
    torch.save(raw, path)
    with pytest.raises(ValueError, match="gripper_joint_pos"):
        ExpertBank(str(path), "cpu", -1.57, robot=ROBOT)


def test_physics_target_delay_and_subset_reset():
    from isaaclab_hiveboard_rl.tasks.franka.ball_valve_rl.delay import TargetDelay, _delay_targets

    delay = object.__new__(TargetDelay)
    delay.arm_ids = [0]
    delay.length = 4
    delay.steps = torch.tensor([0, 2], dtype=torch.int32)
    delay.history = torch.zeros(2, 4, 1)
    delay.cursor = torch.zeros(2, 1, dtype=torch.int32)
    measured = torch.tensor([[5.0], [7.0]])
    delay.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=measured)))
    env = SimpleNamespace(physics_dt=0.01, device="cpu")
    delay(env, torch.tensor([0]), (0.0, 0.0))
    assert (delay.history[0] == 5).all() and not delay.history[1].any()
    desired, output = torch.zeros(2, 1), torch.zeros(2, 1)
    joints = wp.array([0], dtype=wp.int32, device="cpu")
    for i in range(1, 5):
        desired.fill_(float(i))
        wp.launch(
            _delay_targets,
            dim=(2, 1),
            inputs=[
                wp.from_torch(desired),
                wp.from_torch(output),
                joints,
                joints,
                wp.from_torch(delay.history),
                wp.from_torch(delay.cursor, dtype=wp.int32),
                wp.from_torch(delay.steps, dtype=wp.int32),
                delay.length,
            ],
            device="cpu",
        )
        assert output[0, 0] == i
        assert output[1, 0] == max(0, i - 2)
    before = delay.history[0].clone()
    delay(env, torch.tensor([1]), (0.02, 0.02))
    assert torch.equal(delay.history[0], before)
    assert (delay.history[1] == 7).all() and not delay.cursor[1].any()


def test_franka_encoder_grasp_detection():
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.mdp import gripper_object_detected

    # Empty closure, measured nominal lever grasp, still moving, and fully open.
    q = torch.tensor([[0.010], [0.0113], [0.020], [0.040]])
    v = torch.tensor([[0.0], [0.0], [-0.04], [0.0]])
    robot = SimpleNamespace(
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=q), joint_vel=SimpleNamespace(torch=v))
    )
    action = SimpleNamespace(raw_actions=-torch.ones(4, 1))
    env = SimpleNamespace(scene={"robot": robot}, action_manager=SimpleNamespace(get_term=lambda _: action))
    params = dict(FrankaBallValveRLEnvCfg().observations.policy.gripper_object.params)
    params["asset_cfg"] = SimpleNamespace(name="robot", joint_ids=[0])
    assert gripper_object_detected(env, **params).flatten().tolist() == [0.0, 1.0, 0.0, 0.0]


def test_grasp_diversity_axis_follows_same_physical_lever():
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.env import AnymalBallValveRLEnvCfg
    from isaaclab_hiveboard_rl.tasks.franka.ball_valve_rl.env import EXPERT_GRASP_AXES

    from isaaclab.utils.math import quat_apply

    old = AnymalBallValveRLEnvCfg().scene.target_frame.target_frames
    new = FrankaBallValveRLEnvCfg().scene.target_frame.target_frames
    old_rot = next(f.offset.rot for f in old if f.name == "lever_pivot")
    new_rot = next(f.offset.rot for f in new if f.name == "lever_pivot")
    expected = quat_apply(torch.tensor([old_rot]), torch.tensor([[0.0, 1.0, 0.0]]))
    actual = quat_apply(torch.tensor([new_rot]), torch.tensor([EXPERT_GRASP_AXES[0]]))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_production_driver_requires_integrated_replay_coverage_and_quality(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).parent / "rl/run_franka_pipeline.py"
    spec = importlib.util.spec_from_file_location("franka_pipeline_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {
        "task": "Isaac-HiveBoard-Franka-BallValve-RL-Play-v0", "command": "increment",
        "summary": {"complete": 1.0}, "episodes": [{"trajectory": i} for i in range(20)],
    }
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(report))
    module.check_replay(path, 20)
    report["summary"]["complete"] = 0.94
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="below 95%"):
        module.check_replay(path, 20)
    report["summary"]["complete"] = 1.0
    report["command"] = "absolute"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="integrated-action"):
        module.check_replay(path, 20)
    report["command"] = "increment"
    report["episodes"] = report["episodes"][:19]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="cover 20"):
        module.check_replay(path, 20)
