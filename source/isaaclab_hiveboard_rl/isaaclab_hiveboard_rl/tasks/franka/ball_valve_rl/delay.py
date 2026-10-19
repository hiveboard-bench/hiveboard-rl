"""Physics-rate position-target latency, retaining FR3's implicit Newton drives."""

import math

import torch
import warp as wp

from isaaclab.managers import EventTermCfg, ManagerTermBase

from isaaclab_hiveboard_rl.tasks.robot import robot_cfg


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
    """One queue per environment; called after actuator updates in each Newton step."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from isaaclab_newton.physics import NewtonManager

        self.robot = env.scene["robot"]
        profile = robot_cfg(env)
        self.arm_ids = self.robot.find_joints(list(profile.arm_joint_names), preserve_order=True)[0]
        self.length = math.ceil(0.04 / env.physics_dt) + 1
        self.history = torch.zeros(env.num_envs, self.length, len(self.arm_ids), device=env.device)
        self.cursor = torch.zeros(env.num_envs, len(self.arm_ids), device=env.device, dtype=torch.int32)
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
        # Callback lifetime is the simulation's: NewtonManager.close clears it.
        # This hook runs once per physics step on both the eager all-implicit
        # path and the native actuator path, before either solver substep.
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
            from isaaclab_newton.assets.articulation.joint_coordinates import scatter_joint_coordinates

            scatter_joint_coordinates(
                data._joint_coord_map,
                data._sim_bind_joint_position_target,
                data._sim_bind_joint_target_coords,
                self.robot._ALL_ENV_MASK,
            )

    def __call__(self, env, env_ids, delay_range_s=(0.0, 0.04)):
        low, high = delay_range_s
        if not 0 <= low <= high <= 0.04:
            raise ValueError("FR3 target delay must be between 0 and 0.04 seconds")
        ids = slice(None) if env_ids is None else env_ids
        n = self.steps[ids].numel()
        self.steps[ids] = torch.randint(
            math.floor(low / env.physics_dt),
            math.ceil(high / env.physics_dt) + 1,
            (n,),
            device=env.device,
            dtype=torch.int32,
        )
        # This event runs after bank resets, so contact starts seed the complete queue.
        measured = self.robot.data.joint_pos.torch[ids][:, self.arm_ids]
        self.history[ids] = measured[:, None, :]
        self.cursor[ids] = 0


def target_delay_cfg(delay_range_s=(0.0, 0.04)):
    return EventTermCfg(func=TargetDelay, mode="reset", params={"delay_range_s": delay_range_s})
