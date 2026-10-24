"""Use the same PPO algorithm and observation groups as the ANYmal student."""

from isaaclab.utils import configclass

from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.agents.rsl_rl_ppo_cfg import (
    AnymalBallValveStudentPPORunnerCfg,
)


@configclass
class FrankaBallValveStudentPPORunnerCfg(AnymalBallValveStudentPPORunnerCfg):
    experiment_name = "franka_ball_valve_student_ppo"
