# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The ball valve's RSL-RL agents, logging under the M30 thread's experiment names."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.agents.rsl_rl_distillation_cfg import (
    AnymalBallValveStudentRecurrentRunnerCfg,
    AnymalBallValveStudentRunnerCfg,
)
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.agents.rsl_rl_ppo_cfg import (
    AnymalBallValveStudentPPORNDRunnerCfg,
    AnymalBallValveStudentPPORunnerCfg,
    AnymalBallValveTeacherPPORunnerCfg,
)


@configclass
class AnymalM30ThreadTeacherPPORunnerCfg(AnymalBallValveTeacherPPORunnerCfg):
    experiment_name = "anymal_m30_thread_teacher"


@configclass
class AnymalM30ThreadStudentPPORunnerCfg(AnymalBallValveStudentPPORunnerCfg):
    experiment_name = "anymal_m30_thread_student_ppo"


@configclass
class AnymalM30ThreadStudentPPORNDRunnerCfg(AnymalBallValveStudentPPORNDRunnerCfg):
    experiment_name = "anymal_m30_thread_student_ppo"


@configclass
class AnymalM30ThreadStudentRunnerCfg(AnymalBallValveStudentRunnerCfg):
    experiment_name = "anymal_m30_thread_student"


@configclass
class AnymalM30ThreadStudentRecurrentRunnerCfg(AnymalBallValveStudentRecurrentRunnerCfg):
    experiment_name = "anymal_m30_thread_student_rnn"
