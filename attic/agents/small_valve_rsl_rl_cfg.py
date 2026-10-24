# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The ball valve's RSL-RL agents, logging under the small valve's experiment names."""

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
class AnymalSmallValveTeacherPPORunnerCfg(AnymalBallValveTeacherPPORunnerCfg):
    experiment_name = "anymal_small_valve_teacher"


@configclass
class AnymalSmallValveStudentPPORunnerCfg(AnymalBallValveStudentPPORunnerCfg):
    experiment_name = "anymal_small_valve_student_ppo"


@configclass
class AnymalSmallValveStudentPPORNDRunnerCfg(AnymalBallValveStudentPPORNDRunnerCfg):
    experiment_name = "anymal_small_valve_student_ppo"


@configclass
class AnymalSmallValveStudentRunnerCfg(AnymalBallValveStudentRunnerCfg):
    experiment_name = "anymal_small_valve_student"


@configclass
class AnymalSmallValveStudentRecurrentRunnerCfg(AnymalBallValveStudentRecurrentRunnerCfg):
    experiment_name = "anymal_small_valve_student_rnn"
