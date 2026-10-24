# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The ball valve's RSL-RL agents, logging under the circuit breaker's experiment names."""

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
class AnymalCircuitBreakerTeacherPPORunnerCfg(AnymalBallValveTeacherPPORunnerCfg):
    experiment_name = "anymal_circuit_breaker_teacher"


@configclass
class AnymalCircuitBreakerStudentPPORunnerCfg(AnymalBallValveStudentPPORunnerCfg):
    experiment_name = "anymal_circuit_breaker_student_ppo"


@configclass
class AnymalCircuitBreakerStudentPPORNDRunnerCfg(AnymalBallValveStudentPPORNDRunnerCfg):
    experiment_name = "anymal_circuit_breaker_student_ppo"


@configclass
class AnymalCircuitBreakerStudentRunnerCfg(AnymalBallValveStudentRunnerCfg):
    experiment_name = "anymal_circuit_breaker_student"


@configclass
class AnymalCircuitBreakerStudentRecurrentRunnerCfg(AnymalBallValveStudentRecurrentRunnerCfg):
    experiment_name = "anymal_circuit_breaker_student_rnn"
