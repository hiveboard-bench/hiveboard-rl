# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL and imitation learning on top of the HiveBoard environments.

Importing this package registers the core ``isaaclab_hiveboard`` tasks and the
``*-RL-v0`` / ``*-RL-Play-v0`` tasks below.
"""

import gymnasium as gym

import isaaclab_hiveboard  # noqa: F401  (registers the core HiveBoard tasks)

_ANYMAL_BALL_VALVE_RL_AGENTS = {
    # PPO teacher on privileged state (default agent).
    "rsl_rl_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.agents.rsl_rl_ppo_cfg:AnymalBallValveTeacherPPORunnerCfg"
    ),
    "default_agent": "rsl_rl",
    # Deployable student trained with PPO directly, privileged critic (--agent <key>).
    "rsl_rl_student_ppo_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.agents.rsl_rl_ppo_cfg:AnymalBallValveStudentPPORunnerCfg"
    ),
    # The PPO student with RND curiosity (--agent <key>).
    "rsl_rl_student_ppo_rnd_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.agents.rsl_rl_ppo_cfg:AnymalBallValveStudentPPORNDRunnerCfg"
    ),
    # Deployable student distilled from the teacher (--agent <key>).
    "rsl_rl_distillation_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.agents.rsl_rl_distillation_cfg:AnymalBallValveStudentRunnerCfg"
    ),
    "rsl_rl_distillation_recurrent_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.agents.rsl_rl_distillation_cfg:"
        "AnymalBallValveStudentRecurrentRunnerCfg"
    ),
}

gym.register(
    id="Isaac-HiveBoard-Anymal-BallValve-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.env:AnymalBallValveRLEnvCfg",
        **_ANYMAL_BALL_VALVE_RL_AGENTS,
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-BallValve-RL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.env:AnymalBallValveRLEnvCfg_PLAY",
        **_ANYMAL_BALL_VALVE_RL_AGENTS,
    },
)

# The ball valve's agents (same networks and algorithms), logged under the small valve's names.
_ANYMAL_SMALL_VALVE_RL_AGENTS = {
    "rsl_rl_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.small_valve_rl.agents.rsl_rl_cfg:AnymalSmallValveTeacherPPORunnerCfg",
    "default_agent": "rsl_rl",
    "rsl_rl_student_ppo_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.small_valve_rl.agents.rsl_rl_cfg:AnymalSmallValveStudentPPORunnerCfg",
    "rsl_rl_student_ppo_rnd_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.small_valve_rl.agents.rsl_rl_cfg:AnymalSmallValveStudentPPORNDRunnerCfg",
    "rsl_rl_distillation_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.small_valve_rl.agents.rsl_rl_cfg:AnymalSmallValveStudentRunnerCfg",
    "rsl_rl_distillation_recurrent_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.small_valve_rl.agents.rsl_rl_cfg:AnymalSmallValveStudentRecurrentRunnerCfg"
    ),
}

gym.register(
    id="Isaac-HiveBoard-Anymal-SmallValve-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.small_valve_rl.env:AnymalSmallValveRLEnvCfg",
        **_ANYMAL_SMALL_VALVE_RL_AGENTS,
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-SmallValve-RL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.small_valve_rl.env:AnymalSmallValveRLEnvCfg_PLAY",
        **_ANYMAL_SMALL_VALVE_RL_AGENTS,
    },
)

_ANYMAL_M30_THREAD_RL_AGENTS = {
    "rsl_rl_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.m30_thread_rl.agents.rsl_rl_cfg:AnymalM30ThreadTeacherPPORunnerCfg",
    "default_agent": "rsl_rl",
    "rsl_rl_student_ppo_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.m30_thread_rl.agents.rsl_rl_cfg:AnymalM30ThreadStudentPPORunnerCfg",
    "rsl_rl_student_ppo_rnd_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.m30_thread_rl.agents.rsl_rl_cfg:AnymalM30ThreadStudentPPORNDRunnerCfg",
    "rsl_rl_distillation_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.m30_thread_rl.agents.rsl_rl_cfg:AnymalM30ThreadStudentRunnerCfg",
    "rsl_rl_distillation_recurrent_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.m30_thread_rl.agents.rsl_rl_cfg:AnymalM30ThreadStudentRecurrentRunnerCfg"
    ),
}

gym.register(
    id="Isaac-HiveBoard-Anymal-M30Thread-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.m30_thread_rl.env:AnymalM30ThreadRLEnvCfg",
        **_ANYMAL_M30_THREAD_RL_AGENTS,
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-M30Thread-RL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.m30_thread_rl.env:AnymalM30ThreadRLEnvCfg_PLAY",
        **_ANYMAL_M30_THREAD_RL_AGENTS,
    },
)

_ANYMAL_CIRCUIT_BREAKER_RL_AGENTS = {
    "rsl_rl_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.circuit_breaker_rl.agents.rsl_rl_cfg:AnymalCircuitBreakerTeacherPPORunnerCfg"
    ),
    "default_agent": "rsl_rl",
    "rsl_rl_student_ppo_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.circuit_breaker_rl.agents.rsl_rl_cfg:AnymalCircuitBreakerStudentPPORunnerCfg"
    ),
    "rsl_rl_student_ppo_rnd_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.circuit_breaker_rl.agents.rsl_rl_cfg:AnymalCircuitBreakerStudentPPORNDRunnerCfg"
    ),
    "rsl_rl_distillation_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.circuit_breaker_rl.agents.rsl_rl_cfg:AnymalCircuitBreakerStudentRunnerCfg"
    ),
    "rsl_rl_distillation_recurrent_cfg_entry_point": (
        "isaaclab_hiveboard_rl.tasks.anymal.circuit_breaker_rl.agents.rsl_rl_cfg:AnymalCircuitBreakerStudentRecurrentRunnerCfg"
    ),
}

gym.register(
    id="Isaac-HiveBoard-Anymal-CircuitBreaker-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.circuit_breaker_rl.env:AnymalCircuitBreakerRLEnvCfg",
        **_ANYMAL_CIRCUIT_BREAKER_RL_AGENTS,
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-CircuitBreaker-RL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard_rl.tasks.anymal.circuit_breaker_rl.env:AnymalCircuitBreakerRLEnvCfg_PLAY",
        **_ANYMAL_CIRCUIT_BREAKER_RL_AGENTS,
    },
)

_FRANKA_BALL_VALVE_AGENT = (
    "isaaclab_hiveboard_rl.tasks.franka.ball_valve_rl.agents.rsl_rl_ppo_cfg:FrankaBallValveStudentPPORunnerCfg"
)
for _suffix, _cfg in (("RL-v0", "FrankaBallValveRLEnvCfg"), ("RL-Play-v0", "FrankaBallValveRLEnvCfg_PLAY")):
    gym.register(
        id=f"Isaac-HiveBoard-Franka-BallValve-{_suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"isaaclab_hiveboard_rl.tasks.franka.ball_valve_rl.env:{_cfg}",
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": _FRANKA_BALL_VALVE_AGENT,
            "rsl_rl_student_ppo_cfg_entry_point": _FRANKA_BALL_VALVE_AGENT,
        },
    )
