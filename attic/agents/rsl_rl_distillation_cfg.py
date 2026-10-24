# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student distillation: the deployable ``policy`` group imitates the PPO teacher (DAgger-style).

The teacher model must mirror the PPO actor (same hidden dims and observation
normalization) so ``actor_state_dict`` from the teacher checkpoint loads into
it. Train with ``--checkpoint <teacher model_*.pt>``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)

from .rsl_rl_ppo_cfg import TEACHER_HIDDEN_DIMS


def _teacher() -> RslRlMLPModelCfg:
    # Clamped to the executed action range; see ``models.ClippedMLPModel``.
    return RslRlMLPModelCfg(
        class_name="isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.agents.models:ClippedMLPModel",
        hidden_dims=TEACHER_HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )


@configclass
class AnymalBallValveStudentRunnerCfg(RslRlDistillationRunnerCfg):
    """MLP student over the 5-step observation history of the ``policy`` group."""

    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 100
    experiment_name = "anymal_ball_valve_student"
    logger = "wandb"
    wandb_project = "isaaclab-hiveboard"
    clip_actions = 1.0
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}
    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    teacher = _teacher()
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=5.0e-4,
        gradient_length=15,
        max_grad_norm=1.0,
    )


@configclass
class AnymalBallValveStudentRecurrentRunnerCfg(AnymalBallValveStudentRunnerCfg):
    """LSTM student (Zhang et al., CoRL 2024 style) over the same observations."""

    experiment_name = "anymal_ball_valve_student_rnn"
    student = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
