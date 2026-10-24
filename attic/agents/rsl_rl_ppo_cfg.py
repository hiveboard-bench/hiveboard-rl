# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO teacher: actor and critic both read the privileged ``teacher`` group."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlRndCfg

from .critic_warmup import CriticWarmupPpoAlgorithmCfg

TEACHER_HIDDEN_DIMS = [512, 256, 128]


@configclass
class AnymalBallValveTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 100
    experiment_name = "anymal_ball_valve_teacher"
    logger = "wandb"
    wandb_project = "isaaclab-hiveboard"
    clip_actions = 1.0
    obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=TEACHER_HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=TEACHER_HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
    )
    # Plain PPO; agent.algorithm.critic_warmup_updates=<n> first trains only the critic (BC fine-tuning).
    algorithm = CriticWarmupPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AnymalBallValveStudentPPORunnerCfg(AnymalBallValveTeacherPPORunnerCfg):
    """Deployable student trained with PPO directly (asymmetric actor-critic).

    The actor reads only the deployable ``policy`` group; the critic keeps the
    privileged ``teacher`` group. Same rewards and terminations as the
    teacher, so there is no imitation step: distilling teacher v23 left the
    students unable to copy its gripper-close decision (students v9-v11).
    """

    experiment_name = "anymal_ball_valve_student_ppo"
    obs_groups = {"actor": ["policy"], "critic": ["teacher"]}


@configclass
class AnymalBallValveStudentPPORNDRunnerCfg(AnymalBallValveStudentPPORunnerCfg):
    """The PPO student plus random network distillation (RND) curiosity.

    RND pays an intrinsic reward for states its predictor network has not
    learned yet, read from the privileged ``teacher`` group (training only;
    the deployed actor still reads ``policy``). Normalized, it is weighted
    ~10% of the task reward per step, held for 500 iterations, then faded to
    0 by iteration 2000 so the final policy is trained on the task reward alone.
    The schedule counts environment steps (``num_steps_per_env`` per iteration).
    """

    obs_groups = {"actor": ["policy"], "critic": ["teacher"], "rnd_state": ["teacher"]}

    def __post_init__(self):
        super().__post_init__()
        steps = self.num_steps_per_env
        self.algorithm.rnd_cfg = RslRlRndCfg(
            weight=0.1,
            weight_schedule=RslRlRndCfg.LinearWeightScheduleCfg(
                final_value=0.0, initial_step=500 * steps, final_step=2000 * steps
            ),
            reward_normalization=True,
            state_normalization=True,
            learning_rate=1.0e-3,
            num_outputs=16,
            predictor_hidden_dims=[256, 256],
            target_hidden_dims=[256, 256],
        )
