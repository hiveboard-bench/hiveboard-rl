# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO that trains only the critic for its first updates, to fine-tune a pretrained actor.

A behaviour-cloned actor (``scripts/imitation/train_bank_bc.py``) comes without a critic that
knows the task's rollouts: the critic fitted offline to the expert's returns normalizes its
privileged inputs by the expert's statistics, and the actor's own deviations fall far outside
them. PPO's advantages are then noise, and the first updates undo the cloned behaviour.
``critic_warmup_updates`` updates keep the actor (and its std) fixed and the learning rate
schedule paused, so the critic first fits the actor's own returns, at
``critic_warmup_learning_rate``.

The actor then starts at the configured ``learning_rate``, whatever a loaded checkpoint's
optimizer stored, so a small one protects the cloned actor: Adam's first steps move every
parameter by about the learning rate whatever its gradient, and at 5e-4 one update made the
BallValve BC actor's arm 25x less accurate. The adaptive schedule raises it as the KL allows.

``freeze_actor_normalization`` stops the actor's observation normalizer from updating. It is part of
the cloned function: updated on PPO's rollouts (half of them starting mid-trajectory, unlike the
demonstrations), it shifted some inputs by up to 4 standard deviations within 100 iterations and
made the unchanged BC actor's arm 5x less accurate.
"""

from isaaclab.utils.configclass import configclass
from rsl_rl.algorithms import PPO

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


class CriticWarmupPPO(PPO):
    """:class:`rsl_rl.algorithms.PPO` whose first ``critic_warmup_updates`` updates leave the actor unchanged."""

    def __init__(
        self,
        *args,
        critic_warmup_updates: int = 0,
        critic_warmup_learning_rate: float | None = None,
        freeze_actor_normalization: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.critic_warmup_updates = critic_warmup_updates
        self.critic_warmup_learning_rate = critic_warmup_learning_rate
        normalizer = getattr(self._raw_actor, "obs_normalizer", None)
        if freeze_actor_normalization and hasattr(normalizer, "until"):
            # EmpiricalNormalization stops updating once its count reaches ``until``; a loaded checkpoint keeps it.
            normalizer.until = 0
        # Captured before a checkpoint's optimizer state can replace it (PPO.load).
        self.actor_learning_rate = self.learning_rate

    def _set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def update(self) -> dict[str, float]:
        if self.critic_warmup_updates <= 0:
            return super().update()
        self.critic_warmup_updates -= 1
        actor = list(self.actor.parameters())
        schedule = self.schedule
        # No gradient reaches the actor; the KL-adaptive learning rate would read a zero KL and grow.
        for parameter in actor:
            parameter.requires_grad_(False)
        self.schedule = "fixed"
        if self.critic_warmup_learning_rate is not None:
            self._set_learning_rate(self.critic_warmup_learning_rate)
        try:
            return super().update()
        finally:
            for parameter in actor:
                parameter.requires_grad_(True)
            self.schedule = schedule
            if self.critic_warmup_updates == 0:
                self._set_learning_rate(self.actor_learning_rate)


@configclass
class CriticWarmupPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """:class:`RslRlPpoAlgorithmCfg` for :class:`CriticWarmupPPO`; 0 warm-up updates is plain PPO."""

    class_name: str = f"{__name__}:CriticWarmupPPO"
    critic_warmup_updates: int = 0
    """Updates (one per iteration) that train only the critic."""
    critic_warmup_learning_rate: float | None = None
    """Learning rate of the warm-up updates; None keeps ``learning_rate``."""
    freeze_actor_normalization: bool = False
    """Keep the actor's observation normalization as loaded (fine-tuning a cloned actor)."""
