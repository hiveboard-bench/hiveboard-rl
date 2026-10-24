# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL model variants used by the ball-valve distillation."""

import torch
from rsl_rl.models import MLPModel


class ClippedMLPModel(MLPModel):
    """MLP whose output is clamped to ``[-1, 1]``, the range the environment executes.

    Used as the distillation *teacher*. PPO trains with ``clip_actions = 1``, so
    the teacher's raw means drift far outside it without consequence (the
    ball-valve teacher's gripper output is beyond +-1 98% of the time, arm
    outputs peak at 15). Regressing those raw values makes the student's MSE
    chase magnitudes that are clipped away anyway; the clamped targets are the
    actions the teacher actually applied. No parameters are added, so a PPO
    ``actor_state_dict`` loads unchanged.
    """

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return super().forward(*args, **kwargs).clamp(-1.0, 1.0)
