# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a trained robomimic checkpoint across a batch of environments."""

from __future__ import annotations

import torch


class RobomimicPolicy:
    """Batched inference for a robomimic checkpoint.

    Robomimic's own :class:`RolloutPolicy` hard-codes a batch size of one, which
    would force a Python loop over environments on every step. This wrapper
    reproduces its preprocessing - key selection, float cast, observation
    normalization - directly on the ``(num_envs, ...)`` tensors Isaac Lab
    already has on device.

    Args:
        checkpoint: Path to a robomimic ``.pth`` checkpoint.
        device: Torch device for inference.
    """

    def __init__(self, checkpoint: str, device: str):
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils
        import robomimic.utils.tensor_utils as TensorUtils

        self._obs_utils = ObsUtils
        self._device = torch.device(device)

        rollout_policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=checkpoint, device=self._device, verbose=False
        )
        self._rollout_policy = rollout_policy
        self._policy = rollout_policy.policy
        self._obs_keys = list(self._policy.global_config.all_obs_keys)

        stats = rollout_policy.obs_normalization_stats
        if stats is not None:
            stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(stats), self._device))
        self._obs_normalization_stats = stats

        # Training may have rescaled actions into [-1, 1] so the actor's tanh
        # could represent them. Undo that here, so callers always receive
        # actions in the environment's own units.
        from .dataset import load_action_norm, load_dataset_action_norm

        # action_norm.json only lands beside the checkpoints once training
        # ends, so an intermediate checkpoint falls back to the stats stamped
        # on the normalized dataset it trained on. Without them it would
        # silently emit [-1, 1] actions in place of joint angles.
        action_norm = load_action_norm(checkpoint)
        if action_norm is None:
            action_norm = load_dataset_action_norm(self._policy.global_config.train.data)
        if action_norm is None:
            self._action_lo = None
            self._action_span = None
        else:
            lo = torch.tensor(action_norm[0], dtype=torch.float32, device=self._device)
            hi = torch.tensor(action_norm[1], dtype=torch.float32, device=self._device)
            self._action_lo = lo
            self._action_span = hi - lo

        self.checkpoint = checkpoint

    @property
    def normalizes_actions(self) -> bool:
        """Whether this checkpoint's actions are rescaled back to task units."""
        return self._action_lo is not None

    @property
    def observation_keys(self) -> list[str]:
        """Observation keys the checkpoint was trained on."""
        return list(self._obs_keys)

    def reset(self) -> None:
        """Reset any recurrent state and put the network in eval mode.

        Note that the rollout loop calls this once per run, not per episode,
        which is correct for the MLP configuration shipped here but not for a
        recurrent one: swapping ``bc.json`` to ``rnn.enabled = true`` would also
        need this called at every episode boundary, and per environment rather
        than for the batch as a whole.
        """
        self._rollout_policy.start_episode()

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Map a batch of observations to a batch of actions.

        Args:
            obs: Observation group as ``{key: (num_envs, dim)}`` tensors.

        Returns:
            Actions of shape ``(num_envs, action_dim)``.

        Raises:
            KeyError: If the environment does not supply every key the
                checkpoint expects, which otherwise surfaces as a shape error
                deep inside the network.
        """
        missing = [key for key in self._obs_keys if key not in obs]
        if missing:
            raise KeyError(
                f"Observation group is missing {missing}, which this checkpoint was trained on. "
                f"Environment provides {sorted(obs)}. The task's observation group and the "
                "dataset's obs keys must match."
            )

        batch = {key: obs[key].to(self._device, dtype=torch.float32) for key in self._obs_keys}
        if self._obs_normalization_stats is not None:
            batch = self._obs_utils.normalize_obs(batch, obs_normalization_stats=self._obs_normalization_stats)

        with torch.no_grad():
            action = self._policy.get_action(obs_dict=batch)

        if self._action_lo is not None:
            action = self._action_lo + 0.5 * (action + 1.0) * self._action_span
        return action
