# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One rollout loop shared by demo collection, DAgger rounds and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass
class RolloutResult:
    """Outcome of a rollout, counted per episode rather than per step."""

    steps: int = 0
    #: Episodes the recorder wrote out as successful.
    exported_successes: int = 0
    #: Episodes the recorder wrote out as failures (or rejected, when the
    #: recorder only exports successes).
    exported_failures: int = 0
    #: Success flags observed at termination, in completion order. Populated
    #: even when no recorder is active, so evaluation does not need one.
    episode_successes: list[bool] = field(default_factory=list)

    @property
    def episodes(self) -> int:
        """Number of episodes that finished during the rollout."""
        return len(self.episode_successes)

    @property
    def success_rate(self) -> float:
        """Fraction of finished episodes that terminated in success."""
        if not self.episode_successes:
            return float("nan")
        return sum(self.episode_successes) / len(self.episode_successes)


def run_rollout(
    env,
    actor: Callable[[dict], torch.Tensor],
    *,
    max_steps: int,
    expert=None,
    stop_after_successes: int | None = None,
    stop_after_episodes: int | None = None,
    progress: Callable[[RolloutResult], None] | None = None,
    progress_every: int = 200,
) -> RolloutResult:
    """Step ``env`` under ``actor`` until a stopping condition is reached.

    Terminated environments are reset inside :meth:`ManagerBasedRLEnv.step`,
    which is also where the recorder exports the finished episode, so this loop
    only has to step and count.

    Args:
        env: The environment to drive.
        actor: Maps the current observation dict to an action batch. For expert
            collection this is the expert itself; for DAgger it is the learner.
        max_steps: Hard cap on environment steps, so a policy that never
            terminates cannot hang the run.
        expert: Optional expert queried at every state. Its action is stashed on
            the environment as ``expert_actions`` for
            :class:`PreStepExpertActionsRecorder` to record, which is what makes
            a DAgger round store corrections rather than the learner's own
            mistakes. Must be set whenever the recorder relabels.
        stop_after_successes: Stop once this many successful episodes have been
            exported.
        stop_after_episodes: Stop once this many episodes have finished.
        progress: Called every ``progress_every`` steps and on every episode
            boundary, for logging.
        progress_every: Step interval between ``progress`` calls.

    Returns:
        The rollout tally.
    """
    base = env.unwrapped
    recorder = getattr(base, "recorder_manager", None)
    has_recorder = recorder is not None and len(recorder.active_terms) > 0

    result = RolloutResult()
    obs, _ = env.reset()

    while result.steps < max_steps:
        # The expert must be queried *before* stepping: it labels the state the
        # actor is about to act on, not the one the step lands in.
        if expert is not None:
            base.expert_actions = expert.compute()

        with torch.inference_mode():
            action = actor(obs)
            obs, _, terminated, truncated, _ = env.step(action)

        result.steps += 1

        done = terminated | truncated
        if torch.any(done):
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            # `terminated` is the union of non-timeout termination terms, so read
            # the success term directly rather than inferring it from `terminated`.
            if "success" in base.termination_manager.active_terms:
                success = base.termination_manager.get_term("success")
                result.episode_successes.extend(bool(success[i]) for i in done_ids.tolist())
            else:
                result.episode_successes.extend(bool(terminated[i]) for i in done_ids.tolist())

            if has_recorder:
                result.exported_successes = recorder.exported_successful_episode_count
                result.exported_failures = recorder.exported_failed_episode_count
            if progress is not None:
                progress(result)

        elif progress is not None and result.steps % max(progress_every, 1) == 0:
            progress(result)

        if stop_after_successes is not None:
            reached = result.exported_successes if has_recorder else sum(result.episode_successes)
            if reached >= stop_after_successes:
                break
        if stop_after_episodes is not None and result.episodes >= stop_after_episodes:
            break

    return result
