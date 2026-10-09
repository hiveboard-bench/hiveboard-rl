"""Upload only the best and latest saved checkpoints when training ends."""

import math
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


@dataclass
class CheckpointSelection:
    """Best means highest finite rolling mean return among saved checkpoints."""

    latest: tuple[str, int] | None = None
    best: tuple[str, int] | None = None
    best_reward: float | None = None

    def record(self, path: str, iteration: int, reward: float | None):
        self.latest = (str(Path(path).resolve()), iteration)
        if reward is not None and math.isfinite(reward) and (self.best_reward is None or reward > self.best_reward):
            self.best = self.latest
            self.best_reward = reward

    def uploads(self):
        # Preserve model_<iteration>.pt for Isaac Lab's W&B checkpoint resolver.
        return list(dict.fromkeys(item for item in (self.best, self.latest) if item is not None))


@contextmanager
def best_latest_wandb_checkpoints():
    """Scope the upload policy to this training entrypoint, preserving local saves.

    Metrics, config, videos and other loggers retain their normal behavior.
    On interruption, upload the selected checkpoints already saved to disk.
    """
    from rsl_rl.utils.logger import Logger
    from rsl_rl.utils.wandb_log_writer import WandbLogWriter

    import wandb

    original_save = Logger.save_model
    original_upload = WandbLogWriter.save_model
    original_stop = WandbLogWriter.stop
    selections = {}

    def save_model(logger, path, it):
        writer = logger.writer
        if not isinstance(writer, WandbLogWriter):
            original_save(logger, path, it)
            return
        selection = selections.setdefault(writer, CheckpointSelection())
        reward = statistics.mean(logger.rewbuffer) if logger.rewbuffer else None
        selection.record(path, it, reward)

    def flush(writer):
        selection = selections.get(writer)
        if selection is None:
            return
        for path, iteration in selection.uploads():
            original_upload(writer, path, iteration)
        if wandb.run is not None:
            wandb.run.summary.update(
                {
                    "Checkpoints/upload_policy": "best_and_latest_at_training_end",
                    "Checkpoints/best_metric": "rolling_mean_training_episode_return_at_save",
                    "Checkpoints/best_reward": selection.best_reward,
                    "Checkpoints/best_file": Path(selection.best[0]).name if selection.best else None,
                    "Checkpoints/best_iteration": selection.best[1] if selection.best else None,
                    "Checkpoints/latest_file": Path(selection.latest[0]).name if selection.latest else None,
                    "Checkpoints/latest_iteration": selection.latest[1] if selection.latest else None,
                }
            )
        del selections[writer]

    def stop(writer):
        flush(writer)
        original_stop(writer)

    with patch.object(Logger, "save_model", save_model), patch.object(WandbLogWriter, "stop", stop):
        try:
            yield
        finally:
            for writer in list(selections):
                flush(writer)
