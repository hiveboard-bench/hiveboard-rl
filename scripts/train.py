import importlib
import math
import os
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

importlib.import_module("isaaclab_hiveboard_rl")
from isaaclab_rl import run_train_cli


class CheckpointSelection:
    def __init__(self):
        self.latest = None
        self.best = None
        self.best_reward = None

    def record(self, path, iteration, reward):
        self.latest = (str(Path(path).resolve()), iteration)
        finite = reward is not None and math.isfinite(reward)
        if finite and (self.best_reward is None or reward > self.best_reward):
            self.best = self.latest
            self.best_reward = reward

    def uploads(self):
        return list(dict.fromkeys(i for i in (self.best, self.latest) if i is not None))


@contextmanager
def best_latest_wandb_checkpoints():
    import wandb
    from rsl_rl.utils.logger import Logger
    from rsl_rl.utils.wandb_log_writer import WandbLogWriter

    original_save = Logger.save_model
    original_upload = WandbLogWriter.save_model
    original_stop = WandbLogWriter.stop
    selections = {}

    def save_model(logger, path, it):
        if not isinstance(logger.writer, WandbLogWriter):
            original_save(logger, path, it)
            return
        selection = selections.setdefault(logger.writer, CheckpointSelection())
        reward = statistics.mean(logger.rewbuffer) if logger.rewbuffer else None
        selection.record(path, it, reward)

    def flush(writer):
        selection = selections.pop(writer, None)
        if selection is None:
            return
        for path, iteration in selection.uploads():
            original_upload(writer, path, iteration)
        if wandb.run is not None:
            best = selection.best or (None, None)
            latest = selection.latest or (None, None)
            wandb.run.summary.update(
                {
                    "Checkpoints/upload_policy": "best_and_latest_at_training_end",
                    "Checkpoints/best_metric": "rolling_mean_training_episode_return_at_save",
                    "Checkpoints/best_reward": selection.best_reward,
                    "Checkpoints/best_file": best[0] and Path(best[0]).name,
                    "Checkpoints/best_iteration": best[1],
                    "Checkpoints/latest_file": latest[0] and Path(latest[0]).name,
                    "Checkpoints/latest_iteration": latest[1],
                }
            )

    def stop(writer):
        flush(writer)
        original_stop(writer)

    with (
        patch.object(Logger, "save_model", save_model),
        patch.object(WandbLogWriter, "stop", stop),
    ):
        try:
            yield
        finally:
            for writer in list(selections):
                flush(writer)


if __name__ == "__main__":
    if "WANDB_RUN_ID" not in os.environ:
        from wandb.sdk.lib.runid import generate_id

        os.environ["WANDB_RUN_ID"] = generate_id()
    with best_latest_wandb_checkpoints():
        argv = ["--rl_library", "rsl_rl", *sys.argv[1:]]
        if not any(token.startswith(("physics=", "presets=")) for token in argv):
            argv.append("physics=newton_mjwarp")
        sys.exit(run_train_cli(argv) or 0)
