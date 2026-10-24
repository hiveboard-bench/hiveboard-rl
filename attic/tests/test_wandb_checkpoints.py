"""Checkpoint selection and upload routing, without simulation or networking."""

import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "rl"))
from wandb_checkpoints import CheckpointSelection, best_latest_wandb_checkpoints


def test_selection_keeps_best_and_latest_and_ignores_invalid_scores(tmp_path):
    selection = CheckpointSelection()
    for iteration, reward in enumerate([None, 10.0, 8.0, float("nan"), float("inf"), 10.0]):
        selection.record(str(tmp_path / f"model_{iteration}.pt"), iteration, reward)
    assert [iteration for _, iteration in selection.uploads()] == [1, 5]
    assert selection.best_reward == 10.0
    selection.record(str(tmp_path / "model_6.pt"), 6, 12.0)
    assert [iteration for _, iteration in selection.uploads()] == [6]


def test_saved_weights_are_deferred_and_deduplicated(tmp_path):
    from rsl_rl.utils.logger import Logger
    from rsl_rl.utils.wandb_log_writer import WandbLogWriter

    writer = object.__new__(WandbLogWriter)
    logger = SimpleNamespace(writer=writer, rewbuffer=deque([10.0]))
    summary = {}
    with patch.object(WandbLogWriter, "save_model") as upload, patch.object(WandbLogWriter, "stop") as stop:
        with patch("wandb.run", SimpleNamespace(summary=summary)):
            with best_latest_wandb_checkpoints():
                for iteration, score in [(0, 10.0), (100, 20.0), (200, 15.0)]:
                    path = tmp_path / f"model_{iteration}.pt"
                    path.write_bytes(b"weights")
                    logger.rewbuffer = deque([score])
                    Logger.save_model(logger, str(path), iteration)
                upload.assert_not_called()
                writer.stop()
            assert upload.call_count == 2
            assert [call.args[2] for call in upload.call_args_list] == [100, 200]
            stop.assert_called_once_with(writer)
            assert summary["Checkpoints/best_iteration"] == 100
            assert summary["Checkpoints/latest_iteration"] == 200
            assert len(list(tmp_path.glob("*.pt"))) == 3


def test_other_loggers_keep_original_upload_behavior():
    from rsl_rl.utils.logger import Logger

    logger = SimpleNamespace(writer=None)
    with patch.object(Logger, "save_model") as original:
        with best_latest_wandb_checkpoints():
            Logger.save_model(logger, "model_10.pt", 10)
        original.assert_called_once_with(logger, "model_10.pt", 10)


def test_interruption_flushes_saved_checkpoint(tmp_path):
    from rsl_rl.utils.logger import Logger
    from rsl_rl.utils.wandb_log_writer import WandbLogWriter

    writer = object.__new__(WandbLogWriter)
    logger = SimpleNamespace(writer=writer, rewbuffer=deque())
    with patch.object(WandbLogWriter, "save_model") as upload, patch("wandb.run", None):
        try:
            with best_latest_wandb_checkpoints():
                Logger.save_model(logger, str(tmp_path / "model_0.pt"), 0)
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
        upload.assert_called_once_with(writer, str((tmp_path / "model_0.pt").resolve()), 0)
