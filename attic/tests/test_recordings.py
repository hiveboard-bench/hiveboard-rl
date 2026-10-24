# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU checks for recording isolation, episode boundaries, and batch reporting."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "rl"))
import play
import record_all

from isaaclab.envs.utils import video_recorder
from isaaclab_rl.entrypoints.backends import play_rsl_rl


class RecordingTests(unittest.TestCase):
    def test_video_flag_is_forwarded_without_abbreviating_video_output(self):
        for output in (None, "logs/videos/franka"):
            with self.subTest(output=output):
                argv = ["--video", "--video_length", "240", "--agent", "rsl_rl_student_ppo_cfg_entry_point",
                        "--video_length", "340", "--stop-on-termination"]
                if output is not None:
                    argv.extend(["--video-output", output])
                with patch.object(play, "run_recording") as recording:
                    play.main(argv)
                forwarded = recording.call_args.args[0]
                self.assertIn("--video", forwarded)
                self.assertIn("340", forwarded)
                self.assertNotIn("--video-output", forwarded)
                self.assertTrue(recording.call_args.kwargs["stop_on_termination"])
                self.assertEqual(recording.call_args.kwargs["output_dir"], output)

    def check_boundary(self, terminate_at):
        captures, steps = [], []
        reset = SimpleNamespace(any=lambda: len(steps) == terminate_at)
        recorder = SimpleNamespace(_env=SimpleNamespace(reset_buf=reset))

        def simulate(step, **kwargs):
            for _ in range(5):
                step()

        def fake_cli(argv):
            def step():
                steps.append(len(steps) + 1)
                video_recorder.VideoRecorder.step(recorder)
            play_rsl_rl.run_playback(step)

        with patch.object(play, "run_play_cli", fake_cli), \
             patch.object(play_rsl_rl, "run_playback", simulate), \
             patch.object(video_recorder.VideoRecorder, "step", lambda recorder: captures.append(len(steps))):
            play.run_recording([], stop_on_termination=True)
            self.assertIs(play_rsl_rl.run_playback, simulate)
        return steps, captures

    def test_first_termination_omits_reset_frame(self):
        steps, captures = self.check_boundary(3)
        self.assertEqual(steps, [1, 2, 3])
        self.assertEqual(captures, [1, 2])

    def test_step_cap_when_no_termination(self):
        steps, captures = self.check_boundary(None)
        self.assertEqual(len(steps), 5)
        self.assertEqual(captures, steps)

    def test_publication_encoding_preserves_time(self):
        import numpy as np
        original_clip = video_recorder.ImageSequenceClip
        with tempfile.TemporaryDirectory() as folder:
            def fake_cli(argv):
                clip = video_recorder.ImageSequenceClip([np.full((16, 16, 3), 80, dtype=np.uint8)] * 2, fps=20)
                clip.write_videofile(str(Path(folder) / "clip.mp4"), codec="libx264", audio=False, logger=None)
                clip.close()
            with patch.object(play, "run_play_cli", fake_cli):
                play.run_recording([], publication=True)
            result = record_all.core.inspect_video(Path(folder) / "clip.mp4")
            self.assertAlmostEqual(result["duration_seconds"], 0.1, places=2)
            self.assertEqual(result["fps"], "50/1")
            self.assertIs(video_recorder.ImageSequenceClip, original_clip)

    def test_batch_subprocess_checks_mp4_and_reports_crashes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "logs").mkdir()
            example = {"name": "sample", "task": "task", "checkpoint": "model.pt", "timeout": 10}

            def command(example, output):
                output.mkdir(parents=True)
                return [
                    "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=16x16:r=20",
                    "-t", "0.1", "-c:v", "libx264", str(output / "clip.mp4"),
                ]

            with patch.object(record_all, "player_command", command):
                result = record_all.record_example(example, folder, 1)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["frames"], 2)
            self.assertTrue(Path(result["log"]).exists())
            with patch.object(record_all, "player_command", return_value=[sys.executable, "-c", "exit(139)"]):
                result = record_all.record_example(example, folder, 2)
            self.assertEqual(result["status"], "failed")
            self.assertIn(result["returncode"], record_all.core.CRASH_RETURNCODES)

    def test_standard_configuration(self):
        from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.env import AnymalBallValveRLEnvCfg_PLAY
        from isaaclab_hiveboard_rl.tasks.anymal.m30_thread_rl.env import AnymalM30ThreadRLEnvCfg_PLAY
        from isaaclab_hiveboard_rl.tasks.anymal.circuit_breaker_rl.env import AnymalCircuitBreakerRLEnvCfg_PLAY
        for cls in (AnymalBallValveRLEnvCfg_PLAY, AnymalM30ThreadRLEnvCfg_PLAY,
                    AnymalCircuitBreakerRLEnvCfg_PLAY):
            with self.subTest(task=cls.__name__):
                cfg = cls()
                play.configure_standard(cfg)
                self.assertTrue(cfg.events.reset_from_bank.params["standard"])
                self.assertEqual(cfg.events.arm_delay.params["delay_range_s"], (0.0, 0.0))
                self.assertEqual(cfg.events.valve_dynamics.params["stuck_prob"], 0.0)
                self.assertTrue(all(lo == hi for lo, hi in cfg.events.valve_dynamics.params["ranges"].values()))
                self.assertTrue(all(value == 0 for value in cfg.observations.policy.registered_valve.params.values()))

    def test_config_validation_and_batch_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint = folder / "model.pt"
            checkpoint.touch()
            config = folder / "recordings.yaml"
            entry = {"name": "one", "task": "task-one", "checkpoint": str(checkpoint)}
            config.write_text(yaml.safe_dump({"examples": [entry, entry]}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                record_all.load_examples(config)
            config.write_text(yaml.safe_dump({"examples": [entry, {**entry, "name": "two"}]}))
            examples = record_all.load_examples(config)
            command = record_all.player_command(examples[0], folder / "clips")
            self.assertIn("--standard", command)
            self.assertNotIn("--standard", record_all.player_command({**examples[0], "standard": False}, folder))
            self.assertIn("--stop-on-termination", command)
            self.assertEqual(command[command.index("--num_envs") + 1], "1")
            results = [{"status": "failed", "error": "test failure"}, {"status": "ok", "video": "test.mp4"}]
            with (
                patch.object(record_all, "record_example", side_effect=results) as record,
                redirect_stdout(io.StringIO()),
            ):
                status = record_all.main(["--config", str(config), "--output", str(folder / "output")])
            self.assertEqual(status, 1)
            self.assertEqual(record.call_count, 2)
            summary = json.loads(next((folder / "output").glob("*/summary.json")).read_text())
            self.assertEqual([r["status"] for r in summary["results"]], ["failed", "ok"])


if __name__ == "__main__":
    unittest.main()
