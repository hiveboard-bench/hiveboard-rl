#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the first episode of each YAML-configured RL checkpoint."""

import argparse
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Reuse the core recorder's MP4 checks and process-group cleanup without launching Isaac Lab.
_spec = importlib.util.spec_from_file_location(
    "core_recordings", ROOT / "dependencies/isaaclab-hiveboard/scripts/record_all_envs.py"
)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)


def load_examples(path):
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("examples"), list):
        raise ValueError("Config must contain an examples list.")
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping.")
    examples, names = [], set()
    allowed = {"name", "task", "checkpoint", "agent", "publication", "video_length", "seed", "device",
               "timeout", "crash_retries", "standard", "args"}
    for entry in data["examples"]:
        if not isinstance(entry, dict):
            raise ValueError("Each example must be a mapping.")
        example = {**defaults, **entry}
        if set(example) - allowed:
            raise ValueError(f"Unknown config fields: {sorted(set(example) - allowed)}")
        for key in ("name", "task", "checkpoint"):
            if not isinstance(example.get(key), str) or not example[key]:
                raise ValueError(f"Each example needs a nonempty {key}.")
        name = example["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError(f"Invalid or duplicate example name: {name}")
        names.add(name)
        checkpoint = Path(example["checkpoint"])
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint does not exist: {checkpoint}")
        example["checkpoint"] = str(checkpoint.resolve())
        for key, minimum in (("video_length", 1), ("timeout", 1), ("crash_retries", 0)):
            value = example.get(key, {"video_length": 400, "timeout": 900, "crash_retries": 2}[key])
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name}: {key} must be an integer >= {minimum}.")
            example[key] = value
        if type(example.get("publication", True)) is not bool:
            raise ValueError(f"{name}: publication must be true or false.")
        if type(example.get("standard", True)) is not bool:
            raise ValueError(f"{name}: standard must be true or false.")
        extra = example.get("args", [])
        if not isinstance(extra, list) or not all(isinstance(arg, str) for arg in extra):
            raise ValueError(f"{name}: args must be a list of strings.")
        if any(arg.split("=", 1)[0] in {"--num_envs", "--video-output", "--video_interval"} for arg in extra):
            raise ValueError(f"{name}: args cannot override recording isolation or clip scheduling.")
        examples.append(example)
    if not examples:
        raise ValueError("No examples configured.")
    return examples


def player_command(example, output):
    command = [sys.executable, "-u", str(ROOT / "scripts/rl/play.py"),
               "--task", example["task"], "--checkpoint", example["checkpoint"],
               "--agent", example.get("agent", "rsl_rl_student_ppo_cfg_entry_point"),
               "--num_envs", "1", "--video_length", str(example["video_length"]),
               "--video_interval", "0", "--stop-on-termination", "--video-output", str(output),
               "--seed", str(example.get("seed", 0)), "--device", example.get("device", "cuda:0")]
    if example.get("standard", True):
        command.append("--standard")
    command.extend(["--publication"] if example.get("publication", True) else ["--viz", "newton"])
    return [*command, *example.get("args", [])]


def record_example(example, output, attempt):
    clip_dir = output / example["name"] / f"attempt-{attempt}"
    log_path = output / "logs" / f"{example['name']}-{attempt}.log"
    command = player_command(example, clip_dir)
    result = {"name": example["name"], "task": example["task"], "checkpoint": example["checkpoint"],
              "command": command, "log": str(log_path), "status": "failed", "attempts": attempt}
    start, proc = time.monotonic(), None
    with log_path.open("w") as log:
        try:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result["returncode"] = proc.wait(timeout=example["timeout"])
            if result["returncode"]:
                raise RuntimeError(f"Player exited with status {result['returncode']}.")
            clips = list(clip_dir.glob("*.mp4"))
            if len(clips) != 1:
                raise RuntimeError(f"Expected one MP4, found {len(clips)}.")
            result.update(core.inspect_video(clips[0]))
            result["status"] = "ok"
        except subprocess.TimeoutExpired:
            result.update(status="timeout", error=f"Exceeded {example['timeout']} wall seconds.")
        except KeyboardInterrupt:
            result.update(status="interrupted", error="Recording interrupted by the user.")
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            result["error"] = str(exc)
        finally:
            if proc is not None:
                core.stop_process_group(proc)
    result["wall_seconds"] = round(time.monotonic() - start, 3)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/recordings.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "videos/examples")
    parser.add_argument("--task", action="append", help="Exact task ID; repeat to select several.")
    parser.add_argument("--match", default="", help="Substring filter on example names or task IDs.")
    parser.add_argument("--randomized", action="store_true", help="Use randomized task starts and dynamics.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        examples = load_examples(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    examples = [e for e in examples if (not args.task or e["task"] in args.task)
                and args.match.lower() in (e["name"] + " " + e["task"]).lower()]
    if not examples:
        parser.error("No configured examples match the selection.")
    if args.randomized:
        examples = [{**e, "standard": False} for e in examples]
    output = args.output.resolve() / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    if args.list or args.dry_run:
        for example in examples:
            print(f"{example['name']}: {example['task']} -> {example['checkpoint']}" if args.list
                  else shlex.join(player_command(example, output / example["name"] / "attempt-1")))
        return 0
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            parser.error(f"{binary} must be available on PATH.")
    (output / "logs").mkdir(parents=True)
    report = {"config": str(args.config.resolve()), "examples": examples, "results": []}
    report_path = output / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recording {len(examples)} examples to {output}", flush=True)
    for index, example in enumerate(examples, 1):
        print(f"[{index}/{len(examples)}] {example['name']}", flush=True)
        for attempt in range(1, example["crash_retries"] + 2):
            result = record_example(example, output, attempt)
            if result.get("returncode") not in core.CRASH_RETURNCODES or attempt > example["crash_retries"]:
                break
            print("  crashed; retrying", flush=True)
        report["results"].append(result)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  {result['status']}: {result.get('video', result.get('error'))}", flush=True)
        if result["status"] == "interrupted":
            return 130
    passed = sum(result["status"] == "ok" for result in report["results"])
    print(f"Saved {passed}/{len(examples)} videos. Report: {report_path}", flush=True)
    return 0 if passed == len(examples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
