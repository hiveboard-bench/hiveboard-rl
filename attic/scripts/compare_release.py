#!/usr/bin/env python3
"""Matched BallValve cap-only / cap-plus-velocity fine-tuning pilots on W&B.

Both runs resume the same checkpoint and seed. Evaluate each on identical
bank-start seeds and record a standard-condition release trace. All commands,
results, provenance and trajectories are saved locally and on W&B.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from wandb.sdk.lib.runid import generate_id

import wandb

ROOT = Path(__file__).resolve().parents[2]


def run_command(command, log, env):
    with log.open("w") as stream:
        stream.write(" ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def scalars(data, prefix="Evaluation"):
    result = {}
    for key, value in data.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            result.update(scalars(value, name))
        elif isinstance(value, (float, int)) and math.isfinite(value):
            result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    batch = ROOT / "logs" / "release_comparison" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch.mkdir(parents=True)
    report = {"source_checkpoint": str(checkpoint), "iterations": args.iterations, "seed": args.seed, "results": []}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    base = [sys.executable]
    for weight in (0.0, 1.0):
        name = (
            f"ballvalve_release_cap010_{'only' if weight == 0 else 'veltrack_w1_std05'}"
            f"_from999_{args.iterations}it_seed{args.seed}"
        )
        directory = batch / name
        directory.mkdir()
        run_id = generate_id()
        env = dict(
            os.environ,
            WANDB_MODE="online",
            WANDB_RUN_ID=run_id,
            WANDB_RUN_GROUP="ballvalve_release_cap_vs_velocity",
            PYTHONUNBUFFERED="1",
        )
        overrides = [
            "env.actions.arm_action.max_position_error=0.1",
            f"env.rewards.track_expert_velocity.weight={weight}",
        ]
        train = [
            *base,
            "scripts/rl/train.py",
            "--task",
            "Isaac-HiveBoard-Anymal-BallValve-RL-v0",
            "--agent",
            "rsl_rl_student_ppo_cfg_entry_point",
            "--checkpoint",
            str(checkpoint),
            "--num_envs",
            "4096",
            "--max_iterations",
            str(args.iterations),
            "--seed",
            str(args.seed),
            "--logger",
            "wandb",
            "--log_project_name",
            "isaaclab-hiveboard",
            "--run_name",
            name,
            *overrides,
        ]
        entry = dict(
            name=name, wandb_id=run_id, state="training", velocity_weight=weight, command=train, git_commit=revision
        )
        report["results"].append(entry)
        report_path = batch / "summary.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Training {name} (W&B {run_id}); logs: {directory}", flush=True)
        run_command(train, directory / "train.log", env)
        runs = list((ROOT / "logs/rsl_rl/anymal_ball_valve_student_ppo").glob(f"*_{name}"))
        if len(runs) != 1:
            raise RuntimeError(f"Expected one training directory for {name}: {runs}")
        final = max(runs[0].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        entry.update(state="evaluating", checkpoint=str(final))
        report_path.write_text(json.dumps(report, indent=2))
        evaluation_path = directory / "evaluation.json"
        evaluate = [
            *base,
            "scripts/rl/evaluate.py",
            "--checkpoint",
            str(final),
            "--agent",
            "rsl_rl_student_ppo_cfg_entry_point",
            "--episodes",
            str(args.episodes),
            "--num_envs",
            str(min(args.episodes, 100)),
            "--seed",
            str(args.seed),
            "--output",
            str(evaluation_path),
            *overrides,
        ]
        run_command(evaluate, directory / "evaluate.log", env)
        trace = [
            *base,
            "scripts/rl/trace_policy.py",
            "--checkpoint",
            str(final),
            "--seed",
            str(args.seed),
            "--output",
            str(directory / "trace"),
            *overrides,
        ]
        run_command(trace, directory / "trace.log", env)
        data = np.load(directory / "trace/trajectory.npz")
        transitions = np.flatnonzero((data["action"][1:, -1] > 0) & (data["action"][:-1, -1] <= 0)) + 1
        # Ignore transient gripper changes before the mechanism opens.
        transitions = transitions[np.abs(data["valve"][transitions]) > 1.4]
        release = float(data["time"][transitions[0]]) if len(transitions) else None
        window = (
            (data["time"] >= release - 0.2) & (data["time"] <= release + 0.8)
            if release is not None
            else np.zeros(len(data["time"]), dtype=bool)
        )
        metrics = scalars(json.loads(evaluation_path.read_text()))
        metrics.update(
            {
                "Diagnostics/final_valve_rad": float(data["valve"][-1]),
                "Diagnostics/peak_joint_speed_rad_s": float(np.abs(data["velocity"]).max()),
            }
        )
        if release is not None:
            metrics.update(
                {
                    "Diagnostics/release_time_s": release,
                    "Diagnostics/release_peak_joint_speed_rad_s": float(np.abs(data["velocity"][window]).max()),
                }
            )
        # Attach evaluation to the same run rather than creating unrelated W&B runs.
        with wandb.init(
            project="isaaclab-hiveboard",
            id=run_id,
            resume="must",
            name=name,
            group="ballvalve_release_cap_vs_velocity",
            config={
                "release_comparison_commit": revision,
                "source_checkpoint": str(checkpoint),
                "pilot_iterations": args.iterations,
                "evaluation_seed": args.seed,
                "evaluation_episodes": args.episodes,
            },
        ) as run:
            run.summary.update(metrics)
            speed = np.abs(data["velocity"]).max(axis=1)
            reference_speed = np.abs(data["velocity_reference"]).max(axis=1)
            run.log(
                {
                    "Diagnostics/joint_speed": wandb.plot.line_series(
                        xs=data["time"].tolist(),
                        ys=[speed.tolist(), reference_speed.tolist()],
                        keys=["policy", "expert"],
                        title="Standard-condition joint speed",
                        xname="time_s",
                    )
                }
            )
            artifact = wandb.Artifact(f"{name}-diagnostics", type="evaluation")
            artifact.add_file(str(evaluation_path))
            artifact.add_dir(str(directory / "trace"), name="trace")
            run.log_artifact(artifact)
            entry.update(state="completed", metrics=metrics, wandb_url=run.url)
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Completed {name}: {entry['wandb_url']}", flush=True)
    print(f"Comparison report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
