#!/usr/bin/env python3
"""Run production collection and three-seed PPO after the nominal/pilot gates pass.

Every subprocess uses the existing pipeline scripts. Progress, commands, logs,
replay metrics and acceptance results are retained in the output directory.
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TASK = "Isaac-HiveBoard-Franka-BallValve-RL-v0"
PLAY = TASK.replace("-RL-v0", "-RL-Play-v0")
AGENT = "rsl_rl_student_ppo_cfg_entry_point"


def check_replay(path, count):
    result = json.loads(path.read_text())
    if result["task"] != PLAY or result["command"] != "increment":
        raise ValueError(f"{path}: expected FR3 integrated-action replay")
    if len({e["trajectory"] for e in result["episodes"]}) < count:
        raise ValueError(f"{path}: must cover {count} different trajectories")
    if result["summary"]["complete"] < 0.95:
        raise ValueError(f"{path}: complete replay rate is below 95%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--collection-envs", type=int, default=64)
    parser.add_argument("--training-envs", type=int, default=1024)
    parser.add_argument(
        "--dry-run", action="store_true", help="Check input gates and print commands without running them."
    )
    args = parser.parse_args()
    if args.collection_envs < 1 or args.training_envs < 1:
        parser.error("Environment counts must be positive")
    for name, count in (("nominal_increment", 20), ("pilot_replay", 200)):
        check_replay(ROOT / f"logs/expert_bank/franka_{name}.json", count)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "logs/franka_pipeline" / stamp).resolve()
    if not args.dry_run:
        if not torch.cuda.is_available():
            raise RuntimeError("Production training requires GPU access")
        output.mkdir(parents=True, exist_ok=False)
    state = {
        "state": "running",
        "collection_envs": args.collection_envs,
        "training_envs": args.training_envs,
        "iterations": 3000,
        "evaluations": [],
    }

    def status(**values):
        state.update(values)
        state["updated_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        if not args.dry_run:
            temp = output / "status.tmp"
            temp.write_text(json.dumps(state, indent=2))
            temp.replace(output / "status.json")

    def run(stage, script, *flags):
        command = [sys.executable, str(ROOT / "scripts/rl" / script), *map(str, flags)]
        status(stage=stage, command=command)
        print(f"[{stage}] {command}", flush=True)
        if not args.dry_run:
            with (output / f"{stage}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    def bank_override(bank):
        return "env.events.reset_from_bank.params.path=" + json.dumps(str(bank))

    def collect(name, count, seed):
        bank = output / f"{name}.pt"
        run(
            name,
            "build_expert_bank.py",
            "--task",
            TASK,
            "--num_envs",
            args.collection_envs,
            "--plan_batch_size",
            min(args.collection_envs, 32),
            "--num_trajectories",
            count,
            "--max_waves",
            200,
            "--seed",
            seed,
            "--no_wrist_flip",
            "--cuda_graph",
            "--output",
            bank,
        )
        if not args.dry_run:
            raw = torch.load(bank, map_location="cpu", weights_only=False)
            if len(raw["arm_q"]) != count:
                raise ValueError(f"{bank}: collection did not reach {count} trajectories")
        return bank

    try:
        bank = collect("production_bank", 5000, 20)
        replay = output / "production_replay.json"
        run(
            "production_replay",
            "replay_expert_bank.py",
            "--task",
            PLAY,
            "--bank",
            bank,
            "--num_envs",
            256,
            "--episodes",
            5000,
            "--output",
            replay,
            "--viz",
            "none",
            "env.sim.physics.use_cuda_graph=true",
        )
        if not args.dry_run:
            check_replay(replay, 5000)
        heldout = collect("heldout_bank", 300, 10020)
        for seed in (0, 1, 2):
            runs = ROOT / "logs/rsl_rl/franka_ball_valve_student_ppo"
            before = set(runs.iterdir()) if runs.exists() else set()
            run_name = f"{output.name}_seed_{seed}"
            run(
                f"train_seed_{seed}",
                "train.py",
                "--task",
                TASK,
                "--agent",
                AGENT,
                "--num_envs",
                args.training_envs,
                "--max_iterations",
                3000,
                "--seed",
                seed,
                "--run_name",
                run_name,
                "--logger",
                "tensorboard",
                "--viz",
                "none",
                bank_override(bank),
                "env.sim.physics.use_cuda_graph=true",
            )
            if args.dry_run:
                checkpoint = runs / f"<run>_{run_name}" / "model_2999.pt"
            else:
                created = [p for p in set(runs.iterdir()) - before if p.name.endswith("_" + run_name)]
                if len(created) != 1:
                    raise RuntimeError(f"Expected one new run for {run_name}, found {created}")
                checkpoint = created[0] / "model_2999.pt"
            report = output / f"evaluation_seed_{seed}.json"
            run(
                f"evaluate_seed_{seed}",
                "evaluate.py",
                "--task",
                PLAY,
                "--agent",
                AGENT,
                "--checkpoint",
                checkpoint,
                "--episodes",
                300,
                "--num_envs",
                100,
                "--seed",
                100 + seed,
                "--export",
                "--output",
                report,
                bank_override(heldout),
                "env.sim.physics.use_cuda_graph=true",
            )
            if not args.dry_run:
                result = json.loads(report.read_text())
                state["evaluations"].append(
                    {
                        "seed": seed,
                        "checkpoint": str(checkpoint),
                        "success_rate": result["success_rate"],
                        "complete_rate": result["stage_rates"]["complete"],
                    }
                )
                status()
        passed = all(e["success_rate"] >= 0.9 and e["complete_rate"] >= 0.9 for e in state["evaluations"])
        status(state="passed" if passed else "acceptance_failed", stage="finished")
        return 0 if passed else 1
    except (Exception, KeyboardInterrupt) as error:
        status(state="failed", error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
