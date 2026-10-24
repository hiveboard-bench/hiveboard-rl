#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mirror an RSL-RL run's TensorBoard scalars into Weights & Biases, live.

For runs started with the TensorBoard logger (new runs log to wandb directly,
see the agents' ``logger``). Follows the run's event file, uploads what is
already there, then keeps uploading new iterations until the file has been
quiet for ``--idle_exit_s``. The wandb run is named after the run directory and
gets the run's ``params/agent.yaml`` as its config::

    uv run python scripts/rl/wandb_mirror.py logs/rsl_rl/anymal_ball_valve_teacher/<run>
"""

import argparse
import glob
import os
import time

import yaml
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util

import wandb

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run_dir", help="RSL-RL run directory (holds events.out.tfevents.*).")
parser.add_argument("--project", default="isaaclab-hiveboard", help="wandb project.")
parser.add_argument("--poll_s", type=float, default=10.0, help="Seconds between reads of the event file.")
parser.add_argument("--idle_exit_s", type=float, default=600.0, help="Stop after this long without new events.")
args = parser.parse_args()


def scalar(value) -> float | None:
    """The number in a summary value, whether stored as simple_value or a scalar tensor."""
    if value.HasField("simple_value"):
        return float(value.simple_value)
    if value.HasField("tensor"):
        array = tensor_util.make_ndarray(value.tensor)
        if array.size == 1:
            return float(array.reshape(-1)[0])
    return None


def main() -> None:
    run_dir = os.path.abspath(args.run_dir)
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"No TensorBoard event file in {run_dir}")
    config = {}
    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.exists(agent_yaml):
        with open(agent_yaml) as f:
            config = yaml.unsafe_load(f)
    name = os.path.basename(run_dir)
    # A stable id resumes the same wandb run if the mirror is restarted.
    run = wandb.init(project=args.project, name=name, id=f"{name}-tb", resume="allow", config=config)
    loader = EventFileLoader(files[-1])
    step, row, last_event = None, {}, time.time()
    logged = 0
    while True:
        for event in loader.Load():
            last_event = time.time()
            for value in event.summary.value:
                # RSL-RL also logs "<tag>/time" against wall-clock seconds,
                # not the iteration; mixing those in breaks wandb's step order.
                number = scalar(value)
                if number is None or value.tag.endswith("/time"):
                    continue
                if step is not None and event.step != step and row:
                    run.log(row, step=step)
                    logged += 1
                    row = {}
                step = event.step
                row[value.tag] = number
        if time.time() - last_event > args.idle_exit_s:
            break
        time.sleep(args.poll_s)
    if row:
        run.log(row, step=step)
        logged += 1
    print(f"[wandb_mirror] uploaded {logged} iterations of {name} to {run.url}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
