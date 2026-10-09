#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print the Hydra overrides a behaviour-cloned checkpoint must run with.

Reads the ``bc.json`` that ``train_bank_bc.py`` writes beside ``model_0.pt``. An
absolute- or relative-command policy needs its arm command and no action clipping
(clipping would cut its targets to +-1 rad); an ``increment`` one needs nothing, like PPO.
A checkpoint without ``bc.json`` (PPO, or PPO fine-tuned from BC) prints nothing::

    uv run python scripts/rl/evaluate.py ... --checkpoint <ckpt> $(uv run python scripts/imitation/bc_overrides.py <ckpt>)

With ``--ppo`` it refuses all but ``increment``: PPO would clip and penalize their
actions as increments.
"""

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("checkpoint")
parser.add_argument("--ppo", action="store_true", help="Fail unless the checkpoint can be fine-tuned with PPO.")
args = parser.parse_args()

path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "bc.json")
command = "increment"
if os.path.exists(path):
    with open(path) as f:
        command = json.load(f)["command"]
if command != "increment":
    if args.ppo:
        sys.exit(f"{args.checkpoint} outputs '{command}' arm commands; only 'increment' policies can be fine-tuned with PPO.")
    print(f"env.actions.arm_action.command={command} agent.clip_actions=null")
