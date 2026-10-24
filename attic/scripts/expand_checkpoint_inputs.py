#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Append observation inputs to an RSL-RL PPO checkpoint so it can be warm-started.

When observation terms are appended to the end of a group, the actor and critic
input layers grow. This pads each first ``Linear`` layer with zero columns (the
policy's outputs are unchanged until PPO learns to use the new inputs), extends
the observation normalizers with the given prior mean and variance, and pads
the matching Adam moments::

    uv run python scripts/rl/expand_checkpoint_inputs.py \\
        --checkpoint logs/rsl_rl/anymal_ball_valve_teacher/teacher_v4_model_2299.pt \\
        --output logs/rsl_rl/anymal_ball_valve_teacher/teacher_v4_expanded.pt \\
        --mean 0.5 0.5 0.0 --var 0.03 0.1 0.05

Then train with ``--checkpoint <output>``.
"""

import argparse

import torch


def pad_columns(tensor: torch.Tensor, extra: int, value: float = 0.0) -> torch.Tensor:
    """Append ``extra`` columns filled with ``value`` to the last dimension."""
    pad = torch.full((*tensor.shape[:-1], extra), value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mean", type=float, nargs="+", required=True, help="Normalizer prior mean per new input.")
    parser.add_argument("--var", type=float, nargs="+", required=True, help="Normalizer prior variance per new input.")
    args = parser.parse_args()
    if len(args.mean) != len(args.var):
        raise ValueError("--mean and --var need one value per appended input")
    extra = len(args.mean)
    mean, var = torch.tensor(args.mean), torch.tensor(args.var)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old_shapes = {}
    for key in ("actor_state_dict", "critic_state_dict"):
        state = ckpt[key]
        old_shapes[key] = tuple(state["mlp.0.weight"].shape)
        state["mlp.0.weight"] = pad_columns(state["mlp.0.weight"], extra)
        state["obs_normalizer._mean"] = torch.cat((state["obs_normalizer._mean"], mean[None]), dim=-1)
        state["obs_normalizer._var"] = torch.cat((state["obs_normalizer._var"], var[None]), dim=-1)
        state["obs_normalizer._std"] = torch.cat((state["obs_normalizer._std"], var.sqrt()[None]), dim=-1)
        print(f"{key}: input {old_shapes[key][1]} -> {state['mlp.0.weight'].shape[1]}")

    # Adam moments of the two input layers have the old shape; pad them to match.
    padded = 0
    for param_state in ckpt.get("optimizer_state_dict", {}).get("state", {}).values():
        for name in ("exp_avg", "exp_avg_sq"):
            moment = param_state.get(name)
            if moment is not None and tuple(moment.shape) in old_shapes.values():
                param_state[name] = pad_columns(moment, extra)
                padded += 1
    print(f"padded {padded} optimizer moments")
    torch.save(ckpt, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
