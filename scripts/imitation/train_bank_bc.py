#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behaviour-clone the PPO student's actor from recorded bank demonstrations.

Trains exactly the network the PPO student uses (``--agent``, by default
``rsl_rl_student_ppo_cfg_entry_point``: same MLP, observation normalization and
``policy`` observations) on a dataset from ``collect_bank_demos.py``, and saves
it as an RSL-RL checkpoint, so ``scripts/rl/evaluate.py`` and ``play.py`` load
it like any PPO checkpoint and ``rl-student-ppo --checkpoint`` can fine-tune it
(BC-A + PPO). No simulation is launched.

* Arm: mean squared error on per-joint standardized targets; the
  standardization is folded into the last layer when saving.
* Gripper: a logistic loss on the expert's open/close command (its sign is what
  the binary gripper executes), rather than regressing +-1.
* Critic (``--critic_epochs``, off by default): fitted to the expert's discounted
  returns on the ``teacher`` observations. Not used for fine-tuning: its input
  normalization is the expert's, and the actor's own deviations under PPO fall far
  outside it (value loss ~400x a fresh critic's), so its advantages undo the cloned
  behaviour. ``bc-ppo`` trains a fresh critic on the actor's rollouts instead
  (``agent.algorithm.critic_warmup_updates``).
  A timeout bootstraps with the last reward held forever, ``r_T / (1 - gamma)``.
* The actor's exploration std is set to ``--init_std`` for fine-tuning.
* Action history: the ``policy`` observations include the last actions, and the
  expert's are smooth, so a plain fit mostly extrapolates its own past commands
  (copycat) and drifts in closed loop. ``--ignore_action_history`` keeps their
  first-layer weights at zero (the input size, and so PPO fine-tuning, is
  unchanged); ``--action_history_noise`` adds Gaussian noise to them in
  training, in units of their dataset std.

The run is written to ``logs/rsl_rl/<experiment>_bc/<date>_<command>[_<variant>]/model_0.pt``
next to ``bc.json``, which records the data, metrics and the Hydra overrides
evaluation needs (``scripts/imitation/bc_overrides.py``)::

    uv run python scripts/imitation/train_bank_bc.py --dataset logs/imitation/bank_demos/anymal_ball_valve_increment.pt
"""

import argparse
import datetime
import json
import math
import os
import types

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", required=True, help="Demonstrations from collect_bank_demos.py.")
parser.add_argument("--agent", default="rsl_rl_student_ppo_cfg_entry_point", help="Agent whose actor is trained.")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--weight_decay", type=float, default=1.0e-4)
parser.add_argument("--gripper_weight", type=float, default=1.0, help="Weight of the gripper loss vs the arm's.")
parser.add_argument("--val_fraction", type=float, default=0.05, help="Episodes held out for validation.")
parser.add_argument("--max_episodes", type=int, default=None, help="Train on at most this many episodes.")
parser.add_argument("--include_failed", action="store_true", help="Also train on unsuccessful episodes.")
parser.add_argument("--critic_epochs", type=int, default=0, help="Fit the critic offline (0: untrained).")
parser.add_argument("--init_std", type=float, default=0.1, help="Actor std stored for PPO fine-tuning.")
history = parser.add_mutually_exclusive_group()
history.add_argument("--ignore_action_history", action="store_true", help="The actor ignores its last-action inputs.")
history.add_argument(
    "--action_history_noise", type=float, default=0.0, help="Training noise std on the last-action inputs (normalized)."
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--output_dir", default=None, help="Default: logs/rsl_rl/<experiment>_bc/<date>_<command>.")
args = parser.parse_args()

#: Steps either side of an expert gripper switch counted as "near a switch" in the metrics.
SWITCH_WINDOW = 5


def build_algorithm(task: str, policy_dim: int, teacher_dim: int, action_dim: int, device: str):
    """The agent's RSL-RL algorithm, built as its runner would build it, without an environment."""
    import isaaclab_hiveboard_rl  # noqa: F401  (registers the tasks)
    from isaaclab_rl.rsl_rl import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg
    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    from isaaclab.utils import to_dict

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(task, args.agent), check_rsl_rl_version())
    cfg = to_dict(agent_cfg)
    from rsl_rl.algorithms import PPO

    alg_class = resolve_callable(cfg["algorithm"]["class_name"])
    if not issubclass(alg_class, PPO) or cfg["obs_groups"].get("actor") != ["policy"]:
        raise SystemExit(f"{args.agent} must be a PPO agent whose actor reads the 'policy' group.")
    cfg["multi_gpu"] = None
    obs = TensorDict({"policy": torch.zeros(1, policy_dim), "teacher": torch.zeros(1, teacher_dim)}, batch_size=[1])
    env = types.SimpleNamespace(num_envs=1, num_actions=action_dim)
    alg = alg_class.construct_algorithm(obs, env, cfg, device)
    return alg, cfg


def set_normalizer(normalizer: torch.nn.Module, x: torch.Tensor) -> None:
    """Set an ``EmpiricalNormalization`` to the statistics of the whole dataset ``x``."""
    var, mean = torch.var_mean(x, dim=0, unbiased=False, keepdim=True)
    normalizer._mean.copy_(mean)
    normalizer._var.copy_(var)
    normalizer._std.copy_(var.sqrt())
    normalizer.count.fill_(x.shape[0])


def action_history_columns(policy: torch.Tensor, actions: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor):
    """Columns of ``policy`` holding the last actions, every history slot (oldest first).

    Read from the data: the newest slot equals the previous step's action, and each
    older slot, ``A`` columns before it, the action one step further back.
    """
    A = actions.shape[1]
    episodes = [(s, n) for s, n in zip(starts.tolist(), lengths.tolist()) if n > 16][:100]
    t = torch.cat([torch.arange(s + 8, s + n) for s, n in episodes])

    def matches(first: int, back: int) -> bool:
        return first >= 0 and bool((policy[t, first : first + A] - actions[t - back]).abs().max() < 1.0e-6)

    first = int((policy[t] - actions[t - 1, :1]).abs().amax(0).argmin())
    if not matches(first, 1):
        raise SystemExit("No last-action observations found in the 'policy' group.")
    back = 1
    while back < 8 and matches(first - back * A, back + 1):
        back += 1
    first -= (back - 1) * A
    return list(range(first, first + back * A))


def batches(n: int, size: int, shuffle: bool, device: str):
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    yield from order.split(size)


def discounted_returns(rewards: torch.Tensor, lengths: torch.Tensor, timeout: torch.Tensor, gamma: float):
    """Per-step discounted return of every episode, bootstrapping timeouts with ``r_T / (1 - gamma)``."""
    returns = []
    for r, cut in zip(rewards.split(lengths.tolist()), timeout.tolist()):
        r = r.tolist()
        g = r[-1] / (1.0 - gamma) if cut else 0.0
        episode = [0.0] * len(r)
        for t in range(len(r) - 1, -1, -1):
            g = r[t] + gamma * g
            episode[t] = g
        returns.extend(episode)
    return torch.tensor(returns)


def main() -> None:
    torch.manual_seed(args.seed)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    meta = data["meta"]
    command = meta["command"]
    dev = args.device

    # Episodes: successful ones unless asked otherwise, split train / validation by episode.
    lengths = data["lengths"]
    starts = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]))
    keep = torch.ones_like(data["success"]) if args.include_failed else data["success"]
    episodes = keep.nonzero().flatten()
    episodes = episodes[torch.randperm(len(episodes))]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    n_val = max(1, int(round(args.val_fraction * len(episodes))))
    if len(episodes) <= n_val:
        raise SystemExit(f"{args.dataset} has {len(episodes)} usable episodes, too few to train and validate.")
    split = {"val": episodes[:n_val], "train": episodes[n_val:]}

    def steps_of(ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.arange(starts[i], starts[i] + lengths[i]) for i in ids.tolist()])

    def near_switch(ids: torch.Tensor) -> torch.Tensor:
        """Per step of ``ids``: within SWITCH_WINDOW steps of a change in the expert's gripper command."""
        flags = []
        for i in ids.tolist():
            g = data["actions"][starts[i] : starts[i] + lengths[i], -1] > 0
            switch = torch.zeros(len(g))
            switch[1:] = (g[1:] != g[:-1]).float()
            flags.append(F.max_pool1d(switch[None, None], 2 * SWITCH_WINDOW + 1, 1, SWITCH_WINDOW)[0, 0] > 0)
        return torch.cat(flags)

    idx = {name: steps_of(ids) for name, ids in split.items()}
    obs = {name: data["policy"][i].to(dev) for name, i in idx.items()}
    act = {name: data["actions"][i].to(dev) for name, i in idx.items()}
    val_switch = near_switch(split["val"]).to(dev)
    action_dim = act["train"].shape[-1]
    print(
        f"[BC] {meta['task']} ({command}): {len(split['train'])} train / {n_val} val episodes,"
        f" {len(idx['train'])} / {len(idx['val'])} steps, bank expert replay success {meta['replay_success_rate']:.1%}"
    )

    alg, cfg = build_algorithm(meta["task"], data["policy"].shape[-1], data["teacher"].shape[-1], action_dim, dev)
    actor, critic = alg._raw_actor, alg._raw_critic
    set_normalizer(actor.obs_normalizer, obs["train"])
    history_cols = []
    if args.ignore_action_history or args.action_history_noise > 0:
        history_cols = action_history_columns(data["policy"], data["actions"], starts, lengths)
        print(f"[BC] action history: policy columns {history_cols[0]}-{history_cols[-1]}")
    if args.ignore_action_history:
        # Zero weights see zeroed inputs in training, so they get no gradient and stay zero.
        with torch.no_grad():
            actor.mlp[0].weight[:, history_cols] = 0.0

    def train_inputs(x: torch.Tensor) -> torch.Tensor:
        x = actor.obs_normalizer(x)
        if args.ignore_action_history:
            x[:, history_cols] = 0.0
        elif args.action_history_noise > 0:
            x[:, history_cols] += args.action_history_noise * torch.randn_like(x[:, history_cols])
        return x

    # Arm targets standardized per joint; the gripper is a logit (> 0 open, as the binary action reads it).
    arm_mean = act["train"][:, :-1].mean(dim=0)
    arm_std = act["train"][:, :-1].std(dim=0).clamp(min=1.0e-3)

    def loss_terms(out: torch.Tensor, target: torch.Tensor):
        arm = F.mse_loss(out[:, :-1], (target[:, :-1] - arm_mean) / arm_std)
        grip = F.binary_cross_entropy_with_logits(out[:, -1], (target[:, -1] > 0).float())
        return arm, grip

    optimizer = torch.optim.AdamW(actor.mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * math.ceil(len(idx["train"]) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total, pct_start=0.05)

    def validate() -> dict[str, float]:
        actor.eval()
        out = torch.cat([actor.mlp(actor.obs_normalizer(obs["val"][b])) for b in batches(len(obs["val"]), 65536, False, dev)])
        actor.train()
        target = act["val"]
        arm, grip = loss_terms(out, target)
        arm_pred = out[:, :-1] * arm_std + arm_mean
        correct = (out[:, -1] > 0) == (target[:, -1] > 0)
        return {
            "arm_loss": float(arm),
            "gripper_loss": float(grip),
            # In the command's units: unit increments, or rad for the others.
            "arm_rmse": (arm_pred - target[:, :-1]).square().mean(dim=0).sqrt().tolist(),
            "gripper_accuracy": float(correct.float().mean()),
            "gripper_accuracy_near_switch": float(correct[val_switch].float().mean()),
        }

    history = []
    for epoch in range(args.epochs):
        sums = torch.zeros(2, device=dev)
        for b in batches(len(obs["train"]), args.batch_size, True, dev):
            out = actor.mlp(train_inputs(obs["train"][b]))
            arm, grip = loss_terms(out, act["train"][b])
            optimizer.zero_grad()
            (arm + args.gripper_weight * grip).backward()
            torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            sums += torch.stack((arm.detach(), grip.detach())) * len(b)
        train = (sums / len(obs["train"])).tolist()
        val = validate()
        history.append({"epoch": epoch, "train_arm_loss": train[0], "train_gripper_loss": train[1], **val})
        print(
            f"[BC] epoch {epoch:3d}  train arm {train[0]:.4f} grip {train[1]:.4f} | val arm {val['arm_loss']:.4f}"
            f" grip {val['gripper_loss']:.4f} acc {val['gripper_accuracy']:.3f}"
            f" (near switch {val['gripper_accuracy_near_switch']:.3f})",
            flush=True,
        )

    if args.ignore_action_history and actor.mlp[0].weight[:, history_cols].any():
        raise RuntimeError("The ignored action-history weights changed in training.")
    # Fold the arm standardization into the last layer: the saved actor outputs the command itself.
    with torch.no_grad():
        last = actor.mlp[-1]
        scale = torch.cat((arm_std, torch.ones(1, device=dev)))
        shift = torch.cat((arm_mean, torch.zeros(1, device=dev)))
        last.weight.mul_(scale[:, None])
        last.bias.mul_(scale).add_(shift)
        actor.distribution.std_param.fill_(args.init_std)

    critic_metrics = None
    if args.critic_epochs > 0:
        gamma = float(cfg["algorithm"]["gamma"])
        returns = discounted_returns(data["rewards"], lengths, data["timeout"], gamma)
        teacher = {name: data["teacher"][i].to(dev) for name, i in idx.items()}
        value = {name: returns[i].to(dev) for name, i in idx.items()}
        set_normalizer(critic.obs_normalizer, teacher["train"])
        v_mean, v_std = value["train"].mean(), value["train"].std().clamp(min=1.0e-3)
        critic_opt = torch.optim.AdamW(critic.mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for _ in range(args.critic_epochs):
            for b in batches(len(teacher["train"]), args.batch_size, True, dev):
                pred = critic.mlp(critic.obs_normalizer(teacher["train"][b]))[:, 0]
                loss = F.mse_loss(pred, (value["train"][b] - v_mean) / v_std)
                critic_opt.zero_grad()
                loss.backward()
                critic_opt.step()
        with torch.no_grad():
            critic.mlp[-1].weight.mul_(v_std)
            critic.mlp[-1].bias.mul_(v_std).add_(v_mean)
            pred = critic.mlp(critic.obs_normalizer(teacher["val"]))[:, 0]
            err = (pred - value["val"]).square().mean().sqrt()
        critic_metrics = {"gamma": gamma, "val_rmse": float(err), "return_std": float(v_std)}
        print(f"[BC] critic: validation return RMSE {float(err):.3f} (returns std {float(v_std):.3f})")

    variant = "_noactionhistory" if args.ignore_action_history else ""
    if args.action_history_noise > 0:
        variant = f"_actionnoise{args.action_history_noise:g}"
    output_dir = args.output_dir or os.path.join(
        "logs",
        "rsl_rl",
        f"{cfg['experiment_name']}_bc",
        f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{command}{variant}",
    )
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = os.path.join(output_dir, "model_0.pt")
    info = {
        "dataset": os.path.abspath(args.dataset),
        "dataset_meta": {key: value for key, value in meta.items() if key != "hydra_overrides"},
        "command": command,
        "task": meta["task"],
        "agent": args.agent,
        "args": vars(args),
        "episodes": {"train": len(split["train"]), "val": n_val},
        "final": history[-1],
        "critic": critic_metrics,
        "action_history_columns": history_cols,
        "history": history,
    }
    torch.save({**alg.save(), "iter": 0, "infos": {"bc": info}}, checkpoint)
    with open(os.path.join(output_dir, "bc.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[BC] saved {checkpoint}")


if __name__ == "__main__":
    main()
