import argparse
import datetime
import importlib
import math
import os
import types

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--weight_decay", type=float, default=1.0e-4)
parser.add_argument("--gripper_weight", type=float, default=1.0)
parser.add_argument("--val_fraction", type=float, default=0.05, help="episodes held out")
parser.add_argument("--max_episodes", type=int, default=None)
parser.add_argument("--include_failed", action="store_true")
parser.add_argument("--critic_epochs", type=int, default=0, help="0 leaves the critic untrained")
parser.add_argument("--init_std", type=float, default=0.1, help="actor std for PPO fine-tuning")
history = parser.add_mutually_exclusive_group()
history.add_argument("--ignore_action_history", action="store_true")
history.add_argument(
    "--action_history_noise", type=float, default=0.5, help="std on normalized last actions"
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--output_dir", default=None)
args = parser.parse_args()

SWITCH_WINDOW = 5


def build_algorithm(task, policy_dim, teacher_dim, action_dim, device):
    importlib.import_module("isaaclab_hiveboard_rl")
    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    cfg = to_dict(handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version()))
    cfg["multi_gpu"] = None
    obs = TensorDict(
        {"policy": torch.zeros(1, policy_dim), "teacher": torch.zeros(1, teacher_dim)},
        batch_size=[1],
    )
    env = types.SimpleNamespace(num_envs=1, num_actions=action_dim)
    alg_class = resolve_callable(cfg["algorithm"]["class_name"])
    return alg_class.construct_algorithm(obs, env, cfg, device), cfg


def set_normalizer(normalizer, x):
    var, mean = torch.var_mean(x, dim=0, unbiased=False, keepdim=True)
    normalizer._mean.copy_(mean)
    normalizer._var.copy_(var)
    normalizer._std.copy_(var.sqrt())
    normalizer.count.fill_(x.shape[0])


def action_history_columns(policy, actions, starts, lengths):
    A = actions.shape[1]
    episodes = [(s, n) for s, n in zip(starts.tolist(), lengths.tolist()) if n > 16][:100]
    t = torch.cat([torch.arange(s + 8, s + n) for s, n in episodes])

    def matches(first, back):
        if first < 0:
            return False
        diff = policy[t, first : first + A] - actions[t - back]
        return bool(diff.abs().max() < 1.0e-6)

    first = int((policy[t] - actions[t - 1, :1]).abs().amax(0).argmin())
    if not matches(first, 1):
        raise SystemExit("No last-action observations found in the 'policy' group.")
    back = 1
    while back < 8 and matches(first - back * A, back + 1):
        back += 1
    first -= (back - 1) * A
    return list(range(first, first + back * A))


def batches(n, size, shuffle, device):
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    yield from order.split(size)


def discounted_returns(rewards, lengths, timeout, gamma):
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


def main():
    torch.manual_seed(args.seed)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    meta = data["meta"]
    command = meta["command"]
    dev = args.device

    lengths = data["lengths"]
    starts = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]))
    keep = torch.ones_like(data["success"]) if args.include_failed else data["success"]
    episodes = keep.nonzero().flatten()
    episodes = episodes[torch.randperm(len(episodes))]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    n_val = max(1, int(round(args.val_fraction * len(episodes))))
    if len(episodes) <= n_val:
        raise SystemExit(f"{args.dataset} has only {len(episodes)} usable episodes.")
    split = {"val": episodes[:n_val], "train": episodes[n_val:]}

    def steps_of(ids):
        return torch.cat([torch.arange(starts[i], starts[i] + lengths[i]) for i in ids.tolist()])

    def near_switch(ids):
        flags = []
        for i in ids.tolist():
            g = data["actions"][starts[i] : starts[i] + lengths[i], -1] > 0
            switch = torch.zeros(len(g))
            switch[1:] = (g[1:] != g[:-1]).float()
            window = 2 * SWITCH_WINDOW + 1
            near = F.max_pool1d(switch[None, None], window, 1, SWITCH_WINDOW)[0, 0]
            flags.append(near > 0)
        return torch.cat(flags)

    idx = {name: steps_of(ids) for name, ids in split.items()}
    obs = {name: data["policy"][i].to(dev) for name, i in idx.items()}
    act = {name: data["actions"][i].to(dev) for name, i in idx.items()}
    val_switch = near_switch(split["val"]).to(dev)
    action_dim = act["train"].shape[-1]
    print(
        f"[BC] {meta['task']} ({command}): {len(split['train'])} train / {n_val} val"
        f" episodes, {len(idx['train'])} / {len(idx['val'])} steps,"
        f" bank expert replay success {meta['replay_success_rate']:.1%}"
    )
    dims = (data["policy"].shape[-1], data["teacher"].shape[-1], action_dim)
    alg, cfg = build_algorithm(meta["task"], *dims, dev)
    actor, critic = alg._raw_actor, alg._raw_critic
    set_normalizer(actor.obs_normalizer, obs["train"])
    history_cols = []
    if args.ignore_action_history or args.action_history_noise > 0:
        history_cols = action_history_columns(data["policy"], data["actions"], starts, lengths)
        print(f"[BC] action history: policy columns {history_cols[0]}-{history_cols[-1]}")
    if args.ignore_action_history:
        with torch.no_grad():
            actor.mlp[0].weight[:, history_cols] = 0.0

    def train_inputs(x):
        x = actor.obs_normalizer(x)
        if args.ignore_action_history:
            x[:, history_cols] = 0.0
        elif args.action_history_noise > 0:
            x[:, history_cols] += args.action_history_noise * torch.randn_like(x[:, history_cols])
        return x

    arm_mean = act["train"][:, :-1].mean(dim=0)
    arm_std = act["train"][:, :-1].std(dim=0).clamp(min=1.0e-3)

    def loss_terms(out, target):
        arm = F.mse_loss(out[:, :-1], (target[:, :-1] - arm_mean) / arm_std)
        grip = F.binary_cross_entropy_with_logits(out[:, -1], (target[:, -1] > 0).float())
        return arm, grip

    optimizer = torch.optim.AdamW(
        actor.mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total = args.epochs * math.ceil(len(idx["train"]) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total, pct_start=0.05
    )

    def validate():
        actor.eval()
        out = torch.cat(
            [
                actor.mlp(actor.obs_normalizer(obs["val"][b]))
                for b in batches(len(obs["val"]), 65536, False, dev)
            ]
        )
        actor.train()
        target = act["val"]
        arm, grip = loss_terms(out, target)
        arm_pred = out[:, :-1] * arm_std + arm_mean
        correct = (out[:, -1] > 0) == (target[:, -1] > 0)
        return {
            "arm_loss": float(arm),
            "gripper_loss": float(grip),
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
        history.append(
            {"epoch": epoch, "train_arm_loss": train[0], "train_gripper_loss": train[1], **val}
        )
        print(
            f"[BC] epoch {epoch:3d}  train arm {train[0]:.4f} grip {train[1]:.4f}"
            f" | val arm {val['arm_loss']:.4f}"
            f" grip {val['gripper_loss']:.4f} acc {val['gripper_accuracy']:.3f}"
            f" (near switch {val['gripper_accuracy_near_switch']:.3f})",
            flush=True,
        )

    if args.ignore_action_history and actor.mlp[0].weight[:, history_cols].any():
        raise RuntimeError("The ignored action-history weights changed in training.")
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
        critic_opt = torch.optim.AdamW(
            critic.mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
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
        print(f"[BC] critic: validation return RMSE {float(err):.3f} (std {float(v_std):.3f})")

    variant = ""
    overrides = []
    if args.ignore_action_history:
        variant = "_noactionhistory"
    elif args.action_history_noise > 0:
        variant = f"_actionnoise{args.action_history_noise:g}"
    if command != "increment":
        overrides = [f"env.actions.arm_action.command={command}", "agent.clip_actions=null"]
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
        "args": vars(args),
        "episodes": {"train": len(split["train"]), "val": n_val},
        "final": history[-1],
        "critic": critic_metrics,
        "action_history_columns": history_cols,
        "history": history,
    }
    torch.save(
        {**alg.save(), "iter": 0, "infos": {"bc": info}, "bc_overrides": overrides},
        checkpoint,
    )
    print(f"[BC] saved {checkpoint}")


if __name__ == "__main__":
    main()
