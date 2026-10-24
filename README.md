# HiveBoard RL

Bank-guided asymmetric PPO. Deployable actor; privileged critic. No teacher distillation.

```sh
just setup
just assets
just bank
just train
just eval path/to/model.pt
just video path/to/model.pt videos
just bc path/to/demos.pt
just all
```

Select a pair with `RL_ROBOT=anymal RL_TOOL=ball_valve`. Available: ANYmal ball valve,
small valve, M30 thread, circuit breaker; Franka ball valve. Snake-case names.

`just all` builds each bank, skips expert success below 0.8, trains and evaluates.
`just test` runs the disposable small pipeline for every pair, including BC and video.

Replay: `just eval unused --policy expert`.
Collect BC demos: `uv run python scripts/eval.py --policy expert --save demos.pt`.
BC checkpoints carry their action-command overrides. Action-history noise defaults to 0.5.

`source/isaaclab_hiveboard_rl/isaaclab_hiveboard_rl/`: registration, specs, environment.
`scripts/`: bank, train, eval, BC. `tests/`: disposable checks. `attic/`: inactive old code.
Core environments and generated assets live in `dependencies/isaaclab-hiveboard/`.

## TODO

- Move asset physics, randomization and expert corrections to core.
- Split heuristics into asset skills and robot bindings.
- Add task-progress rewards for training without a bank.
- Review and remove attic.
- Retire experimental arm commands, nominal-bank flags, velocity reward and contact gating.
