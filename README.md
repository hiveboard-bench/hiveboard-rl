# HiveBoard RL

Bank-guided asymmetric PPO. Deployable actor, privileged critic.

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

Select a pair with `RL_ROBOT=anymal RL_TASK=open_ball_valve_simple`. Every robot with a
`configs/robots/<robot>.yaml` is registered with every core task. Registration does not imply
expert success. `just all` banks each pair and only trains pairs whose expert success reaches
0.8. `just test` runs the full disposable pipeline.

## Structure

The benchmark comes from `dependencies/isaaclab-hiveboard`. Its
`isaaclab_hiveboard/tasks_simple.py` holds robots, assets and single-goal tasks as Isaac Lab
configclasses, `HiveBoardEnvCfg`, `build()` for the scene, physics and actions, and `expert()`
for the scripted demonstrator. This repository adds learning only.

| File | Owns |
|---|---|
| `configs/train.yaml` | PPO, BC, observations, rewards, terminations, play and publication modes |
| `configs/robots/<robot>.yaml` | robot-specific learning knobs: delays, gripper noise and thresholds |
| `isaaclab_hiveboard_rl/env.py` | MDP, `EnvCfg(HiveBoardEnvCfg)`, `compose(robot, task, mode)` |
| `scripts/` | bank, train, eval and BC |

Layers merge in the order core build, train, robot, then the mode section (`play`, `expert`).
CLI Hydra overrides apply last, for example `env.events.reset_from_bank.params.path=bank.pt`.

## Adding an asset and a task

1. Generate its Newton USD with `just assets`.
2. Add an `AssetCfg` to `ASSETS`: USD, mounting, panel, passive joints, canonical TCP target
   frames, grasp frame, contact mode and the expert `method` it uses.
3. Add a `TaskCfg` to `TASKS`: asset, start, target, tolerance, start distribution exceptions
   and episode length.
4. `just bank` checks expert success. Train only above the gate.

A new kind of motion adds one method to `expert()`; assets with that motion reuse it.

Replay with `just eval unused --policy expert`. Collect BC demos with
`uv run python scripts/eval.py --policy expert --save demos.pt`. `tests/` is disposable.
`attic/` is inactive old code. Open method and physics questions are in `TODO.md`.
