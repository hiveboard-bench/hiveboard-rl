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

Select a pair with `RL_ROBOT=anymal RL_TASK=open_ball_valve`. Every robot is registered with
every task. Registration does not imply expert success. `just all` banks each pair and only
trains pairs whose expert success reaches 0.8. `just test` runs the full disposable pipeline.

## Authoring

Isaac Lab knowledge is expected. YAMLs describe assets, robots and task objectives;
implemented expert procedures are not a second configuration framework.

| File | Owns |
|---|---|
| `configs/assets/<asset>.yaml` | USD bindings, mounting, contact geometry and physical properties |
| `configs/robots/<robot>.yaml` | articulation, TCP, planner, joints, gripper and control timing |
| `configs/tasks/<task>.yaml` | asset, goal, starting distribution and episode limit |
| `configs/train.yaml` | shared PPO, BC, observations, rewards and evaluation settings |

Assets use one common scene builder. USDs, collision overlays and their payload dependencies
remain in `dependencies/isaaclab-hiveboard/`. Adding an asset does not require a scene subclass.
Keep physical geometry in the asset, robot execution in the robot, and desired outcomes in
the task. Task starts set the manipulated joint directly; screw axial references remain
physical asset data.

`mounting` locates the mechanism in the board setup. `panel` is the honeycomb pose relative
to the mechanism root. `frames` locates canonical TCP targets relative to its named USD body.
Expert execution and RL grasp measurements share those targets and the robot's native TCP.
Quaternions use xyzw. Panel reset follows the mechanism with the same full mounting relation.
The canonical ball grasp adopts the expert calibration (0.456 mm from the former RL target).
Breaker grasp measurements now use the push contact rather than the joint-center frame.

Expert procedures live in `scripts/bank.py` and call the existing `SkillSet` primitives.
cuRobo plans the requested robot motions. It does not discover the contact strategy or
success criterion. Demonstration diversity and execution settings belong to bank generation,
not asset YAML. Screw motion consumes the same pitch as reset and simulation coupling.
Repeated screw regrasping has a finite budget. Goals must remain within the implemented
method's supported motion and contact strategy.

Expert and training use the same authored contact-material ranges and solver settings.
Play disables material randomization and uses spawned material defaults. Banks do not yet
persist sampled contact materials for exact replay. Regenerate banks after changing physical
properties or contact geometry.

Omit shared defaults. Task pose sampling defaults to x/z ±0.05 m, y ±0.06 m and roll/pitch/yaw
±15 degrees, intersected with robot pose limits. YAML specifies only exceptions; joint offset
defaults to zero. This distribution generates the expert bank; training resets consume it.

YAML mapping order is for readability; sequence order is behavioral. Short asset section
headings separate description, mounting, contact geometry and physics.

Shared and robot-specific Isaac Lab overrides remain in `configs/train.yaml` and robot YAML
`env`, `play` and `expert` sections. Asset physical data and method-specific wiring are composed
in Python; tasks do not carry arbitrary Isaac manager overrides. Mode overrides apply after
common construction. CLI Hydra overrides apply last, for example
`env.events.reset_from_bank.params.path=bank.pt`. There is no separate `advanced` layer.

Replay with `just eval unused --policy expert`. Collect BC demos with
`uv run python scripts/eval.py --policy expert --save demos.pt`.

`source/isaaclab_hiveboard_rl/isaaclab_hiveboard_rl/` holds registration, MDP and composition.
`scripts/` holds bank, train, eval and BC. `tests/` is disposable. `attic/` is inactive old code.
Unresolved method and physics questions remain in `TODO.md`.
