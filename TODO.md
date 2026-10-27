# TODO

## Core and RL split

- [x] robots, assets and tasks as core configclasses in `isaaclab_hiveboard/tasks_simple.py`
- [x] core `HiveBoardEnvCfg`, `build()` and `expert()`; RL `EnvCfg` adds the MDP only
- [x] RL YAML holds learning knobs only, train plus one file per robot
- [x] core per-robot task tree, scenes, JSON setups, editors and probes deleted
- [x] resolved configs of the 8 old pairs equal the pre-split baseline
- [x] high torque valve task, -180 deg, wheel_turn method, runs end to end with data only
- [ ] high torque expert 9.4 percent (6/64, ANYmal): grip slips on the rim during the 180 deg
      orbit, the wheel coasts in after release, retreat drags it past target; resistance
      ranges made it worse (0/32); needs a regrasp or a rim-following turn, not data
- [ ] restyle surviving core code (`mdp/`, `assets/`, generators) to AGENTS.md
- [ ] prune core `pyproject.toml` dependencies only used by deleted code
- [ ] screw mimic registry is process global and keyed by joint short name; composing two
      screw assets in one process can apply a stale coupling
- [ ] screw mimic hook flag survives simulation close, cannot reinstall in a second sim
- [ ] eval expert replay with `--publication` ignores the standard reset
- [ ] BC action overrides reach eval only, not PPO resume
- [ ] Spot has two articulations upstream (`SPOT_ARM_NEWTON_CFG`, `SPOT_ARM_BENCH_CFG`), keep one

## Scaling, pre-learning

- [ ] M8 and peg tasks: data and screw_regrasp method set, need TaskCfg goals and a bank gate
- [ ] lamp, button, shock, drawer and key are data only; `build()` cannot run them yet
- [ ] Spot builds but has no `configs/robots/spot.yaml`, so it is not registered or simulated
- [ ] lamp task: driven screw with finger contact gating, IdealPD joints, frames on two bodies
- [ ] button task: lid turn then closed-finger press, two joint goals
- [ ] shock task: grasp-push then press on PrismaticJoint, start 0.020, success at 0.004
- [ ] drawer task: free box rigid object, travel goal, own timing (dt 0.001, decimation 75)
- [ ] key task: key carried in hand, contact regex must include `/Key/`
- [ ] mechanism goal as a list of stages, joint or body travel, today one joint
- [ ] ball valve friction ring has a URDF only, no Newton USD
- [ ] expert method dispatch is an if chain in `expert()`; one function per method once there are 6
- [ ] lever_turn has the only robot branch, ANYmal `upright_axis`; make it robot data
- [ ] gate every robot x task on expert success before training

## Robot and asset pair facts

- [ ] Franka small valve and M30 had no successful experts, breaker had 6 percent
- [ ] candidates to test, from old upstream Franka envs: finger close 0.030 small valve,
      0.019 M30, breaker push 15 mm deeper, 20 s episodes, small valve panel rotation,
      floor board at 0.60 m
- [ ] derive pair facts from geometry: asset grasp width and contact surface, robot finger
      width map, tip offset and reach standoff
- [ ] Spot pinch frames need a -90 deg roll about TCP x for M8, peg, shock, drawer and button lid
- [ ] Spot min durations, home posture and diagnostic joint are inferred, check in sim
- [ ] episode length scales with robot speed, not per pair numbers

## Protocol gaps

- [ ] small and high torque valves need one full turn, ours are 90 and 180 deg
- [ ] threads need full length (M30 about 7 turns), ours is 2 turns
- [ ] breaker switches either way, ours only one direction
- [ ] key is welded to the hand, no pickup
- [ ] peg starts engaged two turns, no free pick and align
- [ ] shock pin starts through the spring eye, no free pick and align
- [ ] drawer protocol is open then close, old expert removed the box

## Geometry and physics

- [ ] isolate reported quaternion-sign sensitivity with controlled equivalent inputs
- [ ] justify M30 regrasp budget and task-specific completion tolerances
- [ ] persist sampled contact materials in banks and restore them during replay
- [ ] measure expert-bank yield under the shared contact-material distributions

## Learning

- [ ] full training and held-out evaluation per robot and task
- [ ] tabula rasa progress reward remains separate research scope
- [ ] remove robustness gain, delay and registration knobs when study ends
- [ ] select and retain one arm-command method after performance evidence
- [ ] review and delete attic after user review
