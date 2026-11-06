# Franka FR3 switch circuit breaker (simple)

<img src="method_success.webp" width="640">

`model_999.pt` switches the toggle and withdraws with closed fingers (2x speed, nominal physics).

## Task

The task lives in core `tasks_simple.py`. Learning shares `env.py`, `configs/train.yaml` and
`configs/robots/franka.yaml` with the
[ball-valve policy](../open_ball_valve_simple/README.md).

| | |
|---|---|
| Robot | Franka Research 3 with Franka Hand |
| Goal | push the toggle from +30 to -30 degrees, then withdraw |
| Goal tolerance | 0.1 rad, approximately 5.7 degrees |
| Interaction | either finger pad can push; no pinch or finger opening is required |
| Episode | 9 seconds at a 20 Hz policy rate |
| Starts | states from a task-specific cuRobo expert bank |
| Randomization | mechanism position and dynamics, arm starts, target delay and observation noise |
| Gym ID | `Isaac-HiveBoard-Franka-SwitchCircuitBreakerSimple-RL-v0` |

## Method

A separate PPO policy is trained from scratch with `[512, 256, 128]` actor and critic MLPs.
The actor sees proprioception, an initial mechanism estimate and history; the critic also sees
true state, contacts, dynamics and expert errors.

The expert approaches below the toggle, pushes upward with closed fingers and retreats. The
training bank has 1,000 trajectories (96.8% raw expert success); the held-out bank has 200
(98.6%, seed 10020). Collection used 512 environments and planning batches of 64, filtering out
fallbacks, incomplete withdrawals and switches taking over five seconds.

Half of training resets start mid-trajectory, restoring state, dynamics and prior contact history.
Push-specific rewards replace the valve's pinch and reach incentives. Overrides live under
`tasks.<task_name>` in the shared or robot YAML; the valve configuration remains unchanged.

## Rewards

| Group | Term | Weight | What it does |
|---|---|---|---|
| Task | `task_progress` | 10 | credits new progress after valid pad contact, plus completion |
| Style | `track_expert_mechanism` | 6 | tracks the expert angle, with a contact gate |
| Style | `track_expert_mechanism_coarse` | 5 | same with a wider tolerance |
| Style | `track_expert` | 3 | arm joints follow the expert |
| Style | `track_expert_gripper` | 3 | fingers follow the expert's measured position |
| Style | `track_expert_contact` | 3 | each pad's contact state matches the expert |
| Style | `track_expert_coarse` | 2 | arm joints follow the expert, wider tolerance |
| Style | `track_expert_velocity` | 2 | arm joint velocity follows the expert |
| Hardware | `torque_saturation` | -2 | joint torque near its limit |
| Hardware | `action_rate` | -1 | changes in commanded actions |
| Hardware | `joint_limits` | -1 | joints near their limits |
| Hardware | `gripper_chatter` | -1 | gripper commands toggling quickly |
| Hardware | `arm_joint_acc` | -0.03 | high arm joint acceleration |
| Hardware | `action_magnitude` | -0.02 | large actions |
| Hardware | `arm_joint_vel` | -0.001 | fast arm joints |
| Hardware | `pad_force` | -0.001 | squared pad force above 150 N |

Progress pays only for new gains after either pad exceeds 1 N, including motion after contact ends.
Completion adds a one-time 0.5 bonus before weighting. The critic sees progress and contact history.
Mechanism tracking uses bank angles with 0.1/0.5 rad tolerances and requires current policy contact
whenever the expert touches. Push speed is measured in m/s.

`pinch_lever`, `reach_lever`, `pad_lateral` and `overspeed` are off. Total force replaces lateral
force because pushing requires sideways force. Mechanism-lag termination is disabled to allow timing
differences in the toggle transition. Arm drift, invalid-state and collision terminations remain.

## Evaluation and results

Run `2026-11-05_16-27-55`: seed 0, 4,096 environments, 1,000 PPO iterations. Evaluation used
300 episodes, 100 environments and seed 100, visiting 143 distinct held-out bank entries.
Solver settings matched training (50/10/30). Evaluation starts at the beginning of trajectories,
with bank dynamics, registration noise and target delay. Expert drift terminations are disabled.

**Success** requires reaching the goal after pad contact. **Complete** requires the goal retained
at the end, pad forces below 1 N and TCP clearance of at least 6 cm from `lever_pivot_below`.

| Checkpoint | Complete | Mean time to switch | Maximum pad force |
|---|---|---|---|
| 100 | 272/300 (90.7%) | 3.35 s | 707 N |
| 200 | 300/300 (100%) | 3.27 s | 867 N |
| 300 | 300/300 (100%) | 3.26 s | 707 N |
| 400 | 300/300 (100%) | 3.22 s | 809 N |
| 500 | 299/300 (99.7%) | 3.25 s | 815 N |
| 600 | 300/300 (100%) | 3.26 s | 568 N |
| 700 | 300/300 (100%) | 3.23 s | 367 N |
| 800 | 300/300 (100%) | 3.31 s | 534 N |
| 900 | 300/300 (100%) | 3.30 s | 432 N |
| 999 | **300/300 (100%)** | **3.25 s** | **520 N** |

All checkpoints had zero collisions and invalid episodes. Checkpoint 999 had no gripper command
switches and lateral force p99 of 398 N. The selector preferred checkpoint 700 for its lower peak
force at equal completion. Results cover one training seed; this held-out bank serves as validation.
Reports: `tests/breaker_evaluation/2026-11-05_16-27-55/`.

## What needs to be improved

- **Contact forces:** 150 N is a soft penalty threshold; force peaks remain high.
- **Withdrawal:** checkpoint 100 switched 300/300 times but completed 272. Select by complete rate.
- **Validation:** repeat seeds and collect new evaluation starts after checkpoint selection.

## Reproduce

Run these commands from the repository root after `just setup` and `just assets`.

```bash
export RL_ROBOT=franka RL_TASK=switch_circuit_breaker_simple
bank=logs/expert_bank/franka_switch_circuit_breaker_simple.pt
heldout=logs/expert_bank/franka_switch_circuit_breaker_simple_heldout.pt

just bank --num_trajectories 1000 --plan_batch_size 64 --seed 0

just bank --num_trajectories 200 --plan_batch_size 64 --seed 10020 \
    --output "$heldout"

just train --num_envs 4096 --seed 0 --max_iterations 1000 \
    env.sim.physics.use_cuda_graph=true env.sim.physics.solver_cfg.iterations=50 \
    env.sim.physics.solver_cfg.ls_iterations=10 env.sim.physics.solver_cfg.ccd_iterations=30
```

Keep collection batch sizes fixed for comparisons. Replace the timestamp below for a new run.

```bash
run=logs/rsl_rl/franka_switch_circuit_breaker_simple/2026-11-05_16-27-55
checkpoint="$run/model_999.pt"

just eval "$checkpoint" --episodes 300 --num_envs 100 --seed 100 \
    env.events.reset_from_bank.params.path="$heldout" \
    env.sim.physics.use_cuda_graph=true env.sim.physics.solver_cfg.iterations=50 \
    env.sim.physics.solver_cfg.ls_iterations=10 env.sim.physics.solver_cfg.ccd_iterations=30
```

Use `model_700.pt` for the sweep's selected checkpoint; training return uses a different ranking.

## Visualize and record

Default camera: eye `[0.90, -0.55, 0.50]`, lookat `[0.60, 0.00, 0.10]`, relative to environment 0.

```bash
just eval "$checkpoint" --episodes 100 --num_envs 1 --visualizer newton_gl --seed 100 \
    env.events.reset_from_bank.params.path="$heldout" \
    env.sim.physics.use_cuda_graph=true
```

Publication recording uses 1080p RTX rendering and CRF 12 encoding: one standard-start episode
with nominal physics and no observation noise or delays, separate from held-out evaluation.

```bash
video=videos/franka_switch_circuit_breaker_simple_publication
just video "$checkpoint" "$video" --seed 100 env.sim.physics.use_cuda_graph=true
```
