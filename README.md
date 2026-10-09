# HiveBoard RL

Reinforcement and imitation learning on the [isaaclab-hiveboard](https://github.com/EESC-LabRoM/isaaclab-hiveboard)
environments (ANYmal ball valve, small valve, M30 thread, circuit breaker).
**The current working solution is the PPO student trained with a cuRobo expert trajectory bank.**
Teacher training, student distillation, robomimic behaviour cloning and DAgger remain available as alternative
workflows, but are not the current working solution.

The core environments are a git submodule (`dependencies/isaaclab-hiveboard`), installed editable. This repo only
adds the `isaaclab_hiveboard_rl` package, which registers the `*-RL-v0` / `*-RL-Play-v0` tasks on top of the core
ones, and the training scripts.

## Installation

```bash
git clone https://github.com/EESC-LabRoM/hiveboard-rl.git && cd hiveboard-rl
just setup           # core submodule + the nested ones it needs, then uv sync --extra imitation
just sync-assets     # copy generated USD assets from ../isaaclab-hiveboard (or pass core=<path>)
just list-envs
```

USD assets are gitignored in the core repo, so a fresh submodule has none: either copy them from a core checkout
that built them (`just sync-assets`) or build them inside the submodule with its `generate-newton-usd` /
`generate-anymal-usd` recipes.

`pyproject.toml` mirrors the core repo's `[tool.uv]` indexes, sources and overrides (uv only reads them from the
top-level project). Keep the two in sync when bumping the submodule.

## Usage

The working workflow is **trajectory bank → PPO student → evaluation / playback**. No PPO teacher or teacher
checkpoint is used. The student learns through PPO rewards: the bank supplies episode resets, expert-tracking
rewards and deviation terminations, rather than supervised action targets. The actor uses deployable
observations; the critic uses privileged simulator state, including expert-reference errors.

```bash
just rl-bank                          # build the cuRobo expert bank -> logs/expert_bank/
just rl-student-ppo                   # working solution: bank-guided PPO, no teacher checkpoint
just rl-eval <model_*.pt>             # success rate / stage metrics
just rl-play <model_*.pt>             # Newton viewer + TorchScript/ONNX export
just rl-video-publication <model_*.pt> # 1080p RTX publication video -> <run>/videos/play/

# Select the same task for bank generation, training and evaluation:
RL_TOOL=SmallValve just rl-bank       # BallValve (default), SmallValve, M30Thread, CircuitBreaker
RL_TOOL=SmallValve just rl-student-ppo
```

BallValve used to limit each integrated arm target to 0.1 rad from the measured joint position at command updates,
to limit stored drive error under contact before release. The bound is off for now: at 0.1 rad only 59% of
expert-bank trajectories still open the valve through the increment action, against 98% without it, which caps
the behaviour-cloning baselines below. It will return with a larger value once the expert bank moves smoothly
enough to stay inside it (TODO). Some error is necessary to generate turning torque, so a new bound should be
evaluated for both opening success and release motion. Use `env.actions.arm_action.max_position_error=0.1` to
reproduce the bounded controller (`scripts/rl/compare_release.py` does). PPO-student checkpoints trained before
this change were trained with the bound; pass that override when evaluating or resuming them.

Expert joint-velocity tracking is available as an opt-in reward:
`env.rewards.track_expert_velocity.weight=1.0` (per-joint RMS error scale 0.5 rad/s).
It follows the same idle-free reference timeline as position tracking and asks for zero velocity after the
recorded retreat. A matched pilot comparison resumes one checkpoint twice, with the same seed and target
limit, then attaches 100-episode evaluation metrics and a standard-condition trajectory to each W&B run:

```bash
uv run python scripts/rl/compare_release.py --checkpoint <model_999.pt> --iterations 50 --episodes 100
uv run python scripts/rl/trace_policy.py --checkpoint <model_*.pt> --output logs/diagnostics/release
```

Comparison runs use the W&B group `ballvalve_release_cap_vs_velocity` and names containing the target limit,
velocity-reward variant, source iteration, pilot length and seed. Results are written under
`logs/release_comparison/`. These short, single-seed pilots do not establish convergence. Evaluation reports
release speed, acceleration and expert-velocity error separately from the earlier metrics that stop at opening.

Training retains every checkpoint locally but uploads only the best and latest saved weights to W&B when
training ends. "Best" means the highest rolling mean training episode return among saved checkpoints;
it does not mean the best independently evaluated success rate. The selected filenames, iterations and score
are recorded in the W&B `Checkpoints/*` summary fields. Interrupted runs upload the selected weights already
saved locally. Original `model_<iteration>.pt` filenames preserve the W&B checkpoint-loading workflow.

Publication recording uses the core recorder's publication settings: RTX quality 100, studio lighting,
50 FPS output, and H.264 CRF 12 with the slow preset. Frames are captured at the RL policy's step rate
and repeated for 50 FPS output, preserving simulation timing. Pass `--video_length` to change the
default 240 policy steps (12 seconds), or `--agent rsl_rl_distillation_cfg_entry_point` for a distilled student.
Recording stops at the first termination or episode timeout, omitting the autoreset frame.

For a batch of examples, edit `configs/recordings.yaml` to select tasks, checkpoints, and recording settings:

```bash
just record-all                        # record each configured checkpoint's first episode
just record-all --randomized           # restore randomized starts, dynamics, and registration error
just record-all --list                 # list configured examples
just record-all --dry-run              # inspect commands without launching simulations
just record-all --match ball-valve     # record a subset
just record-all --config path/to.yaml  # use another configuration
```

Like the core repository's `record-all`, this runs examples in separate processes, retries simulation crashes,
checks the resulting MP4s, and saves per-attempt logs plus `summary.json` in a dated folder under
`videos/examples/`. `--output` changes the parent folder. Checkpoint paths in YAML are relative to the repository
root; each example can override `defaults` and pass additional playback options in an `args` list.
`video_length` caps policy steps if an episode has not terminated, while `timeout` caps wall time per attempt.
Recordings use standard conditions by default: authored placement, a closed mechanism, the home arm posture,
fixed nominal dynamics, zero actuator delay, and no observation or registration noise. Set `standard: false`
for an individual YAML example, or pass `--randomized` for the whole batch. Single playback also accepts `--standard`.
The expert bank remains loaded for reward references; those references may differ from the nominal recording start.
The supplied configuration includes BallValve, M30Thread, and CircuitBreaker PPO students. Add other tasks
when their checkpoints are available.

The following alternative workflows remain in the repository. Teacher checkpoints are required only for
student distillation; these commands are not steps in the working PPO-student workflow above.

```bash
just rl-teacher                       # PPO teacher on privileged state
just rl-student <teacher-model.pt>    # distil teacher actions onto deployable observations
just il-collect                       # scripted-expert demos (robomimic layout)
just il-train <dataset.hdf5>          # behaviour cloning
just il-dagger <dataset.hdf5>         # DAgger rounds
just il-eval --checkpoint <ckpt>
```

### Behaviour-cloning baselines

Behaviour cloning from the same expert bank, compared against the PPO student. Each baseline trains the PPO
student's actor network on its `policy` observations, and is evaluated with the same metrics:

| Baseline | Arm command (`env.actions.arm_action.command`) | Purpose |
|---|---|---|
| BC-A | `increment`: the PPO student's action space | Controlled comparison: only the learning signal differs |
| BC-B | `absolute`: the scripted expert's joint-position control | Debugging: the expert's own interface, without the integrator or its speed limit |
| BC-C | `absolute_integrated`: absolute targets through the PPO student's integrator | Tells whether BC-A fails because of imitation or because of the incremental action parameterization |
| BC-B / BC-C, relative | `relative` / `relative_integrated`: the same targets, given relative to the measured joints | As above, without the absolute targets' copy-the-joints shortcut (see below) |
| BC-A + PPO | `increment`, fine-tuned by the PPO student's training | Demonstrations as initialization |
| PPO student | `increment` | The current working solution (reference) |

```bash
just bc-collect increment              # record the bank in the RL task -> logs/imitation/bank_demos/
just bc-collect absolute               # one dataset per command (BC-B)
just bc-collect absolute_integrated    # (BC-C)
just bc-collect relative               # BC-B / BC-C with relative targets
just bc-train logs/imitation/bank_demos/anymal_ball_valve_increment.pt   # -> logs/rsl_rl/<student>_bc/<run>/model_0.pt
just bc-eval <bc model_0.pt>           # rl-eval with the checkpoint's arm command
just bc-ppo <BC-A model_0.pt>          # BC-A + PPO (100 critic-only warm-up iterations, then lr 1e-5)
```

`bc-collect` plays the bank expert in the RL training task, in the chosen arm command, one trajectory per episode,
with the expert terminations off. It stores successful episodes only, unless you pass `--keep_failed`. The
replay success it prints is the ceiling for BC on that dataset. `bc-train` fits the actor offline and writes an
ordinary RSL-RL checkpoint. Its `bc.json` records the overrides that a non-increment command needs, and `bc-eval`
applies them. Use `--action_history_noise 0.5` (or `--ignore_action_history`): without it, every plain fit opens
0% (the copycat problem). `bc-ppo` adds a 100-iteration critic-only warm-up, an actor learning rate of 1e-5 and a frozen
actor normalizer (`agents/critic_warmup.py`). The contact gate on the valve reward is opt-in:
`env.rewards.track_expert_valve.params.gate_on_contact=true env.rewards.track_expert_valve_coarse.params.gate_on_contact=true`.

BallValve results: BC-A 96%, PPO student 88% (999 iterations), BC-A + PPO 41%. Fine-tuning from BC does not work
yet. See [research/bc_baselines.md](research/bc_baselines.md) for all results, findings and runs.

## Updating the core

```bash
git -C dependencies/isaaclab-hiveboard pull origin master
just test && just list-envs
git add dependencies/isaaclab-hiveboard && git commit -m "Bump isaaclab-hiveboard"
```

The RL tasks import core modules directly (`mdp.commands.sequential_pose_command`, `assets.anymal.bench`,
`tasks.anymal.*` scenes, `mdp.events`, `mdp.actions`, `mdp.recorders`, `utils.command_setup`), so run the tests
after every bump.

## Repository structure

```
source/isaaclab_hiveboard_rl/isaaclab_hiveboard_rl/
  __init__.py          # registers the RL tasks (and, via import, the core ones)
  tasks/anymal/        # ball_valve_rl, small_valve_rl, m30_thread_rl, circuit_breaker_rl
  imitation/           # scripted expert, datasets, robomimic policy, rollouts
scripts/rl/            # train, evaluate, play, expert-bank tools
scripts/imitation/     # collect_bank_demos, train_bank_bc (BC baselines); collect_demos, train_bc, dagger, eval_policy (robomimic)
logs/                  # expert banks, rsl_rl runs, imitation datasets (gitignored)
```
