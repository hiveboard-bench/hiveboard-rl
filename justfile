# Justfile for HiveBoard RL / imitation learning

default:
    @just --list

# Init the core submodule and the nested submodules it needs at runtime
setup:
    git submodule update --init dependencies/isaaclab-hiveboard
    git -C dependencies/isaaclab-hiveboard submodule update --init \
        dependencies/IsaacLab dependencies/curobo dependencies/HiveBoard dependencies/duatic_dynaarm
    uv sync --extra imitation

# Copy generated (gitignored) USD assets from a core checkout that already built them
sync-assets core="../isaaclab-hiveboard":
    cd {{core}} && git status --ignored --porcelain source/isaaclab_hiveboard/isaaclab_hiveboard/assets \
        | grep '^!!' | cut -c4- | grep -v __pycache__ \
        | rsync -ar --files-from=- --exclude=__pycache__ ./ {{justfile_directory()}}/dependencies/isaaclab-hiveboard/

# List all registered HiveBoard environments (core + RL)
list-envs:
    uv run python -c "import gymnasium as gym, isaaclab_hiveboard_rl; print('\n'.join(sorted(k for k in gym.registry if 'HiveBoard' in k)))"

# CPU unit tests for the valve RL tasks
test:
    uv run --with pytest python -m pytest scripts/test_valve_rl_tasks.py

# Collect scripted-expert demos for imitation learning (robomimic layout)
il-collect num_demos="50" num_envs="8" *args:
    uv run python scripts/imitation/collect_demos.py --num_demos {{num_demos}} --num_envs {{num_envs}} {{args}}

# Behaviour-clone an MLP policy from a collected dataset
il-train dataset *args:
    uv run python scripts/imitation/train_bc.py --dataset {{dataset}} {{args}}

# Improve a behaviour-cloned policy with DAgger rounds
il-dagger dataset rounds="5" *args:
    uv run python scripts/imitation/dagger.py --initial_dataset {{dataset}} --rounds {{rounds}} {{args}}

# Measure a checkpoint's success rate (pass --expert for the scripted ceiling)
il-eval *args:
    uv run python scripts/imitation/eval_policy.py {{args}}

# Part of the rl-* recipes: BallValve (default), SmallValve, M30Thread or CircuitBreaker, e.g. `RL_TOOL=SmallValve just rl-teacher`
rl_tool := env("RL_TOOL", "BallValve")

# Build the cuRobo expert bank the RL task resets from and tracks
rl-bank num_envs="512" num_trajectories="5000" *args:
    uv run python scripts/rl/build_expert_bank.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-v0 \
        --num_envs {{num_envs}} --num_trajectories {{num_trajectories}} {{args}}

# RL teacher: PPO on privileged state (ANYmal ball valve by default)
rl-teacher num_envs="4096" *args:
    uv run python scripts/rl/train.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-v0 --num_envs {{num_envs}} {{args}}

# RL student: distil the teacher checkpoint onto proprioception + registered valve pose
rl-student teacher num_envs="4096" *args:
    uv run python scripts/rl/train.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-v0 --num_envs {{num_envs}} \
        --agent rsl_rl_distillation_cfg_entry_point --checkpoint {{teacher}} \
        env.terminations.expert_drift=null env.terminations.expert_valve_lag=null {{args}}

# RL student trained directly with PPO: deployable actor, privileged critic
rl-student-ppo num_envs="4096" *args:
    uv run python scripts/rl/train.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-v0 --num_envs {{num_envs}} \
        --agent rsl_rl_student_ppo_cfg_entry_point {{args}}

# Replay expert-bank trajectories in simulation, shown in Viser (http://localhost:9080)
rl-bank-replay *args:
    uv run python scripts/rl/replay_expert_bank.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-Play-v0 {{args}}

# BC baselines: record the expert bank in the RL task, arm command increment (BC-A), absolute (BC-B) or absolute_integrated (BC-C)
bc-collect command="increment" *args:
    uv run python scripts/imitation/collect_bank_demos.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-v0 \
        --command {{command}} {{args}}

# Behaviour-clone the PPO student's actor from recorded bank demos (no simulation)
bc-train dataset *args:
    uv run python scripts/imitation/train_bank_bc.py --dataset {{dataset}} {{args}}

# rl-eval for a BC checkpoint, with the arm command it was trained on
bc-eval checkpoint *args:
    uv run python scripts/rl/evaluate.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-Play-v0 \
        --agent rsl_rl_student_ppo_cfg_entry_point --checkpoint {{checkpoint}} {{args}} \
        $(uv run python scripts/imitation/bc_overrides.py {{checkpoint}})

# BC-A + PPO: the PPO student's training, starting from an increment BC checkpoint; the first
# `warmup` iterations train only a fresh critic on the cloned actor's rollouts, then the actor starts at `lr`
bc-ppo checkpoint num_envs="4096" warmup="100" lr="1e-5" *args:
    uv run python scripts/imitation/bc_overrides.py --ppo {{checkpoint}}
    uv run python scripts/rl/train.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-v0 --num_envs {{num_envs}} \
        --agent rsl_rl_student_ppo_cfg_entry_point --checkpoint {{checkpoint}} --run_name bc_init \
        agent.algorithm.critic_warmup_updates={{warmup}} agent.algorithm.critic_warmup_learning_rate=5e-4 \
        agent.algorithm.learning_rate={{lr}} agent.algorithm.freeze_actor_normalization=true {{args}}

# Success rate / reliability / stage metrics of a teacher or student checkpoint
rl-eval checkpoint *args:
    uv run python scripts/rl/evaluate.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-Play-v0 --checkpoint {{checkpoint}} {{args}}

# Watch a checkpoint in the Newton viewer and export it (TorchScript + ONNX)
rl-play checkpoint *args:
    uv run python scripts/rl/play.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-Play-v0 --checkpoint {{checkpoint}} --viz newton {{args}}

# Record one 12 s episode of a checkpoint to <run>/videos/play/ (student: --agent rsl_rl_distillation_cfg_entry_point)
rl-video checkpoint *args:
    uv run python scripts/rl/play.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-Play-v0 --checkpoint {{checkpoint}} \
        --num_envs 1 --video --video_length 240 --viz newton {{args}}

# Record a checkpoint at 1080p with converged RTX, studio lighting and CRF 12 / slow encoding
rl-video-publication checkpoint *args:
    uv run python scripts/rl/play.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-Play-v0 --checkpoint {{checkpoint}} \
        --num_envs 1 --video_length 240 --publication --stop-on-termination {{args}}

# Record the first episode of every task/checkpoint in configs/recordings.yaml
record-all *args:
    uv run python scripts/record_all.py {{args}}
