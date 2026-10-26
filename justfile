set script-interpreter := ["bash"]

robot := env("RL_ROBOT", "anymal")
task := env("RL_TASK", "open_ball_valve")
id := "Isaac-HiveBoard-" + uppercamelcase(robot) + "-" + uppercamelcase(task) + "-RL"

default:
    @just --list

setup:
    git submodule update --init dependencies/isaaclab-hiveboard
    git -C dependencies/isaaclab-hiveboard submodule update --init \
        dependencies/IsaacLab dependencies/curobo dependencies/HiveBoard \
        dependencies/duatic_dynaarm dependencies/urdf-usd-converter
    uv sync
    uv sync --directory dependencies/isaaclab-hiveboard/dependencies/urdf-usd-converter

core := "dependencies/isaaclab-hiveboard"

assets *args:
    uv run python {{core}}/scripts/generate_anymal_newton_usd.py {{args}}
    uv run python {{core}}/scripts/generate_newton_usd.py --assets all {{args}}
    cp {{core}}/dependencies/HiveBoard/Simulation/Honeycomb/*.usd \
        {{core}}/source/isaaclab_hiveboard/isaaclab_hiveboard/assets/hiveboard/honeycomb/

bank *args:
    uv run python scripts/bank.py --robot {{robot}} --task {{task}} {{args}}

train *args:
    uv run python scripts/train.py --task {{id}}-v0 {{args}}

eval checkpoint *args:
    uv run python scripts/eval.py --task {{id}}-Play-v0 --checkpoint {{checkpoint}} {{args}}

video checkpoint output="videos" *args:
    uv run python scripts/eval.py --task {{id}}-Play-v0 --checkpoint {{checkpoint}} \
        --publication {{output}} {{args}}

bc dataset *args:
    uv run python scripts/bc.py --dataset {{dataset}} {{args}}

test:
    bash tests/smoke.sh

[script]
all threshold="0.8" *args:
    set -eu
    tasks=$(uv run python -c "from isaaclab_hiveboard.tasks import TASKS; print(*TASKS)")
    for robot in $(ls configs/robots | sed 's/.yaml//'); do
        for task in $tasks; do
            just --set robot "$robot" --set task "$task" bank || continue
            bank="logs/expert_bank/${robot}_${task}.pt"
            gate="b = torch.load('$bank', weights_only=False)['meta']['expert_success']"
            gate="$gate; print('Expert success:', b); raise SystemExit(b < {{threshold}})"
            if ! uv run python -c "import torch; $gate"; then
                continue
            fi
            just --set robot "$robot" --set task "$task" train {{args}}
            runs="logs/rsl_rl/${robot}_${task}/*/model_*.pt"
            checkpoint=$(ls -t $runs | head -1)
            just --set robot "$robot" --set task "$task" eval "$checkpoint"
        done
    done
