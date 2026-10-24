set script-interpreter := ["bash"]

robot := env("RL_ROBOT", "anymal")
tool := env("RL_TOOL", "ball_valve")
task := "Isaac-HiveBoard-" + uppercamelcase(robot) + "-" + uppercamelcase(tool) + "-RL"

default:
    @just --list

setup:
    git submodule update --init dependencies/isaaclab-hiveboard
    git -C dependencies/isaaclab-hiveboard submodule update --init dependencies/IsaacLab dependencies/curobo dependencies/HiveBoard dependencies/duatic_dynaarm dependencies/urdf-usd-converter
    uv sync
    uv sync --directory dependencies/isaaclab-hiveboard/dependencies/urdf-usd-converter

assets *args:
    uv run python dependencies/isaaclab-hiveboard/scripts/generate_anymal_newton_usd.py {{args}}
    uv run python dependencies/isaaclab-hiveboard/scripts/generate_newton_usd.py --assets all {{args}}
    cp dependencies/isaaclab-hiveboard/dependencies/HiveBoard/Simulation/Honeycomb/*.usd dependencies/isaaclab-hiveboard/source/isaaclab_hiveboard/isaaclab_hiveboard/assets/hiveboard/honeycomb/

bank *args:
    uv run python scripts/bank.py --task {{task}}-v0 {{args}}

train *args:
    uv run python scripts/train.py --task {{task}}-v0 {{args}}

eval checkpoint *args:
    uv run python scripts/eval.py --task {{task}}-Play-v0 --checkpoint {{checkpoint}} {{args}}

video checkpoint output="videos" *args:
    uv run python scripts/eval.py --task {{task}}-Play-v0 --checkpoint {{checkpoint}} --publication {{output}} {{args}}

bc dataset *args:
    uv run python scripts/bc.py --dataset {{dataset}} {{args}}

test:
    bash tests/smoke.sh

[script]
all threshold="0.8" *args:
    set -eu
    for pair in anymal:ball_valve anymal:small_valve anymal:m30_thread anymal:circuit_breaker franka:ball_valve; do
        IFS=: read -r robot tool <<< "$pair"
        just --set robot "$robot" --set tool "$tool" bank
        bank="logs/expert_bank/${robot}_${tool}.pt"
        if ! uv run python -c "import torch; b=torch.load('$bank', weights_only=False); print('Expert success:', b['meta']['expert_success']); raise SystemExit(b['meta']['expert_success'] < {{threshold}})"; then
            continue
        fi
        just --set robot "$robot" --set tool "$tool" train {{args}}
        checkpoint=$(uv run python -c "import glob,os; print(max(glob.glob('logs/rsl_rl/${robot}_${tool}_student_ppo/*/model_*.pt'), key=os.path.getmtime))")
        just --set robot "$robot" --set tool "$tool" eval "$checkpoint"
    done
