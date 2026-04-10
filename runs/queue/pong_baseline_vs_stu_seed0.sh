#!/usr/bin/env bash
#
# Queue: Pong Atari 100K — vanilla R2-Dreamer baseline vs STU mixer
# (hankel_scaled), seed 0.
#
# Sequential. Single A100. Each run is the full env.steps=4.1e5 (Atari 100K
# spec: 100K agent decisions × 4 action_repeat).
#
# Logs:
#   runs/pong_*/console.log              (in-process stdout)
#   runs/pong_*/events.out.tfevents.*    (TensorBoard scalars)
#   runs/queue/pong_baseline_vs_stu_seed0.queue.log
#                                        (queue-level orchestration log)
#
# W&B mirroring is on (project=rezero-stu). Auth is via ~/.netrc — set up
# beforehand with `wandb login`. The queue script does NOT contain or echo
# any API key.
#
# To launch (foreground for testing, or via nohup for real run):
#   nohup bash runs/queue/pong_baseline_vs_stu_seed0.sh \
#     > runs/queue/pong_baseline_vs_stu_seed0.queue.log 2>&1 &
#
# Monitor:
#   tail -F runs/queue/pong_baseline_vs_stu_seed0.queue.log
#   tail -F runs/pong_baseline_seed0/console.log
#   tensorboard --logdir runs/

set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root
source .venv/bin/activate

ROOT=$(pwd)
ts() { date +'%Y-%m-%d %H:%M:%S'; }
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 || echo "unknown")
echo "[$(ts)] queue start; root=$ROOT  GPU=$GPU_NAME"

export WANDB_PROJECT=rezero-stu

run_one() {
    local name="$1"; shift
    local logdir="$ROOT/runs/$name"
    if [[ -d "$logdir" ]] && [[ -f "$logdir/latest.pt" ]]; then
        echo "[$(ts)] SKIP  $name (already finished: $logdir/latest.pt exists)"
        return
    fi
    if [[ -d "$logdir" ]]; then
        echo "[$(ts)] WARN  $name logdir exists but no latest.pt — wiping"
        rm -rf "$logdir"
    fi
    mkdir -p "$logdir"
    echo "[$(ts)] START $name -> $logdir"
    local rc=0
    WANDB_RUN_NAME="$name" python train.py \
        env=atari100k \
        env.task=atari_pong \
        env.eval_episode_num=10 \
        model.compile=false \
        seed=0 \
        logdir="$logdir" \
        "$@" || rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[$(ts)] FAIL  $name (exit $rc)"
        return $rc
    fi
    echo "[$(ts)] DONE  $name"
}

# 1) Vanilla R2-Dreamer baseline. STU mixer disabled (default).
run_one "pong_baseline_seed0"

# 2) STU mixer with hankel_scaled filters (the offline STUZero global best).
run_one "pong_stu_hankelscaled_seed0" \
    model.stu_mixer.enabled=true \
    model.stu_mixer.filter_type=hankel_scaled

echo "[$(ts)] queue complete"
