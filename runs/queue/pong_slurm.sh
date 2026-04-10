#!/usr/bin/env bash
#SBATCH --job-name=rezero-pong
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=ehazan
#SBATCH --qos=gpu-long
#SBATCH --output=runs/queue/pong_slurm_%j.out
#SBATCH --error=runs/queue/pong_slurm_%j.err
#
# Sequential: Pong baseline then STU mixer (hankel_scaled), seed 0.
# Submit:  sbatch runs/queue/pong_slurm.sh
# Monitor: tail -F runs/queue/pong_slurm_<jobid>.out

set -uo pipefail
cd /home/sk3686/hazan/ReZero
source .venv/bin/activate

ts() { date +'%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] job $SLURM_JOB_ID start; node=$SLURMD_NODENAME gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

export WANDB_PROJECT=rezero-stu

run_one() {
    local name="$1"; shift
    local logdir="$(pwd)/runs/$name"
    if [[ -f "$logdir/latest.pt" ]]; then
        echo "[$(ts)] SKIP  $name (latest.pt exists)"
        return
    fi
    [[ -d "$logdir" ]] && rm -rf "$logdir"
    mkdir -p "$logdir"
    echo "[$(ts)] START $name"
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

run_one "pong_baseline_seed0"
run_one "pong_stu_hankelscaled_seed0" \
    model.stu_mixer.enabled=true \
    model.stu_mixer.filter_type=hankel_scaled

echo "[$(ts)] job complete"
