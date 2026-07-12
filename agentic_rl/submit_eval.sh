#!/bin/bash
#SBATCH --job-name=agentic_eval
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80g|h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err

set -e

WORKDIR=/cephfs/volumes/hpc_home/k24104674/aed22256-9e0b-4f4f-86c1-c56793988876/jingbo/marl/Agentic/agentic_rl
cd $WORKDIR

source /scratch/users/k24104674/Anaconda3/etc/profile.d/conda.sh
conda activate agentic_rl

export VLLM_USE_FLASHINFER_SAMPLER=0

SFT_CKPT=/scratch/users/k24104674/jingbo_checkpoints/sft
RL_CKPT=/scratch/users/k24104674/jingbo_checkpoints/rl-math-grpo_3

echo "======== SFT Model ========"
python evaluate.py --checkpoint $SFT_CKPT --split test --max_samples ${MAX_SAMPLES:-200}

echo ""
echo "======== RL Model (step 128) ========"
python evaluate.py --checkpoint $RL_CKPT --split test --max_samples ${MAX_SAMPLES:-200}
