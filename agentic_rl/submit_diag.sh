#!/bin/bash
#SBATCH --job-name=agentic_diag
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:3
#SBATCH --constraint=l40s
#SBATCH --mem=64G
#SBATCH --output=diag-%j.out

cd /cephfs/volumes/hpc_home/k24104674/aed22256-9e0b-4f4f-86c1-c56793988876/jingbo/marl/Agentic/agentic_rl

source /scratch/users/k24104674/Anaconda3/etc/profile.d/conda.sh
conda activate agentic_rl

export PATH=/scratch/users/k24104674/Anaconda3/envs/agentic_rl/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python diag_fix.py
