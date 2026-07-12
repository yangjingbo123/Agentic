#!/bin/bash
#SBATCH --job-name=test_grad
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /scratch/users/k24104674/Anaconda3/etc/profile.d/conda.sh
conda activate agentic_rl
export PATH=/scratch/users/k24104674/Anaconda3/envs/agentic_rl/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /cephfs/volumes/hpc_home/k24104674/aed22256-9e0b-4f4f-86c1-c56793988876/jingbo/marl/Agentic/agentic_rl
python diagnose_loss.py
