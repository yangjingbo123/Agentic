#!/bin/bash
#SBATCH --job-name=agentic_rl
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:3
#SBATCH --constraint=a100_80g|h100 
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out  # 保存日志文件
#SBATCH --error=slurm-%j.err    # 保存错误日志

set -e

WORKDIR=/cephfs/volumes/hpc_home/k24104674/aed22256-9e0b-4f4f-86c1-c56793988876/jingbo/marl/Agentic/agentic_rl
cd $WORKDIR

mkdir -p logs

source /scratch/users/k24104674/Anaconda3/etc/profile.d/conda.sh
conda activate agentic_rl

# cu13 nvcc 只在 L40S (sm_89) 上需要，A100/H100 用系统 CUDA 即可
if [[ "$SLURM_JOB_CONSTRAINTS" == *"l40s"* ]]; then
  export PATH=/scratch/users/k24104674/Anaconda3/envs/agentic_rl/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH
fi

# Disable FlashInfer JIT sampling — nvcc (cu13/scratch) and flashinfer headers
# (cephfs env) are from different CUDA toolkit versions and are incompatible.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
rm -rf /cephfs/volumes/hpc_home/k24104674/aed22256-9e0b-4f4f-86c1-c56793988876/.cache/flashinfer/

EXP_NAME=${EXP_NAME:-math-grpo}

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Experiment: $EXP_NAME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python train.py \
    exp_name="$EXP_NAME" \
    data=math \
    sft_checkpoint=checkpoints/sft \
    hydra.run.dir=. \
    "$@"
