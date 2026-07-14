#!/bin/bash
# 完整环境搭建脚本：agentic_rl conda 环境
# CUDA 12.8 + H100 + PyTorch 2.6.0
# 用法: bash scripts/setup_env.sh

set -euo pipefail

ENV_NAME="agentic_rl"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.8"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=================================================="
echo "  Building conda env: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}  CUDA: ${CUDA_VERSION}"
echo "  Project root: ${ROOT}"
echo "=================================================="

# ── 1. Create conda env ───────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[SKIP] env '${ENV_NAME}' already exists. To rebuild: conda env remove -n ${ENV_NAME}"
else
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# Activate in this script via full path
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "Active env: $(which python)  $(python --version)"

# ── 2. PyTorch 2.6.0 with CUDA 12.8 ──────────────────────────────────────
echo ""
# cu128 wheel index only has >=2.7.0; cu124 wheel is ABI-compatible with CUDA 12.8
echo "[Step 2] Installing PyTorch 2.6.0 (cu124 wheel, runs fine on CUDA 12.8)..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# ── 3. vLLM 0.8.5.post1 ──────────────────────────────────────────────────
echo ""
echo "[Step 3] Installing vLLM 0.8.5.post1..."
pip install vllm==0.8.5.post1

# ── 4. flash-attn (pre-built wheel for torch2.6 + cu128) ─────────────────
echo ""
echo "[Step 4] Installing flash-attn 2.7.4.post1..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

# ── 5. Core ML packages (pinned versions from ReMA-public) ────────────────
echo ""
echo "[Step 5] Installing core ML packages..."
pip install \
    transformers==4.51.3 \
    peft==0.15.1 \
    accelerate==1.6.0 \
    datasets==3.5.0 \
    tokenizers>=0.21

# ── 6. veRL dependencies ─────────────────────────────────────────────────
echo ""
echo "[Step 6] Installing veRL dependencies..."
pip install \
    ray[default]>=2.10 \
    hydra-core==1.3.2 \
    omegaconf \
    "tensordict<0.6" \
    codetiming \
    numpy \
    pandas \
    pyarrow>=15.0.0 \
    pybind11 \
    pylatexenc \
    torchdata \
    wandb \
    liger-kernel

# ── 7. Math / RL utilities ────────────────────────────────────────────────
echo ""
echo "[Step 7] Installing math and RL utilities..."
pip install \
    math-verify==0.7.0 \
    pebble==5.1.1 \
    jsonlines \
    bitsandbytes \
    "scipy>=1.10" \
    "sympy>=1.12"

# ── 8. Install veRL from ReMA-public (editable) ───────────────────────────
echo ""
echo "[Step 8] Installing veRL from ReMA-public (editable)..."
pip install -e "/home/yangjingbo/ReMA-public/src/verl" --no-deps

# ── 9. Install agentic_rl project itself ─────────────────────────────────
echo ""
echo "[Step 9] Adding agentic_rl to PYTHONPATH via .pth file..."
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
echo "${ROOT}" > "${SITE_PKGS}/agentic_rl.pth"
echo "  Added ${ROOT} → ${SITE_PKGS}/agentic_rl.pth"

# ── 10. Verify key imports ────────────────────────────────────────────────
echo ""
echo "[Step 10] Verifying critical imports..."
python - <<'PYEOF'
import sys

checks = [
    ("torch",             lambda: __import__("torch").__version__),
    ("vllm",              lambda: __import__("vllm").__version__),
    ("flash_attn",        lambda: __import__("flash_attn").__version__),
    ("transformers",      lambda: __import__("transformers").__version__),
    ("peft",              lambda: __import__("peft").__version__),
    ("accelerate",        lambda: __import__("accelerate").__version__),
    ("ray",               lambda: __import__("ray").__version__),
    ("hydra",             lambda: __import__("hydra").__version__),
    ("tensordict",        lambda: __import__("tensordict").__version__),
    ("verl",              lambda: __import__("verl").__version__),
    ("envs.blackboard",   lambda: "ok"),
    ("agents.agentic_executor", lambda: "ok"),
]

failed = []
for name, fn in checks:
    try:
        ver = fn()
        print(f"  ✓ {name:<30} {ver}")
    except Exception as e:
        print(f"  ✗ {name:<30} FAILED: {e}", file=sys.stderr)
        failed.append(name)

if failed:
    print(f"\nFailed: {failed}", file=sys.stderr)
    sys.exit(1)
else:
    print("\nAll imports OK.")
PYEOF

echo ""
echo "=================================================="
echo "  Environment '${ENV_NAME}' ready."
echo ""
echo "  Activate with:  conda activate ${ENV_NAME}"
echo "  Run SFT:        bash scripts/train_sft_4role.sh"
echo "  Run RL:         bash scripts/train_verl_blackboard.sh"
echo "=================================================="
