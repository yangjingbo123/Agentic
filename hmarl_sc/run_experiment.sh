#!/bin/bash
# HMARL-SC 完整实验流程

echo "========================================="
echo "HMARL-SC 实验开始"
echo "========================================="

# 检查目录
mkdir -p data checkpoints logs

# 步骤1: 收集BC数据
echo ""
echo "[1/5] 收集BC训练数据..."
python scripts/collect_data.py

# 步骤2: Stage 0训练
echo ""
echo "[2/5] Stage 0: BC初始化..."
python train.py --stage 0

# 步骤3: Stage 1训练
echo ""
echo "[3/5] Stage 1: 交替冻结训练..."
python train.py --stage 1

# 步骤4: Stage 2训练
echo ""
echo "[4/5] Stage 2: 联合微调..."
python train.py --stage 2

# 步骤5: 评估
echo ""
echo "[5/5] 评估模型..."
python scripts/evaluate.py

echo ""
echo "========================================="
echo "实验完成!"
echo "========================================="
