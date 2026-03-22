# HMARL-SC 实验完整指南

## 第一步: 环境准备

### 1.1 安装依赖
```bash
# 在有GPU的服务器上执行
pip install -r requirements.txt
```

### 1.2 下载Qwen3-7B-Instruct模型

**方法1: 使用Hugging Face (推荐)**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 下载模型到本地
model_name = "Qwen/Qwen3-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./models",  # 保存到本地
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
```

**方法2: 使用ModelScope (国内更快)**
```bash
pip install modelscope
```

```python
from modelscope import snapshot_download

model_dir = snapshot_download('qwen/Qwen3-7B-Instruct', cache_dir='./models')
print(f'模型已下载到: {model_dir}')
```

**方法3: 手动下载**
- 访问: https://huggingface.co/Qwen/Qwen3-7B-Instruct
- 下载所有文件到 `./models/Qwen3-7B-Instruct/`

### 1.3 配置模型路径
编辑 `configs/default.yaml`:
```yaml
llm:
  model: "./models/Qwen3-7B-Instruct"  # 改为本地路径
  backend: "vllm"
  quantization: "awq"
```

---

## 第二步: 数据准备

### 2.1 下载GSM8K数据集
```bash
# 自动下载(需要联网)
python -c "from datasets import load_dataset; load_dataset('gsm8k', 'main')"
```

### 2.2 创建必要目录
```bash
mkdir -p data checkpoints logs
```

---

## 第三步: 实验流程

### 3.1 收集BC训练数据
```bash
python scripts/collect_data.py
```
- 耗时: 约2-3小时 (3000个episode)
- 输出: `data/bc_trajectories.pkl`

### 3.2 Stage 0: BC初始化训练
```bash
python train.py --stage 0
```
- 耗时: 约30分钟 (30 epochs)
- 输出: `checkpoints/stage0_final.pt`

### 3.3 Stage 1: 交替冻结训练
```bash
python train.py --stage 1
```
- 耗时: 约2-3小时 (3轮交替)
- 输出: `checkpoints/stage1_final.pt`

### 3.4 Stage 2: 联合微调
```bash
python train.py --stage 2
```
- 耗时: 约1-2小时 (5000步)
- 输出: `checkpoints/stage2_final.pt`

### 3.5 评估模型
```bash
python scripts/evaluate.py
```
- 耗时: 约30分钟
- 输出: 准确率、平均成本、等价k值

---

## 第四步: 运行实验配置

### E1: 基线实验
```bash
python experiments/e1_baseline.py
```

---

## 常见问题

### Q1: GPU内存不足?
A: 修改 `configs/default.yaml`:
```yaml
llm:
  gpu_memory_utilization: 0.2  # 降低到0.2
  quantization: "awq"  # 确保使用量化
```

### Q2: 训练太慢?
A: 减少训练数据量:
```yaml
stage0:
  num_episodes: 1000  # 从3000降到1000
```

### Q3: 没有GPU?
A: 使用Google Colab或Kaggle Notebooks (免费GPU)

---

## 预期结果

根据实验计划,预期指标:
- **准确率**: 75-80% (GSM8K测试集)
- **等价k**: 8-10 (相当于SC-8到SC-10)
- **平均成本**: 3200-3600 tokens

---

## 完整实验流程 (一键脚本)

创建 `run_experiment.sh`:
```bash
#!/bin/bash
echo "开始HMARL-SC完整实验"

# 1. 收集数据
echo "步骤1: 收集BC数据..."
python scripts/collect_data.py

# 2. Stage 0
echo "步骤2: Stage 0训练..."
python train.py --stage 0

# 3. Stage 1
echo "步骤3: Stage 1训练..."
python train.py --stage 1

# 4. Stage 2
echo "步骤4: Stage 2训练..."
python train.py --stage 2

# 5. 评估
echo "步骤5: 评估模型..."
python scripts/evaluate.py

echo "实验完成!"
```

运行:
```bash
chmod +x run_experiment.sh
./run_experiment.sh
```
