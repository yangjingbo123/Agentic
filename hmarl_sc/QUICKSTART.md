# 快速开始

## 1. 下载模型 (选一种方法)

### 方法A: ModelScope (国内推荐)
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen3-7B-Instruct', cache_dir='./models')"
```

### 方法B: Hugging Face
```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-7B-Instruct', cache_dir='./models')"
```

## 2. 修改配置
编辑 `configs/default.yaml` 第6行:
```yaml
model: "./models/Qwen3-7B-Instruct"
```

## 3. 运行实验
```bash
bash run_experiment.sh
```

就这么简单！
