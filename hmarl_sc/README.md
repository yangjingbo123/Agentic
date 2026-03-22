# HMARL-SC: 分层多智能体强化学习框架

面向大语言模型测试时推理的分层多智能体强化学习框架。

## 项目结构

```
hmarl_sc/
├── configs/          # 配置文件
│   └── default.yaml  # 默认超参数配置
├── envs/            # 环境实现
│   └── blackboard.py # 共享信息板
├── llm/             # LLM接口
│   ├── llm_interface.py    # LLM调用接口
│   └── prompt_templates.py # Prompt模板
├── models/          # 策略网络
│   ├── high_policy.py  # 高层Controller
│   ├── low_policy.py   # 低层三agent
│   └── value_nets.py   # 价值网络
├── training/        # 训练相关
│   └── rule_policies.py # 规则策略
└── utils/           # 工具函数
    └── data_loader.py  # 数据加载器
```

## 已实现组件

### 核心组件
- ✅ Blackboard信息板 (6种消息类型)
- ✅ LLM接口 (支持vLLM + 缓存)
- ✅ Prompt模板 (7种推理任务)
- ✅ 高层策略网络 (Controller)
- ✅ 低层策略网络 (Proposer/Critic/Verifier)
- ✅ 价值网络 (高层+低层)
- ✅ 规则策略 (SimpleRule)
- ✅ 数据加载器 (GSM8K)
- ✅ 环境类 (高层+低层+主环境)
- ✅ 基础测试脚本

- ✅ 训练流程 (Stage 0 BC + PPO训练器)
- ✅ 评估指标 (accuracy, cost, k_equiv等)
- ✅ 实验配置 (E1基线)
- ✅ 主训练脚本

### 待实现
- ⏳ Agent执行逻辑 (LLM调用集成)
- ⏳ Stage 1/2 训练流程
- ⏳ E2-E5 实验配置
- ⏳ 数据收集脚本

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置说明
参见 `configs/default.yaml`

## 技术栈
- Ray RLlib 2.9+ (多智能体RL)
- PyTorch 2.1+ (深度学习)
- vLLM 0.3+ (LLM推理)
- Qwen3-7B-Instruct (本地部署)
