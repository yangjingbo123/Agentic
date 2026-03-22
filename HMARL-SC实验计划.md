# HMARL-SC框架实验计划

## Context
HMARL-SC是一个面向大语言模型测试时推理的分层多智能体强化学习框架。该框架通过两层时间尺度的多智能体协作（高层Controller + 低层三agent：Proposer/Critic/Verifier）来改进Self-Consistency方法，实现动态预算分配、证据对抗和自适应停止。

本实验计划旨在系统性验证框架的有效性，按照文档第11节的设计，分阶段验证各组件的贡献，并通过消融实验和分析揭示系统的工作机制。

---

## 技术栈选择

### 1. 强化学习框架
**主框架：Ray RLlib 2.9+**
- 原生支持多智能体强化学习（MARL）
- 支持分层策略结构（通过MultiAgentEnv）
- 成熟的PPO实现，支持分布式训练
- 灵活的自定义环境接口

**辅助库：**
- **PettingZoo 1.24+**：多智能体环境标准化接口
- **Gymnasium 0.29+**：单智能体环境基础（RLlib兼容）

**选择理由**：
- 需要同时处理高层单agent（Controller）和低层多agent（P/C/V）
- 需要SMDP支持（宏动作持续时间不固定）
- 需要集中式训练、分布式执行（CTDE）
- RLlib的MultiAgentEnv + 自定义Policy可以优雅实现两层结构

### 2. 大语言模型接口
**主LLM：OpenAI API (GPT-3.5-turbo / GPT-4)**
- 推理质量稳定
- API调用成本可控
- 支持temperature、top_p等采样参数

**备选LLM：**
- **Llama-2-70B**（通过vLLM本地部署）：降低成本
- **Qwen-72B**：中文数学推理能力强

**LLM调用库：**
- **LiteLLM 1.30+**：统一多个LLM API接口
- **tiktoken**：token计数（预算管理）

### 3. 深度学习框架
**PyTorch 2.1+**
- RLlib默认后端
- 灵活的自定义网络结构
- 支持混合精度训练（节省显存）

**网络组件：**
- **Transformers 4.36+**：用于文本embedding（问题编码e_x）
- 使用预训练的sentence-transformers模型（如all-MiniLM-L6-v2）

### 4. 实验管理
**Weights & Biases (wandb)**
- 实验跟踪和可视化
- 超参数扫描
- 模型版本管理

**MLflow**（备选）
- 开源替代方案
- 本地部署

### 5. 数据处理
**Datasets (Hugging Face)**
- 统一的数据集加载接口
- 支持GSM8K、MATH等数学推理数据集

**Pandas + NumPy**
- 结果分析和统计

### 6. 开发环境
**Python 3.10+**
- 类型提示支持（提高代码可维护性）
- 性能优化

**依赖管理：Poetry 或 conda**

---

## 实验环境准备

### 1. 硬件配置
**实际配置（单卡A6000）**：
- GPU：NVIDIA A6000 (48GB) × 1
- CPU：16核+
- RAM：64GB+
- 存储：500GB SSD（用于缓存LLM输出）

**显存分配策略**：
- LLM推理：~14GB（Qwen2.5-7B-Instruct，4bit量化）
- 策略网络训练：~8GB
- RLlib环境和缓存：~10GB
- 系统预留：~16GB

### 2. 数据集选择
- **GSM8K**（小学数学）：8,473训练 + 1,319测试
- **MATH**（竞赛数学）：7,500训练 + 5,000测试
- **SVAMP**（数学应用题）：1,000题
- **StrategyQA**（多步推理）：2,290训练 + 490测试

**训练集划分**：
- 训练集：70%用于策略学习
- 验证集：15%用于超参数调优和early stopping
- 测试集：15%用于最终评估

### 3. LLM配置

**主LLM：Qwen3-7B-Instruct（本地部署）**

**部署方案**：
```python
# 使用vLLM进行高效推理
from vllm import LLM, SamplingParams

LLM_CONFIG = {
    "model": "Qwen/Qwen3-7B-Instruct",  # 使用Qwen3
    "tensor_parallel_size": 1,  # 单卡
    "gpu_memory_utilization": 0.85,
    "quantization": "awq",  # 4bit量化，节省显存
    "max_model_len": 4096,
    "dtype": "half",  # fp16
}

# 初始化
llm = LLM(**LLM_CONFIG)

# 采样参数
SAMPLING_PARAMS = {
    "explore": SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512),
    "diagnose": SamplingParams(temperature=0.3, top_p=0.9, max_tokens=512),
}
```

**为什么选择Qwen3**：
- 数学推理能力更强（GSM8K: 78% vs Qwen2.5的72%）
- 推理速度相当
- 支持更长上下文（32K）
- 指令遵循能力更好

**备选方案（如果显存不足）**：
- **Qwen3-7B-Instruct + GPTQ量化**：~7GB显存
- **DeepSeek-Math-7B**：专门针对数学推理优化
- **调用API**：Qwen API（通义千问）或DeepSeek API（成本低）

**Token成本估算**（基于文档5.5节）：
- generate: ~400 tokens（7B模型输出较短）
- generate-diverse: ~400 tokens
- refine: ~350 tokens
- critique-logic: ~250 tokens
- find-counterexample: ~350 tokens
- quick-verify: ~150 tokens
- step-verify: ~350 tokens

**预算设置**：
- 单次CoT平均成本：400 tokens（7B模型）
- 总预算：B = K_equiv × 400
- K_equiv ∈ {5, 10, 20}（减少最大预算，适应7B能力）

### 4. 策略网络架构

**高层Controller网络**：
```python
class HighLevelPolicy(nn.Module):
    def __init__(self, obs_dim=271, hidden_dim=256):
        # obs_dim = 15 (统计特征) + 256 (问题embedding)
        self.shared_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Goal head: 5-way (EXPLORE-open, EXPLORE-minority, CHALLENGE, DIAGNOSE, STOP)
        self.goal_head = nn.Linear(hidden_dim, 5)

        # Focus head: Pointer attention over candidate answers
        self.focus_query = nn.Linear(hidden_dim, 64)
        self.focus_key = nn.Linear(256, 64)  # answer embedding

        # Budget head: 3-way (light, standard, heavy)
        self.budget_head = nn.Linear(hidden_dim, 3)
```

**低层Agent网络（共享主干）**：
```python
class LowLevelPolicy(nn.Module):
    def __init__(self, obs_dim=512, hidden_dim=256):
        # 共享主干
        self.shared_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 角色特定头（3个角色）
        self.role_heads = nn.ModuleDict({
            'proposer': RoleHead(hidden_dim, work_actions=5, comm_actions=5),
            'critic': RoleHead(hidden_dim, work_actions=4, comm_actions=5),
            'verifier': RoleHead(hidden_dim, work_actions=3, comm_actions=5),
        })

        # 信息板注意力编码器
        self.blackboard_encoder = TransformerEncoder(
            d_model=128, nhead=4, num_layers=2
        )

class RoleHead(nn.Module):
    def __init__(self, hidden_dim, work_actions, comm_actions):
        self.work_head = nn.Linear(hidden_dim, work_actions)
        self.comm_head = nn.Linear(hidden_dim, comm_actions)
        self.pointer_query = nn.Linear(hidden_dim, 64)
```

**价值网络**：
```python
class HighLevelValue(nn.Module):
    def __init__(self, obs_dim=271, hidden_dim=256):
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

class LowLevelValue(nn.Module):
    def __init__(self, joint_obs_dim=1536, hidden_dim=256):
        # 集中式Critic：观测所有3个agent的状态
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
```

### 5. 文本Embedding模型

**问题编码器**：
```python
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
# 输出维度：384维
# 用于编码问题文本 e_x
```

**推理链编码**（可选，用于轨迹相似度计算）：
- 同样使用all-MiniLM-L6-v2
- 或使用更轻量的TF-IDF + PCA降维

---

## 代码结构设计

```
hmarl_sc/
├── envs/
│   ├── reasoning_env.py          # 主环境（继承MultiAgentEnv）
│   ├── high_level_env.py         # 高层SMDP环境
│   ├── low_level_env.py          # 低层多agent环境
│   └── blackboard.py             # 共享信息板实现
├── agents/
│   ├── proposer.py               # Proposer agent逻辑
│   ├── critic.py                 # Critic agent逻辑
│   ├── verifier.py               # Verifier agent逻辑
│   └── controller.py             # 高层Controller
├── models/
│   ├── high_policy.py            # 高层策略网络
│   ├── low_policy.py             # 低层策略网络（共享主干）
│   ├── value_nets.py             # 价值网络
│   └── embeddings.py             # 文本编码器
├── llm/
│   ├── llm_interface.py          # LLM调用接口（支持多provider）
│   ├── prompt_templates.py       # 各类推理任务的prompt模板
│   ├── cache.py                  # LLM输出缓存（节省成本）
│   └── token_counter.py          # Token计数和预算管理
├── training/
│   ├── stage0_bc.py              # Stage 0: BC初始化
│   ├── stage1_alternating.py    # Stage 1: 交替冻结训练
│   ├── stage2_joint.py           # Stage 2: 联合微调
│   ├── ppo_trainer.py            # PPO训练器（基于RLlib）
│   └── rule_policies.py          # 规则策略（用于BC数据收集）
├── evaluation/
│   ├── evaluator.py              # 评估器
│   ├── metrics.py                # 评估指标计算
│   └── analysis.py               # 深度分析工具
├── experiments/
│   ├── e1_baseline.py            # E1实验配置
│   ├── e2_verifier.py            # E2实验配置
│   ├── e3_critic.py              # E3实验配置
│   ├── e4_interaction.py         # E4实验配置
│   ├── e5_budget.py              # E5实验配置
│   └── ablations/                # 消融实验配置
│       ├── a1_learned_vs_rule.py
│       ├── a2_interaction.py
│       └── ...
├── utils/
│   ├── data_loader.py            # 数据集加载
│   ├── logger.py                 # 日志和wandb集成
│   └── visualization.py          # 可视化工具
├── configs/
│   ├── default.yaml              # 默认超参数
│   ├── gsm8k.yaml                # GSM8K数据集配置
│   └── math.yaml                 # MATH数据集配置
├── scripts/
│   ├── run_stage0.sh             # Stage 0训练脚本
│   ├── run_experiments.sh        # 批量实验脚本
│   └── analyze_results.py        # 结果分析脚本
├── requirements.txt
├── setup.py
└── README.md
```

---

## RLlib集成方案

### 1. 环境注册

```python
from ray.rllib.env import MultiAgentEnv

class HMARLSCEnv(MultiAgentEnv):
    def __init__(self, config):
        self.high_level_env = HighLevelEnv(config)
        self.low_level_env = LowLevelEnv(config)
        self.current_level = "high"  # "high" or "low"

    def reset(self):
        # 重置为高层环境
        self.current_level = "high"
        return self.high_level_env.reset()

    def step(self, action_dict):
        if self.current_level == "high":
            # 高层动作：选择goal, focus, budget
            obs, reward, done, info = self.high_level_env.step(action_dict["controller"])

            if not done and action_dict["controller"] != "STOP":
                # 切换到低层环境
                self.current_level = "low"
                self.low_level_env.set_macro_action(action_dict["controller"])
                return self.low_level_env.reset()
        else:
            # 低层动作：三个agent同时行动
            obs, reward, done, info = self.low_level_env.step(action_dict)

            if done:  # 低层轮次结束
                # 切换回高层
                self.current_level = "high"
                macro_output = self.low_level_env.get_macro_output()
                return self.high_level_env.update_from_macro(macro_output)

        return obs, reward, done, info
```

### 2. 策略配置

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(HMARLSCEnv, env_config={
        "dataset": "gsm8k",
        "llm_model": "Qwen/Qwen3-7B-Instruct",
        "budget": 4000,  # 降低预算
    })
    .multi_agent(
        policies={
            "high_policy": PolicySpec(
                policy_class=HighLevelPolicy,
                observation_space=high_obs_space,
                action_space=high_action_space,
            ),
            "low_policy": PolicySpec(
                policy_class=LowLevelPolicy,
                observation_space=low_obs_space,
                action_space=low_action_space,
            ),
        },
        policy_mapping_fn=lambda agent_id, episode, **kwargs: (
            "high_policy" if agent_id == "controller" else "low_policy"
        ),
    )
    .training(
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.1,
        entropy_coeff=0.01,
        train_batch_size=2000,  # 减小batch size
        sgd_minibatch_size=64,   # 减小minibatch
        num_sgd_iter=10,
    )
    .resources(
        num_gpus=1,  # 单卡
        num_cpus_per_worker=2,
        num_gpus_per_worker=0.2,  # 共享GPU
    )
    .rollouts(
        num_rollout_workers=4,  # 减少worker数量
        num_envs_per_worker=1,
    )
)
```

---

## 超参数配置

---

## 超参数配置

### 全局超参数（default.yaml）

```yaml
# 环境配置
environment:
  dataset: "gsm8k"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  max_episodes: 30000  # 减少总episode数

# LLM配置
llm:
  model: "Qwen/Qwen3-7B-Instruct"
  backend: "vllm"
  quantization: "awq"  # 4bit量化
  temperature_explore: 0.7
  temperature_diagnose: 0.3
  max_tokens: 512
  cache_enabled: true
  cache_dir: "./llm_cache"
  gpu_memory_utilization: 0.3  # LLM只用30%显存，留70%给RL训练

# 预算配置
budget:
  k_equiv: 10  # 等价SC采样次数
  single_cot_tokens: 400  # 7B模型输出较短
  total_budget: 4000  # k_equiv × single_cot_tokens
  light_ratio: 0.05
  standard_ratio: 0.15
  heavy_ratio: 0.30

# 高层配置
high_level:
  gamma: 0.99
  max_rounds: 8  # 减少最大轮次
  obs_dim: 271  # 15统计特征 + 256 embedding
  hidden_dim: 128  # 减小隐层维度，节省显存

# 低层配置
low_level:
  gamma: 0.95
  k_max_light: 3
  k_max_standard: 5
  k_max_heavy: 8
  obs_dim: 512
  hidden_dim: 128  # 减小隐层维度
  shared_trunk: true

# 奖励配置
reward:
  alpha: 0.2  # accuracy-cost权衡
  beta_high: 0.00025  # 1/4000
  beta_low: 0.00083  # 1/(3×400)
  beta_verifier: 0.1
  eta: 0.5

# PPO训练配置
ppo:
  lr_policy: 3.0e-4
  lr_value: 1.0e-3
  clip_epsilon: 0.1
  lambda_gae: 0.95
  entropy_coeff_high: 0.01
  entropy_coeff_low: 0.02
  train_batch_size: 2000  # 减小batch size
  sgd_minibatch_size: 64  # 减小minibatch
  num_sgd_iter: 10

# Stage 0: BC初始化
stage0:
  num_episodes: 3000  # 减少数据收集量
  rule_policies: ["simple", "verify", "challenge"]  # 减少到3种规则
  bc_epochs: 30  # 减少epoch
  bc_batch_size: 32  # 减小batch size

# Stage 1: 交替冻结训练
stage1:
  num_rounds: 3  # 减少轮数
  phase_a_steps: 1000  # 减少步数
  phase_b_steps: 1000
  decay_factor: 0.8
  min_steps: 300

# Stage 2: 联合微调
stage2:
  num_steps: 5000  # 减少步数
  lr_decay: 0.33
  check_interval: 100

# 监控配置
monitoring:
  wandb_project: "hmarl-sc"
  log_interval: 10
  eval_interval: 50  # 更频繁评估
  save_interval: 200
```

---

## 详细训练流程

### Stage 0: 行为克隆初始化

**目标**：用规则策略收集数据，初始化策略网络

**步骤**：

1. **实现规则策略**（training/rule_policies.py）

```python
class RulePolicy:
    """规则策略基类"""

    def get_high_level_action(self, state, round_num):
        """返回高层动作"""
        raise NotImplementedError

    def get_low_level_actions(self, goal, focus, blackboard, step):
        """返回低层三agent的动作"""
        raise NotImplementedError

class SimpleRule(RulePolicy):
    """EXPLORE×3 → STOP(majority)"""

    def get_high_level_action(self, state, round_num):
        if round_num < 3:
            return ("EXPLORE", "open", "standard")
        else:
            return ("STOP", "majority")

    def get_low_level_actions(self, goal, focus, blackboard, step):
        if step == 1:
            return {
                "proposer": ("generate", "submit-trace"),
                "critic": ("work-idle", "comm-idle"),
                "verifier": ("work-idle", "comm-idle"),
            }
        elif step == 2:
            return {
                "proposer": ("work-idle", "comm-idle"),
                "critic": ("work-idle", "comm-idle"),
                "verifier": ("quick-verify", "submit-score"),
            }
        else:
            return {agent: ("work-idle", "comm-idle") for agent in ["proposer", "critic", "verifier"]}

# 类似实现 VerifyRule, ChallengeRule, FullRule
```

2. **数据收集**（scripts/collect_bc_data.py）

```python
def collect_bc_data(dataset, num_episodes=5000):
    """收集BC训练数据"""

    rules = [SimpleRule(), VerifyRule(), ChallengeRule(), FullRule()]
    trajectories = []

    for episode_id in tqdm(range(num_episodes)):
        # 随机选择规则
        rule = random.choice(rules)

        # 执行episode
        env = HMARLSCEnv(config)
        obs = env.reset()
        done = False
        trajectory = []

        round_num = 0
        while not done:
            if env.current_level == "high":
                action = rule.get_high_level_action(obs, round_num)
                trajectory.append(("high", obs, action))
                round_num += 1
            else:
                actions = rule.get_low_level_actions(
                    env.current_goal, env.current_focus,
                    env.blackboard, env.current_step
                )
                trajectory.append(("low", obs, actions))

            obs, reward, done, info = env.step(action)

        # 只保留成功的episode
        if info["correct"]:
            trajectories.append(trajectory)

    return trajectories
```

3. **行为克隆训练**（training/stage0_bc.py）

```python
def train_bc(trajectories, epochs=50):
    """行为克隆训练"""

    # 分离高层和低层数据
    high_data = [(obs, action) for traj in trajectories
                 for level, obs, action in traj if level == "high"]
    low_data = [(obs, actions) for traj in trajectories
                for level, obs, actions in traj if level == "low"]

    # 初始化网络
    high_policy = HighLevelPolicy()
    low_policy = LowLevelPolicy()

    optimizer_high = torch.optim.Adam(high_policy.parameters(), lr=1e-3)
    optimizer_low = torch.optim.Adam(low_policy.parameters(), lr=1e-3)

    for epoch in range(epochs):
        # 训练高层
        for batch in DataLoader(high_data, batch_size=64, shuffle=True):
            obs, actions = batch
            logits = high_policy(obs)
            loss = F.cross_entropy(logits, actions)

            optimizer_high.zero_grad()
            loss.backward()
            optimizer_high.step()

        # 训练低层（三个agent）
        for batch in DataLoader(low_data, batch_size=64, shuffle=True):
            obs, actions = batch
            logits = low_policy(obs)
            loss = compute_multi_agent_loss(logits, actions)

            optimizer_low.zero_grad()
            loss.backward()
            optimizer_low.step()

    return high_policy, low_policy
```

---

### Stage 1: 交替冻结训练

**目标**：交替训练高层和低层，避免同时优化导致的不稳定

**步骤**：

```python
def stage1_alternating_training(
    high_policy, low_policy,
    high_value, low_value,
    num_rounds=5
):
    """交替冻结训练"""

    phase_a_steps = 2000
    phase_b_steps = 2000

    for round_idx in range(num_rounds):
        print(f"Round {round_idx+1}/{num_rounds}")

        # ===== Phase A: 训练低层，冻结高层 =====
        high_policy.eval()  # 冻结
        low_policy.train()

        for step in range(phase_a_steps):
            # 收集数据（高层用frozen策略）
            batch = collect_rollouts(
                env, high_policy, low_policy,
                num_episodes=10
            )

            # 提取低层trajectory
            low_trajectories = extract_low_level_data(batch)

            # 计算GAE advantage
            advantages = compute_gae(
                low_trajectories, low_value,
                gamma=0.95, lambda_=0.95
            )

            # PPO更新低层策略
            ppo_update(low_policy, low_trajectories, advantages)

            # 更新低层价值网络
            value_loss = update_value_net(low_value, low_trajectories)

            # 监控退化
            if step % 100 == 0:
                check_low_level_degradation(low_policy, batch)

        # ===== Phase B: 训练高层，冻结低层 =====
        high_policy.train()
        low_policy.eval()  # 冻结

        for step in range(phase_b_steps):
            # 收集数据（低层用frozen策略）
            batch = collect_rollouts(
                env, high_policy, low_policy,
                num_episodes=10
            )

            # 提取高层trajectory
            high_trajectories = extract_high_level_data(batch)

            # 计算GAE advantage
            advantages = compute_gae(
                high_trajectories, high_value,
                gamma=0.99, lambda_=0.95
            )

            # PPO更新高层策略
            ppo_update(high_policy, high_trajectories, advantages)

            # 更新高层价值网络
            value_loss = update_value_net(high_value, high_trajectories)

            # 监控
            if step % 100 == 0:
                monitor_high_level_policy(high_policy, batch)

        # 衰减步数
        phase_a_steps = max(int(phase_a_steps * 0.8), 500)
        phase_b_steps = max(int(phase_b_steps * 0.8), 500)
```

---

### Stage 2: 联合微调

**目标**：同时优化高层和低层，精细调整

```python
def stage2_joint_finetuning(
    high_policy, low_policy,
    high_value, low_value,
    num_steps=10000
):
    """联合微调"""

    # 降低学习率
    optimizer_high = torch.optim.Adam(high_policy.parameters(), lr=1e-4)
    optimizer_low = torch.optim.Adam(low_policy.parameters(), lr=1e-4)

    # 动态entropy系数
    entropy_coeff_high = 0.01
    entropy_coeff_low = [0.02, 0.02, 0.02]  # 三个角色

    for step in range(num_steps):
        # 收集数据
        batch = collect_rollouts(
            env, high_policy, low_policy,
            num_episodes=10
        )

        # 同时更新高层和低层
        high_loss = ppo_update(high_policy, batch["high"],
                               entropy_coeff=entropy_coeff_high)
        low_loss = ppo_update(low_policy, batch["low"],
                              entropy_coeff=entropy_coeff_low)

        # 更新价值网络
        high_value_loss = update_value_net(high_value, batch["high"])
        low_value_loss = update_value_net(low_value, batch["low"])

        # 每100步检查退化
        if step % 100 == 0:
            degradation_checks(high_policy, low_policy, batch)

    return high_policy, low_policy

def degradation_checks(high_policy, low_policy, batch):
    """退化监控（文档8.3的CHECK 1-4）"""

    # CHECK 1: 角色活跃度
    for role_id, role_name in enumerate(["proposer", "critic", "verifier"]):
        idle_ratio = compute_idle_ratio(batch, role_id)
        if idle_ratio > 0.95:
            print(f"WARNING: {role_name} idle ratio = {idle_ratio:.2f}")
            # 增加该角色的entropy bonus
            entropy_coeff_low[role_id] *= 2

    # CHECK 2: 交互活跃度
    comm_idle_ratio = compute_comm_idle_ratio(batch)
    if comm_idle_ratio > 0.90:
        print(f"WARNING: comm idle ratio = {comm_idle_ratio:.2f}")
        # 增加comm动作的entropy bonus

    # CHECK 3: 角色区分度
    kl_divs = compute_role_kl_divergence(batch)
    avg_kl = np.mean(kl_divs)
    if avg_kl < 0.1:
        print(f"WARNING: role convergence, avg KL = {avg_kl:.3f}")
        # 增加角色特化的辅助loss

    # CHECK 4: Verifier退化
    score_std = compute_verifier_score_std(batch)
    if score_std < 0.05:
        print(f"WARNING: Verifier score std = {score_std:.3f}")
        # 增加Verifier辅助奖励系数
```

---

## 实验执行计划

### E1: 基础框架（仅Proposer + Controller）
**配置**：
- 高层动作：EXPLORE(open), STOP(majority)
- 低层：仅Proposer生成推理链
- 无Verifier、无Critic

**对比基线**：
- SC-K（K=5,10,20,40）
- 固定轮次策略（3轮EXPLORE + STOP）

**评估指标**：
- Accuracy vs K_equiv曲线
- 平均停止轮次
- 简单题vs难题的预算分配差异

**预期结果**：验证动态停止策略是否优于固定K采样

---

### E2: 加入验证证据（E1 + Verifier）
**新增组件**：
- Verifier agent（quick-verify, step-verify）
- weighted聚合：w(y) = p_T(y) × v̄(y)

**对比**：
- E1（majority聚合）
- E2（weighted聚合）

**评估指标**：
- Accuracy提升
- Verifier score与正确性的相关性（AUC）
- 验证覆盖率（被验证的distinct答案比例）

**预期结果**：验证信号能否改善聚合质量，特别是在答案分布接近的情况下

---

### E3: 加入负向证据（E2 + Critic）
**新增组件**：
- Critic agent（critique-logic, find-counterexample）
- survived机制：过滤被CHALLENGE的FLAW
- weighted聚合扩展：w(y) = p_T(y) × v̄(y) × (1 - surv_attack(y))

**对比**：
- E2（无Critic）
- E3（有Critic + survived过滤）
- E3-nofilter（有Critic但不过滤）

**评估指标**：
- Accuracy提升
- Critic发现的有效缺陷比例
- survived机制的过滤效果（被过滤的FLAW中有多少是误报）

**预期结果**：负向证据能否发现并修正错误答案

---

### E4: 完整交互动作（E3 + 交互协议）
**新增组件**：
- REQUEST：agent间主动请求
- CHALLENGE：质疑已有消息
- ENDORSE：支持已有消息
- 信息板机制

**对比**：
- E3（agent独立执行，无交互）
- E4（完整交互）
- E4-broadcast（仅广播，无定向交互）

**评估指标**：
- Accuracy提升
- REQUEST响应率（被请求的agent是否执行了对应动作）
- CHALLENGE成功率（被CHALLENGE的消息中有多少确实有问题）
- 交互频率统计（各类comm动作的使用频率）

**预期结果**：agent间交互是否产生协同效应

---

### E5: 预算自适应分配（E4 + δB_t）
**新增组件**：
- 高层动作包含预算分配：δB_t ∈ {light, standard, heavy}
- 轮内预算约束

**对比**：
- E4（固定standard预算）
- E5（学习预算分配）

**评估指标**：
- Accuracy vs 总预算曲线（Pareto前沿）
- 简单题vs难题的预算分配模式
- 预算利用效率（正确题的平均成本 vs 错误题的平均成本）

**预期结果**：自适应预算分配能否提高资源利用效率

---

## 阶段二：消融实验（A1-A11）

### A1: 学习策略 vs 最佳固定规则
**对比**：
- HMARL-SC（学习策略）
- Rule-Simple, Rule-Verify, Rule-Challenge, Rule-Full（文档0a中的规则）
- 最佳规则（在验证集上选择表现最好的规则）

**问题**：RL是否必要？

---

### A2: 有交互 vs 无交互
**对比**：
- E4（完整交互）
- E4-independent（三agent独立执行，不共享信息板）

**问题**：交互是否有价值？

---

### A3: REQUEST动作的价值
**对比**：
- E4（有REQUEST）
- E4-noREQUEST（禁用REQUEST动作）

**分析**：
- REQUEST的使用频率
- 被请求agent的响应率
- REQUEST对最终结果的影响

---

### A4: CHALLENGE/ENDORSE的价值
**对比**：
- E4（有CHALLENGE/ENDORSE）
- E4-noCE（禁用CHALLENGE/ENDORSE）

**分析**：
- CHALLENGE的准确性（被CHALLENGE的消息中有多少确实有问题）
- ENDORSE的一致性（被多agent ENDORSE的消息是否更可靠）

---

### A5: survived过滤机制
**对比**：
- E3（有survived过滤）
- E3-nofilter（所有FLAW都计入）
- E3-oracle（用ground truth过滤FLAW）

**问题**：survived机制是否有效过滤误报？

---

### A6: 参数共享 vs 独立网络
**对比**：
- 共享主干 + 角色头（当前设计）
- 三个完全独立的网络

**分析**：
- 训练效率（样本效率、收敛速度）
- 角色区分度（三角色的work action分布的KL散度）

**问题**：共享是否导致角色塌缩？

---

### A7: 角色行为分析
**分析内容**：
- 各角色的work-idle率
- 各角色的主要work动作分布
- 各角色的comm动作分布
- 角色间的交互模式（谁最常REQUEST谁？）

**问题**：角色是否真的分工？

---

### A8: Verifier辅助奖励
**对比**：
- 有Verifier辅助奖励（β^V=0.1）
- 无Verifier辅助奖励（β^V=0）

**问题**：局部shaping是否必要？

---

### A9: Accuracy-Cost权衡
**实验**：
- 不同α值（0, 0.1, 0.2, 0.5, 1.0）
- 绘制Pareto曲线（Accuracy vs 平均成本）

**问题**：如何在准确性和成本间权衡？

---

### A10: BC warm-start vs 随机初始化
**对比**：
- BC初始化（Stage 0）
- 随机初始化

**评估**：
- 训练曲线（收敛速度、最终性能）
- 样本效率

---

### A11: 交替冻结 vs 联合训练
**对比**：
- 交替冻结（Stage 1）
- 从头联合训练
- 仅联合微调（跳过Stage 1）

**评估**：
- 训练稳定性
- 最终性能
- 训练时间

---

## 阶段三：深度分析

### 分析1：交互行为分析
**统计内容**：
- 各类comm动作的频率分布
- REQUEST的响应率（按请求类型分组）
- CHALLENGE的成功率（被CHALLENGE的消息中有多少确实有问题）
- ENDORSE的一致性（被多agent ENDORSE的消息的正确率）

**可视化**：
- 交互网络图（agent间的交互频率）
- 时间序列图（episode内交互模式的演变）

---

### 分析2：停止策略分析
**按题目难度分组**（用SC-40的准确率作为难度代理）：
- 简单题（SC-40准确率>90%）
- 中等题（60%-90%）
- 困难题（<60%）

**分析内容**：
- 各难度组的平均停止轮次
- 各难度组的平均预算消耗
- 答对题 vs 答错题的预算消耗对比

**可视化**：
- 停止轮次分布直方图（按难度分组）
- 预算消耗箱线图（按难度和正确性分组）

---

### 分析3：角色贡献分析
**方法**：
- 各角色的work-idle率
- 各角色产出对最终答案的影响（通过ablation：移除某角色的所有产出）
- 各角色的"关键贡献"频率（该角色的产出改变了最终答案）

**可视化**：
- 角色活跃度热力图（按episode阶段）
- 角色贡献饼图

---

### 分析4：Case Study
**选择标准**：
- 成功案例：系统答对但SC-10答错的题目
- 失败案例：系统答错但SC-10答对的题目
- 交互丰富案例：包含多种comm动作的episode

**展示内容**：
- 完整的信息板内容（按微步展开）
- agent间的交互序列
- 关键决策点的分析（为什么高层选择了某个goal？为什么Critic发起了CHALLENGE？）

---

### 分析5：失败分析
**分类失败模式**：
- 早停错误：过早STOP，未充分探索
- 晚停浪费：已有正确答案但继续探索
- 验证误导：Verifier给错误答案高分
- 批判失效：Critic未能发现明显错误
- 交互失败：REQUEST未被响应或CHALLENGE误伤正确答案

**统计各失败模式的占比**

---

## 阶段四：扩展实验（可选）

### 扩展1：更多数据集
- AQuA（代数推理）
- CommonsenseQA（常识推理）
- 评估框架的泛化能力

### 扩展2：不同LLM
- GPT-4（更强基础模型）
- Llama-2-13B（更弱基础模型）
- 评估框架对基础模型能力的依赖

### 扩展3：更长预算
- K_equiv ∈ {60, 80, 100}
- 评估框架在充足预算下的表现

---

## 实验执行顺序

### 第一周：环境搭建 + Stage 0
- 数据集准备
- 规则策略实现
- BC数据收集（5000 episodes）
- BC初始化

### 第二周：E1-E3（基础组件验证）
- E1：基础框架
- E2：加入Verifier
- E3：加入Critic

### 第三周：E4-E5 + A1-A5（完整系统 + 核心消融）
- E4：完整交互
- E5：预算分配
- A1-A5：核心消融实验

### 第四周：A6-A11 + 深度分析
- A6-A11：剩余消融实验
- 分析1-5：深度分析

### 第五周：论文撰写 + 扩展实验（可选）

---

## 评估指标汇总

### 主要指标
1. **Accuracy**：最终答案正确率
2. **平均成本**：平均token消耗（以K_equiv为单位）
3. **Pareto效率**：Accuracy-Cost曲线下面积

### 辅助指标
4. **停止轮次**：平均高层步数
5. **预算利用率**：实际消耗/总预算
6. **验证覆盖率**：被验证的distinct答案比例
7. **批判成立率**：survived=1的FLAW比例
8. **交互频率**：平均每episode的非idle comm动作数

### 分析指标
9. **角色活跃度**：各角色的work-idle率
10. **角色区分度**：三角色work action分布的平均KL散度
11. **REQUEST响应率**：被响应的REQUEST比例
12. **CHALLENGE准确率**：被CHALLENGE的消息中确实有问题的比例

---

## 预期挑战与应对

### 挑战1：训练不稳定
**症状**：策略网络崩溃、角色塌缩、全员idle
**应对**：
- 增加entropy bonus
- 使用BC warm-start
- 监控退化指标（文档8.3的CHECK 1-4）

### 挑战2：LLM调用成本
**症状**：实验成本过高
**应对**：
- 使用较小数据集（SVAMP）进行快速迭代
- 使用开源模型（Llama-2）
- 缓存LLM输出（相同输入复用结果）

### 挑战3：评估时间长
**症状**：完整评估需要数天
**应对**：
- 使用验证集（15%）进行快速评估
- 并行化episode执行
- 优先评估关键配置

---

## 成功标准

### 最低标准（必须达到）
- E5在GSM8K上超越SC-K（相同预算下）
- 至少一个消融实验显示组件有显著贡献（p<0.05）
- 训练过程稳定（无角色塌缩）

### 理想标准（期望达到）
- E5在所有数据集上超越SC-K和BoN
- 所有核心组件（Verifier, Critic, 交互）都有显著贡献
- Case study展示清晰的agent协作模式
- Pareto曲线显示框架在不同预算下都有优势

### 卓越标准（最佳情况）
- 在K_equiv=10时达到SC-40的准确率
- 失败分析揭示清晰的改进方向
- 框架在不同LLM和数据集上都表现稳定

---

## 评估指标实现

### 主要指标计算（evaluation/metrics.py）

```python
def compute_accuracy(predictions, ground_truth):
    """计算准确率"""
    correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred == gt)
    return correct / len(predictions)

def compute_average_cost(episodes):
    """计算平均token成本"""
    total_cost = sum(ep["total_cost"] for ep in episodes)
    return total_cost / len(episodes)

def compute_k_equiv(episodes, single_cot_cost=500):
    """计算等价SC采样次数"""
    avg_cost = compute_average_cost(episodes)
    return avg_cost / single_cot_cost

def compute_pareto_efficiency(results):
    """计算Pareto曲线下面积"""
    # results: [(accuracy, cost), ...]
    sorted_results = sorted(results, key=lambda x: x[1])  # 按成本排序
    auc = np.trapz([r[0] for r in sorted_results], [r[1] for r in sorted_results])
    return auc

def compute_verification_coverage(episodes):
    """计算验证覆盖率"""
    total_answers = 0
    verified_answers = 0

    for ep in episodes:
        distinct_answers = set(trace["answer"] for trace in ep["traces"])
        verified = set(score["answer"] for score in ep["verification_scores"])
        total_answers += len(distinct_answers)
        verified_answers += len(verified)

    return verified_answers / total_answers if total_answers > 0 else 0

def compute_critique_survival_rate(episodes):
    """计算批判成立率"""
    total_critiques = 0
    survived_critiques = 0

    for ep in episodes:
        for critique in ep["critiques"]:
            total_critiques += 1
            if critique["survived"]:
                survived_critiques += 1

    return survived_critiques / total_critiques if total_critiques > 0 else 0
```

---

## 实验运行脚本示例

### 完整实验流程（scripts/run_full_pipeline.sh）

```bash
#!/bin/bash

# ========================================
# HMARL-SC 完整实验流程
# ========================================

set -e  # 遇到错误立即退出

# 配置
DATASET="gsm8k"
WANDB_PROJECT="hmarl-sc-full"
NUM_GPUS=2

echo "========================================="
echo "Stage 0: BC初始化"
echo "========================================="

# 收集BC数据
python scripts/collect_bc_data.py \
  --dataset $DATASET \
  --num_episodes 5000 \
  --output_dir ./data/bc_trajectories \
  --num_workers 8

# BC训练
python training/stage0_bc.py \
  --data_dir ./data/bc_trajectories \
  --epochs 50 \
  --batch_size 64 \
  --output_dir ./checkpoints/stage0 \
  --wandb_project $WANDB_PROJECT

echo "========================================="
echo "Stage 1: 交替冻结训练"
echo "========================================="

python training/stage1_alternating.py \
  --config configs/default.yaml \
  --dataset $DATASET \
  --init_checkpoint ./checkpoints/stage0/final.pt \
  --num_rounds 5 \
  --phase_a_steps 2000 \
  --phase_b_steps 2000 \
  --num_gpus $NUM_GPUS \
  --wandb_project $WANDB_PROJECT

echo "========================================="
echo "Stage 2: 联合微调"
echo "========================================="

python training/stage2_joint.py \
  --config configs/default.yaml \
  --dataset $DATASET \
  --init_checkpoint ./checkpoints/stage1/final.pt \
  --num_steps 10000 \
  --num_gpus $NUM_GPUS \
  --wandb_project $WANDB_PROJECT

echo "========================================="
echo "分阶段验证实验 (E1-E5)"
echo "========================================="

for exp in e1_baseline e2_verifier e3_critic e4_interaction e5_budget; do
  echo "Running $exp..."
  python experiments/${exp}.py \
    --config configs/${exp}.yaml \
    --dataset $DATASET \
    --k_equiv 10 \
    --num_trials 3 \
    --wandb_project $WANDB_PROJECT
done

echo "========================================="
echo "消融实验 (A1-A11)"
echo "========================================="

python experiments/run_ablations.py \
  --config configs/e5_budget.yaml \
  --dataset $DATASET \
  --ablation_list a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11 \
  --wandb_project $WANDB_PROJECT

echo "========================================="
echo "深度分析"
echo "========================================="

python evaluation/run_analysis.py \
  --results_dir ./results \
  --output_dir ./analysis \
  --dataset $DATASET

echo "========================================="
echo "实验完成！"
echo "========================================="
```

---

## 依赖清单（单卡A6000优化版）

### requirements.txt

```txt
# 强化学习框架
ray[rllib]==2.9.0
gymnasium==0.29.1
pettingzoo==1.24.3

# 深度学习
torch==2.1.0
transformers==4.36.0
sentence-transformers==2.2.2

# LLM推理（本地部署）
vllm==0.3.0  # 高效推理引擎
autoawq==0.1.8  # 4bit量化
bitsandbytes==0.41.3  # 备选量化方案

# 数据处理
datasets==2.16.0
pandas==2.1.4
numpy==1.24.3

# 实验管理
wandb==0.16.2

# 可视化
matplotlib==3.8.2
seaborn==0.13.0

# 工具
tqdm==4.66.1
pyyaml==6.0.1
```

### 安装脚本（scripts/setup_env.sh）

```bash
#!/bin/bash

# 创建conda环境
conda create -n hmarl-sc python=3.10 -y
conda activate hmarl-sc

# 安装PyTorch（CUDA 11.8）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 安装vLLM（用于高效LLM推理）
pip install vllm==0.3.0

# 安装其他依赖
pip install -r requirements.txt

# 下载并量化Qwen2.5-7B-Instruct
python scripts/download_and_quantize_model.py

# 验证安装
python -c "import ray; import torch; import vllm; print('Setup complete!')"

echo "环境安装完成！"
```

### 模型下载脚本（scripts/download_and_quantize_model.py）

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM

# 下载模型
model_path = "Qwen/Qwen3-7B-Instruct"
quant_path = "./models/qwen3-7b-instruct-awq"

print("下载模型...")
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("量化模型（4bit AWQ）...")
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4}
model.quantize(tokenizer, quant_config=quant_config)

print("保存量化模型...")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"量化完成！模型保存在 {quant_path}")
```

---

## 单卡A6000显存优化策略

### 显存分配方案

```
总显存: 48GB
├─ LLM推理 (vLLM): 14GB (30%)
│  └─ Qwen2.5-7B-Instruct (4bit AWQ量化)
├─ RL训练: 20GB (42%)
│  ├─ 策略网络: 8GB
│  ├─ 价值网络: 4GB
│  ├─ Replay buffer: 4GB
│  └─ 梯度和优化器: 4GB
├─ 环境和缓存: 8GB (17%)
└─ 系统预留: 6GB (11%)
```

### 关键优化技术

**1. LLM推理优化**
```python
# 使用vLLM的PagedAttention，显存效率提升2-4x
from vllm import LLM, SamplingParams

llm = LLM(
    model="./models/qwen2.5-7b-instruct-awq",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.3,  # 只用30%显存
    max_model_len=2048,  # 限制上下文长度
    swap_space=4,  # 使用CPU swap
)
```

**2. 策略网络轻量化**
```python
# 减小隐层维度
high_level:
  hidden_dim: 128  # 原256 → 128，参数量减少75%

low_level:
  hidden_dim: 128  # 原256 → 128

# 使用混合精度训练
torch.set_float32_matmul_precision('medium')
```

**3. 批量大小调整**
```yaml
ppo:
  train_batch_size: 2000  # 原4000 → 2000
  sgd_minibatch_size: 64  # 原128 → 64
```

**4. 梯度累积**
```python
# 如果显存仍不足，使用梯度累积
gradient_accumulation_steps: 2  # 等效batch size = 2000 × 2 = 4000
```

**5. LLM输出缓存**
```python
# 缓存LLM输出，避免重复推理
class LLMCache:
    def __init__(self, cache_dir="./llm_cache"):
        self.cache = {}
        self.cache_dir = cache_dir

    def get_or_generate(self, prompt, params):
        key = hash((prompt, str(params)))
        if key in self.cache:
            return self.cache[key]

        output = llm.generate(prompt, params)
        self.cache[key] = output
        return output
```

**6. 并行化策略**
```python
# 使用Ray的actor模式，LLM推理和RL训练分离
@ray.remote(num_gpus=0.3)
class LLMInferenceActor:
    def __init__(self):
        self.llm = LLM(...)

    def generate(self, prompt):
        return self.llm.generate(prompt)

@ray.remote(num_gpus=0.7)
class RLTrainingActor:
    def __init__(self):
        self.policy = Policy(...)

    def train_step(self, batch):
        return self.policy.update(batch)
```

---

## 预期性能对比

### 7B模型 vs GPT-3.5-turbo

| 指标 | GPT-3.5-turbo | Qwen3-7B-Instruct | 差异 |
|------|---------------|-------------------|------|
| GSM8K准确率（单次CoT） | 78% | 78% | 持平 |
| 推理速度 | ~2s/次（API） | ~0.5s/次（本地） | 4x快 |
| 成本 | $0.002/1K tokens | 免费（本地） | 节省100% |
| 输出长度 | ~500 tokens | ~400 tokens | -20% |
| 显存占用 | N/A | 14GB（4bit） | - |

**预期影响**：
- Qwen3-7B在数学推理上与GPT-3.5-turbo持平
- 训练速度快4x，成本为0
- 适合快速迭代和验证框架有效性

---

## 单卡资源下的实验简化建议

### 优先级调整

**必做实验（核心验证）**：
1. **E1-E5**：分阶段验证（证明框架有效性）
2. **A1**：学习策略 vs 规则（证明RL必要性）
3. **A2**：有交互 vs 无交互（证明交互价值）
4. **分析1-2**：交互行为分析 + 停止策略分析

**可选实验（时间允许）**：
- A3-A5：细粒度交互分析
- A6-A8：训练策略分析
- 分析3-5：深度分析

**可跳过实验**：
- A9：Pareto曲线（可用E5不同α值替代）
- A10-A11：训练策略对比（如果Stage 1效果好）
- 扩展实验：其他数据集、其他LLM

### 数据集简化

**主数据集**：GSM8K（小学数学，8473题）
- 训练集：5931题（70%）
- 验证集：1271题（15%）
- 测试集：1271题（15%）

**快速验证集**：GSM8K-mini（自建，200题）
- 用于快速迭代和调试
- 包含简单/中等/困难题各占1/3

**跳过数据集**：MATH、SVAMP、StrategyQA
- 如果时间紧张，专注GSM8K即可

### 训练加速技巧

**1. 使用更小的验证集**
```python
# 每100步只在200题上验证，而非全部1271题
eval_subset_size: 200
```

**2. Early stopping**
```python
# 验证集准确率连续3次不提升则停止
early_stopping_patience: 3
```

**3. 检查点复用**
```python
# E2-E5复用E1的部分权重
load_pretrained: "./checkpoints/e1/final.pt"
freeze_shared_trunk: true  # 冻结共享主干，只训练新增部分
```

**4. 减少BC数据量**
```python
# Stage 0只收集1000个成功episode
stage0:
  num_episodes: 1000  # 原3000
  min_success_rate: 0.3  # 只要30%成功即可
```

---

## 修订后的时间线（单卡优化版）

**第1周：环境搭建 + Stage 0**
- Day 1: 安装vLLM，下载并量化Qwen2.5-7B（~4小时）
- Day 2: 实现环境类和LLM接口（测试单个episode）
- Day 3: 实现规则策略（3种）
- Day 4-5: 收集BC数据（1000 episodes，~8小时）
- Day 6-7: BC训练（~4小时）+ 验证

**第2周：E1-E3**
- Day 1-2: E1实验（~12小时训练 + 评估）
- Day 3-4: E2实验（复用E1权重，~8小时）
- Day 5-7: E3实验（~12小时）

**第3周：E4-E5 + 核心消融**
- Day 1-3: E4实验（~16小时）
- Day 4-5: E5实验（~12小时）
- Day 6-7: A1-A2消融（~8小时）

**第4周：分析**
- Day 1-3: 交互行为分析 + 停止策略分析
- Day 4-5: Case Study（选3-5个代表性案例）
- Day 6-7: 失败分析 + 整理结果

**第5周：论文撰写**
- Day 1-5: 论文撰写
- Day 6-7: 补充实验（如果reviewer要求）

**总训练时间估算**：~80小时GPU时间（单卡A6000）

---

## 关键文件清单

### 必须实现的核心文件（按优先级）

**第一优先级（Week 1）**：
1. `envs/reasoning_env.py` - 主环境类
2. `envs/blackboard.py` - 共享信息板
3. `llm/llm_interface.py` - LLM调用接口
4. `llm/prompt_templates.py` - Prompt模板
5. `training/rule_policies.py` - 规则策略

**第二优先级（Week 2）**：
6. `models/high_policy.py` - 高层策略网络
7. `models/low_policy.py` - 低层策略网络
8. `models/value_nets.py` - 价值网络
9. `training/stage0_bc.py` - BC训练
10. `experiments/e1_baseline.py` - E1实验

**第三优先级（Week 3-4）**：
11. `training/stage1_alternating.py` - 交替训练
12. `training/stage2_joint.py` - 联合微调
13. `experiments/e2_verifier.py` - E2实验
14. `experiments/e3_critic.py` - E3实验
15. `experiments/e4_interaction.py` - E4实验
16. `experiments/e5_budget.py` - E5实验

**第四优先级（Week 4-5）**：
17. `evaluation/metrics.py` - 评估指标
18. `evaluation/interaction_analysis.py` - 交互分析
19. `evaluation/stopping_analysis.py` - 停止策略分析
20. `evaluation/failure_analysis.py` - 失败分析

---

## 验证计划

### 单元测试

```python
# tests/test_blackboard.py
def test_blackboard_message_storage():
    """测试信息板消息存储"""
    bb = Blackboard()
    msg = Message(sender=1, type="TRACE", content=("tau", "y"))
    bb.add_message(msg)
    assert len(bb.messages) == 1

# tests/test_high_policy.py
def test_high_policy_forward():
    """测试高层策略前向传播"""
    policy = HighLevelPolicy(obs_dim=271)
    obs = torch.randn(1, 271)
    goal_logits, focus_logits, budget_logits = policy(obs)
    assert goal_logits.shape == (1, 5)

# tests/test_low_policy.py
def test_low_policy_role_heads():
    """测试低层策略角色头"""
    policy = LowLevelPolicy()
    obs = torch.randn(1, 512)
    for role in ["proposer", "critic", "verifier"]:
        work_logits, comm_logits = policy(obs, role)
        assert work_logits.shape[0] == 1
```

### 集成测试

```python
# tests/test_integration.py
def test_full_episode():
    """测试完整episode执行"""
    env = HMARLSCEnv(config)
    obs = env.reset()
    done = False
    steps = 0

    while not done and steps < 100:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        steps += 1

    assert done or steps == 100
    assert "correct" in info
```

### 端到端验证

```bash
# 快速验证脚本（scripts/quick_validation.sh）
#!/bin/bash

echo "运行快速验证..."

# 1. 测试环境
python -c "from envs.reasoning_env import HMARLSCEnv; print('Environment OK')"

# 2. 测试LLM接口（使用缓存，不实际调用）
python -c "from llm.llm_interface import LLMInterface; print('LLM Interface OK')"

# 3. 运行1个episode
python scripts/test_single_episode.py --dataset gsm8k --use_cache

# 4. 运行BC训练1个epoch
python training/stage0_bc.py --data_dir ./data/bc_trajectories --epochs 1 --dry_run

echo "验证完成！"
```

---

## 预期输出示例

### 实验结果表格

```
| Method      | GSM8K Acc | Avg Cost | K_equiv | Stop Rounds |
|-------------|-----------|----------|---------|-------------|
| SC-5        | 65.2%     | 2500     | 5.0     | -           |
| SC-10       | 71.8%     | 5000     | 10.0    | -           |
| SC-20       | 75.3%     | 10000    | 20.0    | -           |
| E1-Baseline | 72.5%     | 4200     | 8.4     | 3.2         |
| E2-Verifier | 74.1%     | 4500     | 9.0     | 3.5         |
| E3-Critic   | 76.2%     | 4800     | 9.6     | 3.8         |
| E4-Interact | 77.8%     | 4600     | 9.2     | 3.6         |
| E5-Budget   | 78.5%     | 4300     | 8.6     | 3.4         |
```

### 可视化示例

1. **Accuracy vs K_equiv曲线**：展示HMARL-SC在不同预算下的表现
2. **Pareto前沿**：Accuracy-Cost权衡曲线
3. **停止轮次分布**：按题目难度分组的直方图
4. **交互网络图**：agent间交互频率的有向图
5. **角色活跃度热力图**：各角色在不同阶段的活跃度

---

## 总结

本实验计划提供了HMARL-SC框架的完整实现路线图，针对单卡A6000 GPU进行了全面优化：

1. **技术栈**：Ray RLlib + PyTorch + Qwen3-7B-Instruct（本地部署）
2. **硬件配置**：单卡A6000 (48GB)，显存分配：LLM 30% + RL训练 42% + 环境 17% + 系统 11%
3. **优化策略**：vLLM + 4bit AWQ量化 + 减小模型维度（hidden_dim: 128）+ 减小批量（batch_size: 2000）
4. **代码结构**：模块化设计，20个核心文件
5. **训练流程**：3阶段训练（BC初始化 → 交替冻结 → 联合微调）
6. **实验设计**：5个分阶段实验（E1-E5）+ 核心消融（A1-A2）+ 深度分析
7. **时间规划**：5周完成，总训练时间约80小时GPU时间
8. **验证方案**：单元测试 + 集成测试 + 端到端验证

**关键创新点**：
- 两层时间尺度的多智能体协作
- 显式的可学习交互动作
- 动态预算分配和自适应停止
- 正向+负向+诊断三类证据融合

**预期贡献**：
- 在相同预算下超越SC基线5-10个百分点
- 揭示agent交互对推理质量的提升机制
- 提供可复现的开源实现
- 验证框架在资源受限环境下的有效性
