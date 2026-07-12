# Method Notes — Agentic RL Framework

> 论文 Method 部分参考文档。所有数值、公式、结构均直接来自代码。

---

## 1. 系统概述

本框架是一个基于**黑板（Blackboard）共享记忆**的多智能体强化学习系统，用于数学推理任务。单个语言模型（Qwen3-8B）通过四个独立 LoRA 适配器扮演不同角色，在同一问题上进行多轮协作推理，使用 GRPO 算法联合训练。

---

## 2. 智能体角色设计

系统包含四个角色，每个角色对应一个独立 LoRA 适配器：

### 2.1 Controller（协调者）
**职责**：每轮分析黑板状态，决定本轮协作策略。

**输出格式**：
```
<meta-plan>
strategy: [explore|refine|verify|stop]
focus: [proposer|critic|verifier|balanced]
reason: [一句话说明]
</meta-plan>
```

**策略语义**：
- `explore`：黑板信息不足，启动 Proposer 提出新解法
- `refine`：已有解法存在错误，由 Critic 主导改进
- `verify`：已有候选答案，由 Verifier 确认置信度
- `stop`：答案置信度足够，终止协作

### 2.2 Proposer（提议者）
**职责**：生成或改进推理链与候选答案；决定是否发起交互。

**输出格式**：
```
<interaction>
action: [none|request_critic|request_verifier|support:<答案>|challenge:<问题>]
target: [critic|verifier|none]
reason: [一句话]
</interaction>
推理过程：[逐步推导]
最终答案：[数值]
```

**输入上下文**：meta-plan + 问题 + 黑板当前状态

### 2.3 Critic（评论员）
**职责**：找出逻辑或计算错误；决定是否发起交互。

**输出格式**：
```
<interaction>
action: [none|request_proposer|request_verifier|support:<答案>|challenge:<问题>]
target: [proposer|verifier|none]
reason: [一句话]
</interaction>
错误分析：[有错误则描述，无错误则写"无错误"]
```

**输入上下文**：最新解法 + 答案 + 黑板状态

### 2.4 Verifier（验证者）
**职责**：独立验证答案正确性，输出置信度评分（0.0–1.0）；决定是否交互。

**输出格式**：
```
<interaction>
action: [none|request_proposer|request_critic|support:<答案>|challenge:<问题>]
target: [proposer|critic|none]
reason: [一句话]
</interaction>
分数: [0.0-1.0]
验证说明：[简要说明]
```

**输入上下文**：最新答案 + 推理过程 + 黑板状态

---

## 3. 黑板通信机制

黑板（Blackboard）是所有智能体共享的结构化记忆，支持异步读写。

### 3.1 数据结构

| 字段 | 类型 | 写入角色 | 内容 |
|------|------|----------|------|
| `traces` | `List[(reasoning, answer)]` | Proposer | 推理链和答案 |
| `flaws` | `List[{"content": str}]` | Critic | 发现的错误描述 |
| `scores` | `List[(answer, float)]` | Verifier | 答案置信度评分 |
| `interactions` | `List[{from, action, target, reason}]` | 所有角色 | 交互记录 |

### 3.2 黑板文本摘要（输入 LLM 时）
```
已有{N}个解法，答案：[answer_list]
发现问题：{latest_flaw[:80]}
最高置信答案：{best_answer}（分数{score:.2f}）
最近交互：{from}→{target}（{action}）
```

### 3.3 最终答案聚合
采用**多数投票（Majority Vote）**：对 `blackboard.traces` 中所有 Proposer 输出的答案取众数。

---

## 4. Episode 执行流程

每个 episode 对应一道问题，流程如下：

```
for round in range(max_rounds=4):
    1. Controller 读取黑板 → 输出 meta-plan
    2. if strategy == "stop": break
    3. 根据 focus 选择起始角色（focus=="balanced" → proposer）
    4. 起始角色执行，写入黑板
    5. 解析 <interaction> 块：
       for interaction in range(max_interactions=3):
           if action not in (none, support):
               目标角色响应，写入黑板
    6. 若本轮未执行 proposer，强制执行一次
    7. 记录 round_records（proposer_answer, critic_flagged, verifier_score, turn_ids）

final_answer = majority_vote(blackboard.traces)
is_correct = normalize(final_answer) == normalize(correct_answer)
```

**答案归一化**：`re.sub(r"[^0-9.\-]", "", s.strip())`（只保留数字、小数点、负号）

---

## 5. 奖励函数设计

### 5.1 终局奖励
$$r_T = \begin{cases} 1.0 & \text{if } \text{final\_answer} = \text{correct\_answer} \\ 0.0 & \text{otherwise} \end{cases}$$

分配给 episode 最后一个 turn。

### 5.2 Role-specific Shaped Rewards

对每轮（round）中各 turn 按角色精确分配：

**Critic 奖励**（当 Critic 在本轮发现错误时）：
$$r_{\text{critic}} = \begin{cases} +0.3 & \text{if critic\_flagged} \wedge \text{is\_correct} \\ -0.1 & \text{if critic\_flagged} \wedge \neg\text{is\_correct} \end{cases}$$

**Verifier 奖励**（当 Verifier 置信度判断与该轮 Proposer 答案一致时）：
$$r_{\text{verifier}} = \begin{cases} +0.1 & \text{if } (s \geq 0.5 \wedge \text{prop\_correct}) \vee (s < 0.5 \wedge \neg\text{prop\_correct}) \\ 0 & \text{otherwise} \end{cases}$$

其中 $s$ 为 Verifier 输出的置信度分数。

---

## 6. 训练算法（GRPO）

### 6.1 Advantage 估计

每个训练步对同一问题采样 $G=16$ 条轨迹，基于终局正确性（0/1）计算 episode-level advantage：

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}, \quad \mu_r = \frac{1}{G}\sum_{i=1}^G r_i, \quad \epsilon = 10^{-8}$$

### 6.2 策略梯度损失（Turn-level Ratio）

对每个 turn $t$，计算该 turn 所有响应 token 的平均对数概率比：

$$\bar{\rho}_t = \frac{1}{|t|}\sum_{k=1}^{|t|} \left[\log \pi_\theta(a_k^t | s^t) - \log \pi_{\text{old}}(a_k^t | s^t)\right]$$

$$\text{ratio}_t = \exp\left(\text{clamp}(\bar{\rho}_t, -10, 10)\right)$$

PPO 裁剪目标（clip range $\varepsilon = 0.05$）：

$$\mathcal{L}_{\text{PG}}^t = \max\left(-\hat{A} \cdot \text{ratio}_t,\ -\hat{A} \cdot \text{clamp}(\text{ratio}_t, 1-\varepsilon, 1+\varepsilon)\right)$$

当 $\hat{A} < 0$ 时额外截断：$\mathcal{L}_{\text{PG}}^t = \min(\mathcal{L}_{\text{PG}}^t,\ -\hat{A} \cdot 3.0)$

### 6.3 KL 散度约束

参考模型为禁用所有 LoRA 后的基础模型权重：

$$\mathcal{L}_{\text{KL}}^t = \frac{1}{|t|}\sum_{k=1}^{|t|} \left[\log \pi_\theta(a_k^t | s^t) - \log \pi_{\text{ref}}(a_k^t | s^t)\right]$$

### 6.4 总损失

$$\mathcal{L} = \frac{1}{N_{\text{valid}}} \sum_{t} \left(\mathcal{L}_{\text{PG}}^t + \beta \cdot \mathcal{L}_{\text{KL}}^t\right)$$

其中 $\beta$ 由自适应 KL 控制器动态调整（目标 KL = 6.0）：

$$\beta \leftarrow \beta \cdot \left(1 + \text{clip}\left(\frac{\text{KL}_{\text{cur}}}{\text{KL}_{\text{target}}} - 1,\ -0.2,\ 0.2\right) \cdot \frac{n}{H}\right)$$

horizon $H = 10000$。

---

## 7. 模型与 LoRA 配置

| 参数 | 值 |
|------|-----|
| 基础模型 | Qwen3-8B |
| 精度 | bfloat16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| 目标模块 | q_proj, v_proj |
| 独立 adapter 数量 | 4（每个角色一个） |
| 参考模型 | 禁用所有 LoRA 的同一基础模型 |
| 优化器 | AdamW8bit（8-bit 量化） |

---

## 8. 训练超参数

| 超参数 | 值 | 说明 |
|--------|-----|------|
| `epochs` | 3 | 训练轮数 |
| `batch_size` | 16 | 每步处理的问题数 |
| `n_samples` | 16 | 每道题采样轨迹数 |
| `lr` | 1e-6 | 学习率 |
| `clip_epsilon` | 0.05 | PPO 裁剪范围 |
| `max_tokens` | 512 | 每次 LLM 生成最大 token 数 |
| `max_rounds` | 4 | 每个 episode 最大协作轮数 |
| `max_interactions` | 3 | 每轮最大交互次数 |
| `kl_coef` | 0.1 | 初始 KL 系数 |
| `target_kl` | 6.0 | 目标 KL 散度 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |

---

## 9. 评估指标

| 指标 | 计算方式 |
|------|----------|
| `accuracy` | 最终答案正确率（majority vote 后） |
| `avg_turns` | 平均 turn 数 |
| `interaction_rate` | 平均交互次数 |
| `proposer_accuracy` | Proposer 直接输出正确率（不经投票） |
| `critic_precision` | $\text{TP} / (\text{TP} + \text{FP})$，Critic 标记错误时实际有误的比例 |
| `critic_flag_rate` | Critic 触发标记的比例 |
| `verifier_consistency` | Verifier 置信度判断与最终答案一致的比例 |

**Critic TP/FP 定义**：
- TP：Critic 标记了错误 且 proposer 答案确实错误
- FP：Critic 标记了错误 但 proposer 答案实际正确

---

## 10. SFT 预热

RL 训练前使用监督微调（SFT）对 Proposer adapter 预热，checkpoint 路径：`checkpoints/sft`。其余三个 adapter（controller/critic/verifier）从随机初始化开始 RL 训练。

---

## 11. 数据集

| 集合 | 文件 | 用途 |
|------|------|------|
| GSM8K train | `data/gsm8k_train.jsonl` | RL 训练（默认） |
| GSM8K test | `data/gsm8k_test.jsonl` | 评估 |
| Math train | `data/math_train_rl.jsonl` | 可选训练集 |
| Math test | `data/math_test_clean.jsonl` | 可选评估集 |

数据格式：`{"question": ..., "answer": ...}`
