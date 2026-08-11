# 论文技术细节文档
> 本文档直接从代码提取，所有数值、公式、结构均以代码为准。供论文 Method 部分写作使用。

---

## 1. 问题定义

**任务**：复杂数学推理（GSM8K / MATH），输入自然语言问题 $q$，目标输出最终正确答案 $a^*$。

**核心动机**：单智能体 LLM 在多步推理中易产生累积错误，缺乏自我校验机制。本框架引入多角色智能体协作，通过角色分工、结构化交互与强化学习联合提升推理准确性。

---

## 2. 系统架构

### 2.1 总体框架

系统由三个核心模块构成：

1. **多角色智能体执行器（AgenticExecutor）**：管理 episode 中多角色的交互与执行
2. **黑板共享记忆（Blackboard）**：所有智能体读写的结构化全局状态
3. **GRPO 训练器（GRPOAgenticTrainer）**：基于 rollout 轨迹进行策略梯度优化

**参数共享设计**：系统仅使用**一个**基础语言模型（Qwen3-8B，bfloat16），通过四个独立 LoRA 适配器区分不同角色，实现参数高效的多角色建模。

| 角色 | LoRA Adapter 名 | 职责 |
|------|----------------|------|
| Controller | `controller` | 读取全局状态，制定协作元策略 |
| Proposer   | `proposer`   | 生成推理链与候选答案 |
| Critic     | `critic`     | 检测推理错误 |
| Verifier   | `verifier`   | 对答案给出置信度评分 |

**LoRA 配置**：rank $r=16$，scaling $\alpha=32$，目标模块 `[q_proj, v_proj]`，每个 adapter 独立训练，互不干扰。

---

## 3. 黑板通信机制（Blackboard）

### 3.1 数据结构

黑板是所有智能体共享的结构化记忆，定义如下：

```
Blackboard:
  traces       : List[(reasoning: str, answer: str)]   # Proposer 写入
  flaws        : List[{"content": str}]                # Critic 写入
  scores       : List[(answer: str, score: float)]     # Verifier 写入
  interactions : List[{from, action, target, reason}]  # 所有角色写入
```

| 字段 | 类型 | 写入方 | 语义 |
|------|------|--------|------|
| `traces` | `List[(str, str)]` | Proposer | 推理过程与对应答案 |
| `flaws` | `List[dict]` | Critic | 错误描述（最近一条） |
| `scores` | `List[(str, float)]` | Verifier | 答案→置信度映射 |
| `interactions` | `List[dict]` | 全部 | 交互行为日志 |

### 3.2 黑板文本化（注入 LLM Prompt）

每次调用 LLM 前，黑板状态序列化为自然语言摘要：

```
已有{N}个解法，答案：[answer_list]
发现问题：{latest_flaw[:80]}
最高置信答案：{best_answer}（分数{score:.2f}）
最近交互：{from}→{target}（{action}）
```

### 3.3 最终答案聚合

采用**多数投票（Majority Vote）**：

$$a^* = \arg\max_{a} \sum_{(r_i, a_i) \in \text{traces}} \mathbf{1}[a_i = a]$$

答案归一化规则（消除格式差异）：

$$\text{normalize}(s) = \texttt{re.sub}(\texttt{r"[\^{}0-9.\backslash-]"}, \texttt{""}, s.\text{strip}())$$

---

## 4. 智能体角色详细设计

### 4.1 Controller（协调者）

**输入**：当前黑板状态文本 + 问题 $q$

**输出格式**：
```
<meta-plan>
strategy: [explore|refine|verify|stop]
focus: [proposer|critic|verifier|balanced]
reason: [一句话说明]
</meta-plan>
```

**策略语义**：

| 策略 | 触发条件 | 后续动作 |
|------|---------|---------|
| `explore` | 黑板为空或解法不足 | 启动 Proposer 生成新解 |
| `refine` | 存在 Critic 标记的错误 | 由 Critic 主导改进 |
| `verify` | 有候选答案，需确认 | 由 Verifier 给出置信度 |
| `stop` | 置信度足够高 | 终止协作，取当前答案 |

### 4.2 Proposer（提议者）

**输入**：meta-plan + 问题 $q$ + 黑板状态

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

### 4.3 Critic（评论员）

**输入**：最新 Proposer 解法 + 答案 + 黑板状态

**输出格式**：
```
<interaction>
action: [none|request_proposer|request_verifier|support:<答案>|challenge:<问题>]
target: [proposer|verifier|none]
reason: [一句话]
</interaction>
错误分析：[有错误则描述，无错误则写"无错误"]
```

### 4.4 Verifier（验证者）

**输入**：最新答案 + 推理过程 + 黑板状态

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

---

## 5. Episode 执行流程

### 5.1 流程伪代码

```
输入：问题 q，标准答案 a*

初始化空黑板 B

for round = 1, 2, ..., max_rounds (=4):
    1. Controller(B, q) → meta-plan {strategy, focus}
    2. if strategy == "stop": break
    3. focus_role = meta-plan.focus  (若 "balanced" 则默认 proposer)
    4. focus_role(B, q) → response；写入 B；解析 <interaction>
    5. for i = 1, 2, ..., max_interactions (=3):
           if action ∈ {none, support}: break
           target_role(B, q) → response；写入 B；解析 <interaction>
    6. if proposer 本轮未执行: 强制执行 proposer 一次

final_answer = majority_vote(B.traces)
is_correct = (normalize(final_answer) == normalize(a*))
```

### 5.2 Turn 编号与日志

每次 LLM 调用分配一个全局递增的 `turn_id`，并记录：
- `log_probs[]`：该 turn 所有 response token 的对数概率（来自 vLLM）
- `seq_input_ids[]`：完整 input token ids
- `response_ids[]`：仅 response 部分的 token ids
- `role_name`：执行该 turn 的角色名

---

## 6. 奖励函数

### 6.1 终局稀疏奖励

奖励仅在 episode 结束时分配，**附加到最后一个 Proposer turn**：

$$r_T = \begin{cases} +1.0 & \text{if } \text{final\_answer} = a^* \\ -0.5 & \text{otherwise} \end{cases}$$

### 6.2 角色 Shaped 奖励

**Critic 奖励**（当 Critic 在某轮标记了错误 `critic_flagged = True`）：

$$r_{\text{critic}} = \begin{cases} +0.3 & \text{if } \text{critic\_flagged} \wedge \text{is\_correct} \\ -0.1 & \text{if } \text{critic\_flagged} \wedge \neg\,\text{is\_correct} \end{cases}$$

逻辑：鼓励 Critic 准确识别真实错误，惩罚误报（最终答案正确但 Critic 仍报错）。

**Verifier 奖励**（当 Verifier 输出置信度 $s \in [0,1]$）：

$$r_{\text{verifier}} = \begin{cases} +0.1 & \text{if } (s \geq 0.5 \wedge \text{prop\_correct}) \vee (s < 0.5 \wedge \neg\,\text{prop\_correct}) \\ 0 & \text{otherwise} \end{cases}$$

其中 `prop_correct` 为该轮 Proposer 的答案是否正确（与标准答案逐字比较）。

### 6.3 奖励分配总结

| 角色 | 分配位置 | 条件 | 数值 |
|------|---------|------|------|
| Proposer | episode 最后一个 Proposer turn | 最终答案正确 | +1.0 |
| Proposer | episode 最后一个 Proposer turn | 最终答案错误 | −0.5 |
| Critic | 每轮 Critic turn | 标记错误 ∧ 最终正确 | +0.3 |
| Critic | 每轮 Critic turn | 标记错误 ∧ 最终错误 | −0.1 |
| Verifier | 每轮 Verifier turn | 置信度与 Proposer 一致 | +0.1 |

每个 episode 的总奖励 $r_i = \sum_t r_t$（各 turn 奖励之和）。

---

## 7. 训练算法：GRPO with Turn-level PPO

### 7.1 分组采样与 Advantage 估计

对同一问题 $q$ 独立采样 $G$ 条轨迹（代码中 `n_samples=4`，METHOD_NOTES 记录 $G=16$），得到奖励集合 $\{r_1, \ldots, r_G\}$。

组内归一化得到 **episode-level advantage**：

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}, \qquad \mu_r = \frac{1}{G}\sum_{i=1}^G r_i,\quad \sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \mu_r)^2},\quad \epsilon=10^{-8}$$

当 $\sigma_r < 10^{-6}$ 时（所有轨迹奖励相同），跳过本批次更新（无信息梯度）。

**与 PPO 的区别**：GRPO 无需额外价值网络，以同组样本的平均奖励作为基线，大幅降低训练复杂度。

### 7.2 Turn-level 概率比

对 episode $i$ 中的每个 turn $t$，其 response 包含 $|t|$ 个 token。

定义 **turn-level 对数概率比**（取所有 response token 的平均）：

$$\bar{\rho}_t = \frac{1}{|t|}\sum_{k=1}^{|t|} \Bigl[\log \pi_\theta(a_k^t \mid s^t) - \log \pi_{\text{old}}(a_k^t \mid s^t)\Bigr]$$

其中：
- $\pi_\theta$：当前策略（启用对应角色的 LoRA adapter，前向传播计算 logits）
- $\pi_{\text{old}}$：rollout 时的策略（log_probs 在 vLLM 生成时记录并保存）
- $s^t$：turn $t$ 的完整 prompt（system + user 对话模板编码后的 token 序列）
- $a_k^t$：turn $t$ 的第 $k$ 个 response token

对数概率比截断（数值保护）：

$$\text{ratio}_t = \exp\!\left(\text{clamp}(\bar{\rho}_t,\ -10,\ 10)\right)$$

如果 $|\bar{\rho}_t| > 50$，该 turn 直接跳过（不参与 loss 计算）。

**为什么取 turn 级平均而非 token 级**：token 级 ratio 乘积在长序列下方差极大（ratio$^{|t|}$ 易爆炸或下溢），turn 级平均等价于对对数概率比做算术平均，方差更小，clip 范围 $\varepsilon=0.02$ 相应更紧。

### 7.3 PPO Clipping 目标

对每个 valid turn $t$，PPO 裁剪损失为（注意 loss 取 **maximize** 形式，代码中用 `max(-adv * ratio, ...)` 实现）：

$$\mathcal{L}_{\text{PG}}^t = \max\!\left(-\hat{A}_i \cdot \text{ratio}_t,\ -\hat{A}_i \cdot \text{clamp}\!\left(\text{ratio}_t,\ 1-\varepsilon,\ 1+\varepsilon\right)\right), \qquad \varepsilon = 0.02$$

等价的标准 PPO 写法（minimize）：

$$\mathcal{L}_{\text{PG}}^t = -\min\!\left(\hat{A}_i \cdot \text{ratio}_t,\ \hat{A}_i \cdot \text{clamp}\!\left(\text{ratio}_t,\ 1-\varepsilon,\ 1+\varepsilon\right)\right)$$

### 7.4 KL 散度约束

**参考模型**：禁用所有 LoRA adapter 后的基础模型权重（无需额外存储，zero-overhead reference model）：

$$\log \pi_{\text{ref}}(a_k^t \mid s^t) = \log \pi_{\text{base}}(a_k^t \mid s^t)\Big|_{\text{all LoRA disabled}}$$

每个 turn 的 KL 散度估计（前向 KL 近似）：

$$\mathcal{L}_{\text{KL}}^t = \frac{1}{|t|}\sum_{k=1}^{|t|} \Bigl[\log \pi_\theta(a_k^t \mid s^t) - \log \pi_{\text{ref}}(a_k^t \mid s^t)\Bigr]$$

### 7.5 总损失函数

对 episode $i$ 中所有 $N_{\text{valid}}$ 个有效 turn 求平均：

$$\mathcal{L}_i = \frac{1}{N_{\text{valid}}} \sum_{t=1}^{N_{\text{valid}}} \left(\mathcal{L}_{\text{PG}}^t + \beta \cdot \mathcal{L}_{\text{KL}}^t\right)$$

batch 总 loss（跨 $B \cdot G$ 个 episode 累加，每 episode 独立 backward）：

$$\mathcal{L}_{\text{batch}} = \frac{1}{B \cdot G} \sum_{i=1}^{B \cdot G} \mathcal{L}_i$$

### 7.6 自适应 KL 系数控制

$\beta$ 根据实测 KL 与目标 KL 的偏差动态调整（类 PID 控制）：

$$\beta \leftarrow \beta \cdot \left(1 + \text{clip}\!\left(\frac{\text{KL}_{\text{cur}}}{\text{KL}_{\text{target}}} - 1,\ -0.2,\ 0.2\right) \cdot \frac{n}{H}\right)$$

参数：$\beta_0 = 0.1$，$\text{KL}_{\text{target}} = 6.0$，Horizon $H = 10000$，$n$ 为当前步数。

**直觉**：KL 超目标时增大 $\beta$（加强约束），KL 低于目标时减小 $\beta$（放松约束），变化幅度被 clip 限制在 $[-0.2, 0.2] \times n/H$ 以保证稳定性。

---

## 8. 模型与参数配置

### 8.1 模型架构

```
基础模型: Qwen3-8B
精度:     bfloat16
设备:     cuda:0（训练）/ cuda:1（推理）
```

**LoRA 配置**：

| 参数 | 值 |
|------|-----|
| rank $r$ | 16 |
| alpha $\alpha$ | 32 |
| scaling $s = \alpha/r$ | 2.0 |
| 目标模块 | `q_proj`, `v_proj` |
| Adapter 数量 | 4（controller/proposer/critic/verifier） |
| 参考模型 | 同一模型，禁用所有 adapter |

**优化器**：AdamW（8-bit 量化，bitsandbytes `AdamW8bit`），减少优化器状态显存占用约 75%。

### 8.2 训练超参数

| 超参数 | 值 | 说明 |
|--------|-----|------|
| `epochs` | 3 | 训练 epoch 数 |
| `batch_size` | 4 | 每步处理的问题数（代码实测）|
| `n_samples` | 4 | 每题采样轨迹数（代码实测）|
| `lr` | $1\times10^{-6}$ | AdamW 学习率 |
| `clip_epsilon` $\varepsilon$ | 0.02 | PPO ratio clip 范围 |
| `max_tokens` | 512 | 每次 LLM 生成最大 token 数 |
| `max_rounds` | 4 | 每 episode 最大协作轮数 |
| `max_interactions` | 3 | 每轮最大链式交互次数 |
| `kl_coef` $\beta_0$ | 0.1 | KL 系数初始值 |
| `target_kl` | 6.0 | 目标 KL 散度 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `ppo_epochs` | 1 | 每批数据的 PPO 更新次数 |

---

## 9. 推理引擎与训练/推理解耦

### 9.1 双 GPU 架构

```
GPU 0: PyTorch 训练进程
       - 持有 PeftModel（base + 4 LoRA adapters）
       - 执行前向/反向传播
       - 仅做训练态 forward（计算 new_lps / ref_lps）

GPU 1: vLLM 推理子进程
       - 执行 rollout 阶段的高吞吐量生成
       - 返回 (生成文本, log_probs, token_ids)
       - 与父进程通过 TCP socket JSON-RPC 通信
```

**权重同步**：每次 `optimizer.step()` 后，将 LoRA adapter 权重序列化保存到临时目录，通过 RPC 通知 vLLM worker 重新加载（`vllm_engine.sync_lora(model)`）。

### 9.2 并发 Rollout

使用 `ThreadPoolExecutor(max_workers=4)` 并发提交 `n_samples` 个 episode，利用 vLLM 的批处理能力提升吞吐量。

---

## 10. 内存优化策略

| 策略 | 实现 | 效果 |
|------|------|------|
| 逐 episode backward | `(loss / N).backward()` 后立即 `del loss` | 避免多 episode 计算图同时驻留 |
| 每 episode 清理 | `torch.cuda.empty_cache()` | 释放 CUDA 内存碎片 |
| Gradient Checkpointing | `model.gradient_checkpointing_enable()` | 用重计算换显存 |
| 8-bit Optimizer | `bnb.optim.AdamW8bit` | 优化器状态减少 75% |
| Reference Model | 禁用 LoRA（无额外模型副本） | 节省一份 8B 模型显存 |

---

## 11. 数值稳定性保护

代码中存在多层 NaN/Inf 防护：

1. **模型权重检查**：每次 `_compute_loss` 前检测 LoRA 参数是否含 NaN/Inf，异常则跳过整个 episode
2. **old_lp 清洗**：`torch.nan_to_num(old_lps, nan=0.0, posinf=0.0, neginf=0.0)`
3. **log_ratio 阈值**：`|ρ̄_t| > 50` 则跳过该 turn
4. **log_ratio 截断**：`clamp(ρ̄_t, -10, 10)` 防止 exp 溢出
5. **new_lps / ref_lps 有限性检查**：`torch.isfinite().all()` 失败则跳过
6. **KL 有限性检查**：`torch.isfinite(kl)` 失败则跳过
7. **零方差跳过**：`σ_r < 10^{-6}` 时跳过整批更新

---

## 12. 两阶段训练 Pipeline

### 阶段一：SFT 预热（train_sft.py）

- **数据**：专家轨迹 episode 文件，提取 `(system, user, response)` 三元组
- **损失**：仅对 response tokens 计算 Cross-Entropy（prompt 部分 label 设为 `-100`）

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|R|}\sum_{k \in R} \log \pi_\theta(a_k \mid s)$$

- **设置**：lr=$2\times10^{-5}$，batch size=2，epochs=3，只训练 Proposer adapter
- **输出**：`checkpoints/sft/`

### 阶段二：GRPO 强化学习（train.py）

- 加载 SFT checkpoint 初始化 Proposer adapter
- Controller / Critic / Verifier 从随机初始化开始 RL 训练
- 使用 Section 7 中的完整 GRPO 算法

---

## 13. 评估指标

| 指标 | 公式 | 语义 |
|------|------|------|
| `accuracy` | $\frac{1}{N}\sum_i \mathbf{1}[\hat{a}_i = a_i^*]$ | 多数投票后最终正确率 |
| `proposer_accuracy` | Proposer 直接输出的正确率 | 核心推理能力，不依赖协作 |
| `avg_turns` | $\frac{1}{N}\sum_i \|\text{turns}_i\|$ | 平均协作复杂度 |
| `interaction_rate` | $\frac{1}{N}\sum_i \|\text{interactions}_i\|$ | 智能体间交互活跃度 |
| `critic_precision` | $\text{TP}/(\text{TP}+\text{FP})$ | Critic 标记的准确性 |
| `critic_flag_rate` | Critic 触发标记的比例 | Critic 活跃度 |
| `verifier_consistency` | Verifier 置信度判断与最终答案一致率 | Verifier 校准程度 |

**Critic TP/FP 定义**：
- TP：Critic 标记了错误 $\wedge$ Proposer 答案确实错误
- FP：Critic 标记了错误 $\wedge$ Proposer 答案实际正确

---

## 14. 数据集

| 集合 | 文件 | 用途 |
|------|------|------|
| GSM8K train | `data/gsm8k_train.jsonl` | RL 训练（默认配置）|
| GSM8K test | `data/gsm8k_test.jsonl` | 评估 |
| MATH train | `data/math_train_rl.jsonl` | 可选训练集 |
| MATH test | `data/math_test.jsonl` | 可选评估集 |

数据格式：`{"question": "...", "answer": "..."}`

---

## 15. 核心设计贡献点（对比已有工作）

| 贡献 | 本工作 | 对比 |
|------|--------|------|
| **参数高效多角色** | 单模型 + 4×LoRA | 多数工作使用多个独立模型 |
| **黑板全局通信** | 结构化共享记忆，支持异步读写 | 点对点消息传递难以扩展 |
| **Turn 级 PPO** | 每个 LLM call 独立计算 ratio | Token 级高方差，Episode 级信号稀疏 |
| **GRPO 组内归一化** | 无需 Critic/价值网络 | PPO 需额外价值网络，增加不稳定性 |
| **零成本参考模型** | 禁用 LoRA = 参考模型 | 传统方法需额外存储完整参考模型副本 |
| **Controller 元认知** | 显式学习"何时停止" | 多数 MARL 框架缺乏显式终止策略 |
