# RACA：Role-Aware Credit Assignment

> 本文件是当前实验的算法纲领，所有代码修改均以此为准。
> 最后更新：2026-08-11
>
> **版本说明**：本文档包含两个版本。下文首先是 v1 原文（保留不动，对应已完成的
> ex6/ex8 实验），文末为 **RACA v2**（当前生效的纲领，所有新代码修改以 v2 为准）。

---

# 第一版（v1，原文保留）

---

## 系统设定

**模型架构**：冻结的 base LLM（Qwen3-8B）搭载 4 个完全独立的 LoRA adapter，
每个角色 `k ∈ {ctrl, prop, crit, verif}` 对应独立可训练参数 θ_k。
base 权重各角色共享但保持冻结，4 套 LoRA 权重彼此完全隔离，梯度仅在对应角色的
turn 上累积。

**符号约定**

| 符号 | 含义 |
|------|------|
| q, a* | 问题和 ground truth 答案 |
| N | 每题 rollout 数（默认 16） |
| T_max | 最大轮次（默认 4） |
| i | rollout 索引，i = 1..N |
| t | 轮次索引，t = 1..T_max |
| p^i_t | 1[proposer 第 t 轮答案 == a*] |
| f^i_t | 1["无错误" ∉ critic 输出] |
| v^i_t | verifier 置信分数 ∈ [0, 1] |
| t^i_stop | rollout i 最后实际执行的轮次 |
| c^i | 1[final answer == a*] |

---

## Phase 1：Rollout 收集

对同一道题 q 用当前策略并行生成 N 个完整 episode，记录上表所有统计量。
所有信用分配计算均在 rollout 完成后**离线**进行，无需额外前向传播。

---

## Phase 2：角色专属奖励

> **设计原则**：每个角色的奖励只衡量其职责范围内的行为质量，信号来源不交叉、不叠加。

### 2.1 Proposer（逐轮可验证奖励）

```
r_prop[i, t] = p^i_t
```

每轮答对得 1，答错得 0。所有轮次均有信号，消除原设计中"只有最后一轮才有奖励"的稀疏问题。

---

### 2.2 Critic（因果干预奖励）

按四格矩阵定义，将干预后**下一轮** proposer 的表现 p^i_{t+1} 内嵌入奖励，
捕捉 critic 行为的直接因果效果，而非 episode 最终结局的间接关联。

```
         flagged (f=1)                     silent (f=0)
p_t=0    真阳性: 0.3*p_{t+1} + 0.1*(1-p_{t+1})    漏检: 0
p_t=1    假阳性: -0.2                              真阴性: +0.1
```

末轮 t = t^i_stop 时以 c^i 代替 p^i_{t+1}。

**论文辩护**：critic 奖励与 p^i_{t+1} 绑定，是为了训练"可通信性（Communicability）"。
数学上正确但 proposer 无法理解的反馈对系统无效；这一设计强制 critic 生成 actionable 的反馈。

---

### 2.3 Verifier（校准奖励）

```
r_verif[i, t] = 1 - |v^i_t - p^i_t|
```

verifier 分数与该轮实际对错完全对齐得 1，完全相反得 0。

---

### 2.4 Controller（效率奖励）

仅赋给 rollout i 的**最后一个 controller turn**，拆分为三项：

```
r_ctrl[i] = c^i                                        # 答对底薪
           + α · c^i · (T_max - t^i_stop) / T_max     # 效率提成（答对才有）
           - β · (1 - c^i)                             # 答错惩罚
```

推荐初始值：α = 0.3，β = 0.2。

| 情形 | 奖励值 |
|------|--------|
| 第1轮答对 | 1 + α·(T_max-1)/T_max ≈ 1.23 |
| 最后一轮答对 | 1 + 0 = 1.0 |
| 任意轮答错 | -β = -0.2 |

梯度差 ≥ 1.2，确保"求对"始终优先于"求快"，消除原公式的 dead zone 问题。

---

## Phase 3：两层优势计算（角色路由）

> **核心路由原则**：Controller 负责 episode 全局策略 → 使用 Layer 1（episode 级）；
> Proposer / Critic / Verifier 职责局部化 → 使用 Layer 2（step 级）。
> 不做硬相加，按角色职责分配信号来源。

### 3.1 Layer 1：Episode 级优势（仅 Controller）

在 N 个 rollout 内计算：

```
μ_ctrl = mean({r_ctrl[i]}, i=1..N)
σ_ctrl = std ({r_ctrl[i]}, i=1..N)

A_E_ctrl[i] = (r_ctrl[i] - μ_ctrl) / max(σ_ctrl, δ)
```

δ = 1e-4 为方差下界，低方差时保留弱梯度而非跳过。

---

### 3.2 Layer 2：Step 级优势（Proposer / Critic / Verifier）

**Anchor state 定义**：`s̃_{k,σ} = (role=k, strategy=σ)`

其中 σ ∈ {explore, refine, verify} 是 controller 在该轮输出的策略标签。

**为什么用 strategy 而不是 round：**
- `round=t` 是时间位置，到第 t≥2 轮时不同 rollout 的黑板状态已经分叉（有的经历过 critic 干预，有的没有），组内不再是控制变量比较。
- `strategy=σ` 是认知任务上下文：controller 输出相同策略的轮次，意味着系统处于相似的推理状态（例如，所有 strategy=refine 下的 proposer，都是在 critic 发现错误后被要求改进的）。组内成员面对可比的情境，奖励差异更能反映动作质量。

对每个角色 k ∈ {prop, crit, verif} 和策略 σ，构建锚定组：

```
G_S(k, σ) = {(i, r_k[i,t]) | rollout i 中角色 k 在 strategy=σ 的轮次被调用}
```

组内归一化：

```
μ_{k,σ} = mean over G_S(k, σ)
σ_{k,σ} = std  over G_S(k, σ)

A_S[k, i, σ] = (r_k[i,t] - μ_{k,σ}) / max(σ_{k,σ}, δ)
```

当 |G_S| < 2 时跳过该 anchor（没有足够的对比样本）。

---

### 3.3 最终优势路由

```
A_total[ctrl, i, t] = A_E_ctrl[i]       # controller 所有 turn 用同一个 episode 值
A_total[k,   i, t] = A_S[k, i, σ_t]    # k ∈ {prop, crit, verif}，σ_t 为该轮 strategy
```

---

## Phase 4：策略更新

对每个 turn (k, i, t)，切换到对应 LoRA adapter θ_k，计算 PPO clip loss：

```
ρ_θ = π_{θ_k}(a^i_t | s^i_t) / π_{θ_k_old}(a^i_t | s^i_t)

L(k, t) = max(
    -A_total[k,i,t] · ρ_θ,
    -A_total[k,i,t] · clip(ρ_θ, 1 ± clip_ε)
)
```

梯度在全部有效 turn 上累积后除以 N_valid（总有效 turn 数），一次 optimizer step。
4 个 adapter 的梯度因 set_adapter() 切换天然隔离，无需额外 mask。

---

## 完整伪代码

```
Algorithm RACA
─────────────────────────────────────────────────────────────────
Input : 问题集 {(q_j, a_j*)},
        4 个独立 LoRA adapter {θ_ctrl, θ_prop, θ_crit, θ_verif},
        冻结 base model
Params: N=16, T_max=4, clip_ε=0.2, α=0.3, β=0.2, δ=1e-4
─────────────────────────────────────────────────────────────────
For each training step:

  ▌ Phase 1: Rollout
  Sample batch of questions {q, a*}
  For each q:
    Run N episodes via vLLM → collect {p,f,v,t_stop,c} for all i,t

  ▌ Phase 2: Role-specific rewards
  For i=1..N, t=1..T_max:
    r_prop[i,t]  ← p^i_t
    r_crit[i,t]  ← causal_reward(f^i_t, p^i_t, p^i_{t+1})
    r_verif[i,t] ← 1 - |v^i_t - p^i_t|
  For i=1..N:
    r_ctrl[i] ← c^i + α·c^i·(T_max - t^i_stop)/T_max - β·(1-c^i)

  ▌ Phase 3a: Layer 1 — Episode advantage (Controller only)
  μ_ctrl, σ_ctrl ← mean/std({r_ctrl[i]})
  For i=1..N:
    A_E_ctrl[i] ← (r_ctrl[i] - μ_ctrl) / max(σ_ctrl, δ)

  ▌ Phase 3b: Layer 2 — Step advantage (Prop / Crit / Verif)
  For k in {prop, crit, verif}:
    For t in 1..T_max:
      G ← {(i, r_k[i,t]) | rollout i has turn (k,t)}
      if |G| < 2: skip
      μ_{k,t}, σ_{k,t} ← mean/std over G
      For (i, _) in G:
        A_S[k,i,t] ← (r_k[i,t] - μ_{k,t}) / max(σ_{k,t}, δ)

  ▌ Phase 3c: Advantage routing
  A_total[ctrl,i,*] ← A_E_ctrl[i]          # controller 用 episode 级
  A_total[k,i,t]    ← A_S[k,i,t]           # 其余角色用 step 级

  ▌ Phase 4: Update
  N_valid ← count of all (k,i,t) with valid A_total
  optimizer.zero_grad()
  For each turn (k, i, t):
    set_adapter(θ_k)
    compute ρ_θ(k,t)
    loss ← clip_loss(A_total[k,i,t], ρ_θ) / N_valid
    loss.backward()
  clip_grad_norm(all θ_k, max_norm=1.0)
  optimizer.step()
  vllm.sync_lora()
─────────────────────────────────────────────────────────────────
```

---

## 与现有方法的对比

| 组件 | 来源 | RACA 的改造 |
|------|------|------------|
| per-agent 归一化 | Dr. MAS (2026) | R^i_k 定义本身按角色职责重设，非共享 episode reward |
| Anchor 分组 | GiGPO (2025) | 用 (role, round) 结构锚点替代环境状态哈希 |
| Step 级奖励 | GiGPO 用折扣回报 | 用数学验证直接结果 p^i_t 替代，无偏 |
| Critic 奖励 | 无直接对应 | 因果干预四格矩阵，内嵌 p^i_{t+1} |
| Controller 奖励 | 无直接对应 | 底薪 + 效率提成分离，消除 dead zone |
| 优势路由 | 无直接对应 | 按时间视野路由：全局角色用 A_E，局部角色用 A_S |

**核心贡献一句话**：在可验证结果的结构化多角色协作推理中，用任务内生的语义结构
（角色-轮次）作为 credit 分解的自然锚点，结合精确的逐轮验证信号，无需额外模型
调用或环境重执行即可实现精确的两层信用分配。

---

## 消融实验设计

```
Baseline : 原代码（全局 GRPO，episode-level reward sum）
+ Dr.MAS  : 仅加 per-role episode normalization
+ Reward  : 加角色专属 reward 定义（Phase 2）
+ RACA    : 加 Layer 2 step-level anchor（完整 RACA）
```

每组在相同步数（200 steps）下比较 eval_acc 曲线和最终精度。

---

## 待实现的代码改动清单

- [ ] `agentic_executor.py`：重写 reward 计算（Phase 2，四个角色全部）
- [ ] `agentic_executor.py`：记录 per-round `prop_correct` 供 critic reward 使用
- [ ] `training/grpo_trainer.py`：实现 Phase 3 两层优势（Layer 1 + Layer 2 + 路由）
- [ ] `training/grpo_trainer.py`：`_compute_loss` 接收 per-turn advantage 而非单一 episode advantage
- [ ] `configs/agentic/default.yaml`：新增 α, β, δ 超参数













-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
来源：
RACA：Role-Aware Credit Assignment for Multi-Agent Reasoning

  ---
  1. 系统设定

  模型架构：系统使用一个冻结的 base LLM（Qwen3-8B），搭载4 个完全独立的 LoRA adapter，每个角色 $k \in \mathcal{K} = {\text{ctrl, prop, crit, verif}}$ 对应独立的可训练参数集 $\theta_k$。base 权重在所有角色间共享但保持冻结，LoRA 参数彼此完全隔离，梯度只在对应角色的 turn 上累积。

  交互结构：给定问题 $q$ 和 ground truth $a^*$，系统在一个共享黑板（Blackboard）上运行最多 $T_{max}$ 轮。每轮由 controller决策策略，再由 proposer、critic、verifier 按需交互。最终答案通过多数投票（majority vote）产生。

  训练设定：每道题采样 $N$ 个独立 rollout（当前 $N=16$），所有 rollout 在同一道题的 group 内计算 GRPO 优势。

  ---
  2. Phase 1：Rollout 收集

  对同一道题 $q$ 生成 $N$ 个完整 episode，记录每个 rollout $i \in {1,\ldots,N}$、每轮 $t \in {1,\ldots,T_{max}}$ 的以下信息：

  $$p^i_t = \mathbf{1}\bigl[\text{norm}(\hat{a}^i_t) = \text{norm}(a^*)\bigr] \tag{proposer 第 $t$ 轮是否答对}$$

  $$f^i_t = \mathbf{1}\bigl[\text{"无错误"} \notin o^i_{crit,t}\bigr] \tag{critic 第 $t$ 轮是否 flag 错误}$$

  $$v^i_t \in [0,1] \tag{verifier 第 $t$ 轮的置信分数}$$

  $$t^i_{stop} = \text{最后实际执行的轮次}, \qquad c^i = \mathbf{1}[\text{final answer} = a^*]$$

  ---
  3. Phase 2：角色专属奖励设计

  这是RACA 区别于 Dr. MAS 的第一个关键点。Dr. MAS 对所有 agent 使用同一个 episode reward $R^i$，只改变归一化方式；RACA 按照每个角色的职责语义，为每个角色独立定义 $r^i_{k,t}$，再分别归一化。

  2.1 Proposer：逐轮可验证奖励

  $$r^i_{prop,t} = p^i_t$$

  每轮答对得1，答错得 0。不再只有最后一轮才有信号——每一轮的提案质量都被独立衡量。

  2.2 Critic：因果干预奖励（维度三，核心创新）

  按四格矩阵定义，奖励中内嵌了critic 行为对下一轮 proposer 的实际因果效果：

  $$r^i_{crit,t} = \begin{cases} 0.3 \cdot p^i_{t+1} + 0.1 \cdot (1 - p^i_{t+1}) & \text{if } f^i_t = 1 \wedge p^i_t = 0 \quad \text{（真阳性）} \[4pt] -0.2 & \text{if } f^i_t = 1 \wedge p^i_t = 1 \quad \text{（假阳性）} \[4pt] +0.1 & \text{if } f^i_t = 0 \wedge p^i_t = 1 \quad
  \text{（真阴性）} \[4pt] 0 & \text{if } f^i_t = 0 \wedge p^i_t = 0 \quad \text{（漏检）} \end{cases}$$

  其中末轮 $t = t^i_{stop}$ 时用 $c^i$ 代替 $p^i_{t+1}$。

  与原设计的本质区别：原代码用 is_correct（episode 最终结果）给critic 打分，存在幸存者偏差——critic 只在"整个 episode 最后答对了"的条件下才得正奖励。RACA 用 $p^i_{t+1}$（本critic 行为之后下一轮 proposer 的表现），测量的是 critic 干预的直接因果效果，而非最终结局的间接关联。

  2.3 Verifier：校准奖励

  $$r^i_{verif,t} = 1 - \left|v^i_t - p^i_t\right|$$

  当 verifier 分数与该轮 proposer 的实际对错完全对齐时得1，完全相反时得 0。

  2.4 Controller：效率奖励（维度四，核心创新）

  仅赋给 rollout $i$ 中最后一个 controller turn：

  $$r^i_{ctrl} = \alpha \cdot c^i \cdot \frac{T_{max} - t^i_{stop}}{T_{max}} - \beta \cdot (1 - c^i)$$

  答对且用轮次少→ 高奖励；答错 → 惩罚 $\beta$；答对但用满了轮次 → 接近 0。超参数建议 $\alpha=0.5,\ \beta=0.3$。

  2.5 Episode 级角色汇总奖励

  $$R^i_k = \sum_t r^i_{k,t}$$

  ---
  4. Phase 3：两层优势计算

  4.1 第一层——角色解耦 episode 优势（Dr. MAS 变体）

  对每个角色 $k$，在 $N$ 个 rollout 内独立计算均值和标准差：

  $$\mu_k = \frac{1}{N}\sum_{i=1}^N R^i_k, \qquad \sigma_k = \sqrt{\frac{1}{N}\sum_{i=1}^N (R^i_k - \mu_k)^2}$$

  $$\boxed{A^i_{E,k} = \frac{R^i_k - \mu_k}{\sigma_k + \varepsilon}}$$

  若 $\sigma_k < 10^{-6}$（zero variance），跳过该角色在该 batch 的更新。

  与Dr. MAS 的区别：Dr. MAS 各agent 的$R^i$ 是同一个 episode 终端奖励，只有归一化统计不同。RACA 的 $R^i_k$ 定义本身就因角色而异（proposer 是逐轮正确率之和，critic 是因果干预分之和，等等），信号来源根本不同。

  4.2 第二层——角色-轮次锚定步骤优势（GiGPO 变体）

  Anchor state定义（这是 RACA 对 GiGPO 的核心改造）：

  $$\tilde{s}_{k,t} \triangleq (\text{role} = k,\ \text{round} = t)$$

  GiGPO 的 anchor 是物理环境中字面相同的状态（同一网页、同一房间）。RACA 的 anchor 是任务结构中语义等价的位置——同一道题里，所有 rollout 的"第$t$ 轮 proposer"处于等价的功能状态，可以直接横向比较。

  对角色 $k$、轮次 $t$，构建锚定组：

  $$G_S(k, t) = \bigl{(i,\ r^i_{k,t}) ;\bigm|; i \in {1,\ldots,N},\ \text{rollout } i \text{ 中角色 } k \text{ 在第} t \text{ 轮被调用}\bigr}$$

  组内归一化：

  $$\mu_{k,t} = \frac{1}{|G_S|}\sum_{(i,\cdot)\in G_S} r^i_{k,t}, \qquad \sigma_{k,t} = \text{std}\bigl({r^i_{k,t}}_{G_S}\bigr)$$

  $$\boxed{A^i_{S,k,t} = \frac{r^i_{k,t} - \mu_{k,t}}{\sigma_{k,t} + \varepsilon}}$$

  与 GiGPO 的区别：GiGPO 用折扣回报 $R^i_t = \sum_{s=t}^T \gamma^{s-t} r_s$ 作为 anchor组内的比较量，这在稀疏奖励的Web 任务中是必要的近似。RACA 中每个角色有精确的 per-turn reward（$p^i_t$ 直接从数学验证获得），无需折扣近似，比较量更干净、无偏。

  4.3 合并优势

  $$\boxed{A^i_{total}(k, t) = A^i_{E,k} + \omega \cdot A^i_{S,k,t}}$$

  $\omega \ge 0$ 为步骤权重，初始设为 $\omega = 1.0$，消融实验中对 ${0, 0.5, 1.0, 2.0}$ 进行敏感性分析。

  ---
  5. Phase 4：策略更新

  对每个 turn $(i, t)$，角色 $k$，在切换到对应 LoRA adapter $\theta_k$ 后计算 PPO clip loss：

  $$\rho_\theta = \frac{\pi_{\theta_k}(a^i_t \mid s^i_t)}{\pi_{\theta_k^{old}}(a^i_t \mid s^i_t)}, \qquad L_{clip} = \max\Bigl(-A^i_{total} \cdot \rho_\theta,\ -A^i_{total} \cdot \text{clip}(\rho_\theta,1-\epsilon, 1+\epsilon)\Bigr)$$

  梯度在所有有效 turn 上累积后除以总有效 turn 数，一次optimizer step。由于各 adapter梯度天然隔离，4 个角色的 LoRA 权重在同一个 backward pass 中按各自的 $A^i_{total}(k,t)$ 分别更新。

  ---
  6. 算法伪代码

  Algorithm RACA
  ─────────────────────────────────
  Input :问题集 {(q_j, a_j*)}, 初始化好的4-adapter 模型 {θ_k}
  Params: N,ω, α, β, T_max, ε, clip_ε
  ─────────────────────────────────
  For each training step:// Phase 1: Rollout
    Sample batch of questions {q, a*}
    For each q: run N episodes → collect {p^i_t, f^i_t, v^i_t, t^i_stop, c^i}

    // Phase 2: Role rewards
    For i=1..N, t=1..T:
      r_prop[i,t] ← p^i_t
      r_crit[i,t] ← causal_credit(f^i_t, p^i_t, p^i_{t+1})  // Eq. 2.2
      r_verif[i,t]← 1 - |v^i_t - p^i_t|r_ctrl[i]← α·c^i·(T_max - t^i_stop)/T_max - β·(1-c^i)

    R_k[i]← Σ_t r_k[i,t]  for each k

    // Phase 3: Advantages
    For k in {ctrl, prop, crit, verif}:
      μ_k, σ_k  ← mean/std({R_k[i]})
      A_E[k,i]  ← (R_k[i] - μ_k) / (σ_k + ε)// Layer 1

    For k in {ctrl, prop, crit, verif}, t in 1..T_max:
      G_S(k,t)  ← {(i, r_k[i,t]) | turn(k,t) exists in rollout i}
      μ_{k,t}, σ_{k,t} ← mean/std(G_S)
      A_S[k,i,t]← (r_k[i,t] - μ_{k,t}) / (σ_{k,t} + ε)  // Layer 2

      A_total[k,i,t] ← A_E[k,i] + ω · A_S[k,i,t]    // Eq. 4.3

    // Phase 4: Update
    total_valid← count of all (k,i,t) triples
    For each turn (k, i, t):
      set_adapter(θ_k)
      computeρ_θ, clip loss with A_total[k,i,t]
      accumulate gradient / total_valid
    optimizer.step()
  ─────────────────────────────────

  ---
  7. 创新点汇总

  ┌─────────────────┬──────────────────┬────────────────────────────────────────┬────────────────────────────┐
  │      组件       │     直接来源     │              RACA 的改造               │          使能条件          │
  ├─────────────────┼──────────────────┼────────────────────────────────────────┼────────────────────────────┤
  │ Layer 1 归一化  │ Dr. MAS          │ $R^i_k$的定义本身按角色职责重新设计    │ 4个角色奖励语义不同        │
  ├─────────────────┼──────────────────┼────────────────────────────────────────┼────────────────────────────┤
  │ Layer 2 anchor  │ GiGPO            │ 用结构化(role, round) 替代环境状态哈希 │ 任务有固定角色-轮次结构    │
  ├─────────────────┼──────────────────┼────────────────────────────────────────┼────────────────────────────┤
  │ Layer 2奖励     │ GiGPO 用折扣回报 │ 用数学验证结果 $p^i_t$ 直接替代        │ 中间结果可验证（数学任务） │
  ├─────────────────┼──────────────────┼────────────────────────────────────────┼────────────────────────────┤
  │ Critic奖励      │ 无直接对应       │ 四格矩阵 + 因果效果内嵌                │ 逐轮 proposer 可验证       │
  ├─────────────────┼──────────────────┼────────────────────────────────────────┼────────────────────────────┤
  │ Controller 奖励 │ 无直接对应       │ 效率信用：正确性×轮次节省              │ 轮次可计数                 │
  └─────────────────┴──────────────────┴────────────────────────────────────────┴────────────────────────────┘

  核心论文贡献一句话：在可验证结果的结构化多角色协作推理中，用任务内生的语义结构（角色-轮次）作为 credit 分解的自然锚点，结合精确的逐轮验证信号，无需额外模型调用或环境重执行即可实现精确的两层信用分配。

---
---

# 第一版实施与修复记录（v1.x，后补留档）

> 本节为后补记录：v1 阶段在代码层面修复的关键问题。
> 这些修复均已在当前代码中生效，**v2 全部继承**。
>
> **两批修复，证据来源不同：**
>
> | 批次 | 内容 | 依据 |
> |------|------|------|
> | A（基础正确性） | Fix 3~6、9~12、14~16 | commit `3611725` 的 git diff |
> | B（行为诊断） | Fix 1、2、7、17 | commit `110fb86` + `train_20260807_062356.log`（200 步） |
>
> 每条 Fix 均已用 `git show <commit>^:<file>` 对比修改前后代码验证。
> **Fix 13 经核验为误记，已作废**（详见该条）。
>
> 批次 B 的三个观测事实（下文多处引用）：
> - `eval_avg_turns` 8.0 → 2.0 在 50 步内坍塌并永久锁死
> - `loss` 从 −15.7 衰减到 ≈0.0000（step 60 之后基本空转）
> - `eval_acc` 200 步仅 0.650 → 0.693（峰值 0.710@step180）
>
> ⚠️ **另有一项代码已修但数据未重新生成的遗留问题，见本节末尾「遗留问题」。**

### Fix 1：stop turn 零梯度（最严重）

- **症状**：controller 输出 `strategy: stop` 的 turn 从不进入 round_records，
  因此没有 reward 条目、没有 advantage、没有梯度——**controller 最关键的一个动作
  （终止决策）完全没被训练**；而省轮 bonus 却落在更早的 "continue" turn 上，
  credit 错位。更极端地，第 0 轮就 stop 的 episode 产生完全空的 turn data，
  整条 rollout 被从 batch 中丢弃。
- **修复**：记录 `stop_ctrl_tid`，episode 结局奖励优先赋给显式 stop 的那个 turn
  （耗尽 max_rounds 时回退到最后一个工作 turn）；stop turn 不在 round_records
  时为其单独创建条目（strategy="stop"）。
- **位置**：`agentic_executor.py` 的 `stop_ctrl_tids` 追踪与
  `_compute_raca_turn_data` 的 outcome_tid 逻辑。

### Fix 2：单边效率奖励导致 avg_turns 坍塌 → 引入 γ 对称惩罚

- **症状**：原公式省轮只有 bonus（答对才有），答错的成本与用了几轮无关，
  使得"立刻 stop"在任何置信水平下都是**弱占优**策略。日志实证：
  `eval_avg_turns` 8.0 → 2.0（50 步内坍塌并锁死），critic/verifier 的修订
  循环被完全绕过，多智能体管线退化成单次 proposer 直出。
- **数值证据**（α=0.3, β=0.2, T_max=4，修复前公式）：

  | p(correct) | E[r \| stop@1] | E[r \| stop@4] | 占优方 |
  |---|---|---|---|
  | 0.3 | +0.2275 | +0.1600 | 早停 |
  | 0.5 | +0.5125 | +0.4000 | 早停 |
  | 0.7 | +0.7975 | +0.6400 | 早停 |

  答对时早停多拿 +0.225，答错时代价恒为 −β 与轮次无关 → 早停在**所有**
  置信水平下都不劣。这不是调参问题，是奖励结构的定性缺陷。
- **修复**：新增 `ctrl_gamma`（默认 0.3）：省下的轮次在答对时是 bonus（α 项）、
  答错时是等强度惩罚（γ 项）。省轮项系数变为 `p·α − (1−p)·γ`，γ=α 时
  分界点落在 p=0.5——只有答案更可能对时提前停才划算。
  修复后实测：p=0.3 继续工作赢 / p=0.5 打平 / p=0.7 早停赢。
- **位置**：`agentic_executor.py` `_compute_raca_turn_data` 的 outcome 奖励计算。

**⚠️ 本文档 v1 正文 §2.4 的两处论断已被实验推翻**（原文按"保留不动"未改，在此更正）：

1. §2.4 表格"任意轮答错 → −β = −0.2"——正是这个"与轮次无关"使早停无成本。
2. §2.4 结尾"梯度差 ≥ 1.2，确保'求对'始终优先于'求快'，消除原公式的
   dead zone 问题"——**该结论不成立**。底薪项 c^i 确实消除了原始公式
   （见文末"来源"节 §2.4：`α·c·rem − β·(1−c)`，答对但用满轮次得 0）的
   dead zone，但"求对优先于求快"只保证了答对的**绝对**收益更高，
   并未阻止在**同等正确性下**无条件偏好求快。两者是不同的性质，
   混淆导致了坍塌。

**Controller 公式演化三阶段**（便于对照）：

```
① 原始（文末"来源"节 §2.4）  α·c·rem − β·(1−c)                    有 dead zone
② v1 正文 §2.4（加底薪）      c + α·c·rem − β·(1−c)                 无 dead zone，但早停弱占优
③ 修复后 / v2 §4.5（加 γ）    c + α·c·rem − β·(1−c) − γ·(1−c)·rem   早停仅在 p>0.5 时划算
```

### Fix 3：critic_flagged 几乎恒为 True → 鲁棒解析

- **症状**：旧判定逻辑是 `"无错误" not in output`。交互响应会覆盖
  `role_outputs["critic"]`，而响应文本不遵循 critic 标准格式、几乎不含
  "无错误"三字，导致 critic_flagged 几乎恒为 True，四格矩阵奖励的输入信号
  失真。
- **修复**：`_critic_found_errors` 改为专门解析"错误分析"小节；小节缺失
  （如交互响应）时保守地返回 False。
- **位置**：`agentic_executor.py` `_critic_found_errors`。
- **备注**：这是对症状的补丁；根因（响应覆盖主输出）由 v2 的
  primary/responses 双槽存储根治。

### Fix 4：controller turn 未进 round_records → 无 RACA 信号

- **症状**：round_turn_ids 只收集工作角色的 turn，controller 的 continue turn
  不在其中，拿不到奖励条目。
- **修复**：每轮 round_turn_ids 以 `(ctrl_tid, "controller")` 初始化，
  continue turn 拿 0.0 占位奖励、共享 episode 级 advantage（Layer 1 路由）。
- **位置**：`agentic_executor.py` 两条执行路径（batch / 单 episode）均已处理。

### Fix 5：Layer 2 anchor 从 (role, round) 改为 (role, strategy)

- **症状**：初始设计（本文档末尾"来源"节 §4.2）用 round=t 做锚点，但 t≥2 时
  不同 rollout 的黑板状态已分叉（有无 critic 干预经历不同），组内不再是
  控制变量比较。
- **修复**：锚点改为 controller 输出的 strategy 标签（v1 正文 §3.2 已按修复后
  口径书写）。
- **备注**：v2 进一步把标签改为从黑板状态机械推导的 σ（§3），消除策略漂移。

### Fix 6：optimizer / grad clip 只收集激活 adapter → 非激活 adapter 梯度污染

- **症状**：PEFT `set_adapter()` 只把激活 adapter 的 requires_grad 设为 True。
  若仅收集 requires_grad=True 的参数，非激活 adapter 既不被
  `optimizer.step()` 更新、也不被 `zero_grad()` 清零，梯度跨 step 持续累加。
- **修复**：optimizer 构造与 `clip_grad_norm_` 均改为收集全部 4 个 adapter 的
  `lora_parameters()`，不依赖 requires_grad。
- **位置**：`grpo_trainer.py` `__init__` 与 `update`。

### Fix 7：零方差组发零优势 → 直接丢弃该层/该组

- **症状**：全对/全错的组若用方差下界发零优势，仍要为每个 turn 跑
  forward+backward，且膨胀 total_valid，稀释有真实信号的组的梯度。
- **修复**：Layer 1 零方差时整层丢弃（ctrl_adv=None）；Layer 2 anchor 组
  σ≤δ 或 |G|<2 时跳过该组；无任何有效 advantage 的 episode 从 batch 剔除。
- **位置**：`grpo_trainer.py` `_compute_raca_advantages` / `update`。

### Fix 8：数值稳定性防护（NaN/Inf 与 log_ratio 飞车）

- **症状**：训练中出现过 NaN loss（见 diagnose_nan.py 系列诊断脚本）与
  极端 importance ratio 导致的梯度爆炸。
- **修复**：① 每 episode 更新前检查全部 LoRA 权重有限性，NaN/Inf 则跳过；
  ② rollout 旧 log_probs 过 `nan_to_num`；③ |mean log_ratio| > 50 的 turn
  直接跳过；④ new_lps 非有限则跳过该 turn；⑤ 全局 grad clip（max_norm=1.0）。
- **位置**：`grpo_trainer.py` `_compute_loss`。

### Fix 9：SFT checkpoint 路径不匹配 → 静默跳过（实际用随机 LoRA 训练）

- **症状**：实际 checkpoint 目录结构是嵌套的 `{sft}/{role}/{role}/`，
  而加载代码只找扁平路径，找不到时**静默跳过**。后果：4 个 adapter 全是
  随机初始化的 LoRA，RL 训练从头开始而非从 SFT 起点——但日志里看不出异常。
- **修复**按顺序搜索嵌套 `{sft}/{role}/{role}/` → 扁平 `{sft}/{role}/` →
  顶层 `{sft}/`；三者均不存在时 `raise FileNotFoundError` 而非跳过。
  成功时打印每个 adapter 的已加载参数数（现为 144）作为可验证凭据。
- **位置**：`llm/trainable_llm.py` `load_trainable_models`。

### Fix 10：`extract_math_answer` 嵌套花括号截断（污染 ground truth）

- **症状**：旧正则 `r"\\boxed\{([^}]+)\}"` 遇到 `\boxed{\frac{1}{2}}` 只能捕到
  `\frac{1`——`[^}]` 在第一个 `}` 就停了。ground truth 被截断，
  **模型答对也会被判错**。
- **实测影响**（统计 answer 字段花括号不平衡的比例）：

  | 数据文件 | 被截断 | 占比 |
  |---|---|---|
  | `math_train_rl.jsonl`（训练集） | 1303/5185 | **25.1%** |
  | `math_test_clean.jsonl`（测试集） | 942/3663 | **25.7%** |
  | eval 子集（Level5 前 300） | 61/300 | **20.3%** |

  典型样例：`'\\frac{1'`、`'\\dfrac{9'`、`'(-\\sqrt{3'`、`'\\frac{2\\sqrt{53'`。
- **修复**：改为花括号深度匹配解析器（depth 计数），不平衡时返回空串。
- **位置**：`data/prepare_data.py` `extract_math_answer`。
- **⚠️ 代码已修但数据未重新生成，见本节末尾「遗留问题」。**

### Fix 11：`normalize_answer` 有损比较 → `math_equal`

- **症状**：旧比较先过 `normalize_answer`（剔除所有非数字字符），
  `\frac{1}{2}` 变成 `"12"`，与整数 `12` **假阳性匹配**。该比较函数是
  p^i_t / c^i 的唯一来源，因此污染的是**所有四个角色奖励的输入信号**。
- **修复**：新增 `math_equal()`：先试数值比较（支持 `\frac{a}{b}`、`a/b`、
  小数、`\text{...}` 包裹），仅在无法提取数值时才回退到字符串归一化。
  `normalize_answer` 保留但加上有损警告注释。
- **位置**：`agents/agentic_executor.py` `_extract_number` / `math_equal`；
  调用方 executor（3 处）与 `evaluate.py` 均已替换。

### Fix 12：eval 用采样解码（temperature 硬编码 1.0）

- **症状**：temperature 在 worker 层硬编码为 1.0，eval 与训练用同一套采样参数。
  eval_acc 带量化噪声、不可复现，跨 step 比较的差异可能只是采样注入。
- **修复**：`AgenticExecutor` 新增 `eval_mode` 参数，
  `self.temperature = 0.0 if eval_mode else 1.0`；temperature 逐层透传
  executor → `vllm_engine` → `vllm_worker` 的 `SamplingParams`。
  `train.py` / `evaluate.py` / `evaluate_baseline.py` 的 eval 路径均传
  `eval_mode=True`（贪心解码）。
- **位置**：`agentic_executor.py`、`llm/vllm_engine.py`、`llm/vllm_worker.py`。

### Fix 13（作废）

此编号原记为"`MultiVLLMEngine.sync_lora` 并发竞态"，**经 git 核验为误记，已作废**。

实际情况：`sync_lora` 从来就是串行的（`git show 3611725^` 确认）：

```python
def sync_lora(self, model):          # 修改前，已是串行
    for eng in self.engines:
        eng.sync_lora(model)
    self._lora_loaded = self.engines[0]._lora_loaded
```

那个 `ThreadPoolExecutor` 属于同文件的 `generate_batch`，与 `sync_lora` 无关。
commit 3611725 对该函数**只添加了解释性 docstring**（说明为何不该改成并发），
未修正任何缺陷。编号保留空位以免打乱下文引用。

### Fix 14：KL 项恒为 0（根本没有参考前向）

- **症状**：日志里 `kl` 一直是 0.0——不是策略没漂，而是代码里压根没有算
  reference log-probs，KL 惩罚项不存在。策略可以无限远离 base model（熵崩塌）。
- **修复**：在 `_compute_loss` 里用 `model.as_ref()`（`disable_adapter_layers()`）
  + `torch.no_grad()` 做一次 base model 前向，
  `kl = mean(new_logprob − ref_logprob)`，
  `turn_loss = pg_loss + kl_coef · kl`；新增 `kl_coef`（默认 0.04）。
  代价：每 turn 多一次前向。
- **位置**：`grpo_trainer.py` `_compute_loss`；`configs/agentic/default.yaml`。
- **备注**：修复后日志中 kl 从 0.032 漂到 0.167（200 步）。大部分漂移发生在
  轮次坍塌期（step 30～60）；Fix 1/2 生效后 KL 压力应自然减小。
  若仍偏高可考虑 kl_coef → 0.1 或自适应 KL。
- **v2.1 更新**：参考模型从裸 base 改为冻结的 SFT 快照（见 §14.4），
  本条的 `disable_adapter_layers()` 实现仅作为无 ref adapter 时的回退路径保留。

### Fix 15：LoRA adapter dtype 不统一（F32 混入 bf16）

- **症状**：`get_peft_model` 的 `autocast_adapter_dtype` 只作用于首个 adapter
  （proposer），其余三个由 `add_adapter` 创建的 adapter 可能留在 F32，
  与 bf16 base 混算。
- **修复**：`add_adapter` 后统一 cast：取 `base_dtype = next(base.parameters()).dtype`，
  将所有 `lora_` 参数 `.to(base_dtype)`。
- **位置**：`llm/trainable_llm.py` `load_trainable_models`。

### Fix 16：checkpoint 加载按 requires_grad 过滤 → 只载入 1/4 adapter

- **症状**：与 Fix 6 同根因（`set_adapter()` 只让激活 adapter `requires_grad=True`），
  但发生在**权重加载**路径：resume 与 eval 用 `param.requires_grad` 筛选要
  `copy_` 的参数，导致只有当前激活的那一个 adapter 被真正载入，
  其余三个静默保持随机初始值。
- **修复**：筛选条件改为 `"lora_" in name`，不依赖 requires_grad。
- **位置**：`train.py` resume 分支；`evaluate.py` checkpoint 加载。
- **备注**：另修 `evaluate_baseline.py` 的返回值解包错误
  （`models, _, tokenizer = ...` → `model, tokenizer = ...`，
  `load_trainable_models` 只返回 2 个值），否则 baseline 评测直接报错。

### Fix 17：无梯度的空步消耗 step 预算

- **症状**：`step += 1` 无条件执行。Fix 7 丢弃退化组后，整批全退化时
  本步什么都没学，但照样计入 `max_steps=200`——200 步里可能有一部分是空转。
  另一条路径（无组满足 `len(valid)>=2`）虽不计步但完全静默。
- **修复**：两条无梯度路径统一走 `note_skip()` 记账，`step` 不递增；
  累计 `skipped_batches` 并在下次真实 step 写入 wandb。
  **因移除了保证终止的无条件递增**，新增连续跳过上限
  `max_consecutive_skips=50`，超限 `raise` 而非静默 break
  （避免保存坏 checkpoint 却看似训练成功）。
  skipped 分支不再 `wandb.log`，避免向已写过的 step 重复写入。
- **位置**：`train.py` 训练循环；`configs/agentic/default.yaml`。
- **副作用**：`max_steps` 现在表示**200 次真实更新**而非 200 次取批尝试，
  退化率高时 wall-clock 会变长，不可与旧实验直接比耗时。

### Fix 18：`math_equal` "取最后一个数字"兜底 → 分层严格判等（grader.py）

- **症状**：`_extract_number` 的最后兜底分支会从任意字符串里抽"最后一个数字"：
  真值 `2\sqrt{3}` vs 回答 `3` → 都压成 3.0 → **假阳性**；
  真值 `\frac{\pi}{2}` vs 回答 `1.57` → 抽成 2.0 vs 1.57 → **假阴性**。
  实测：在 math_test 的 1365 个符号型答案上，"只答符号答案里最后一个数字"
  的作弊策略旧判分器接受率 **66.2%**。eval 子集是 Level 5（符号答案占比最高，
  全测试集 38% 为符号型），失真恰好集中在区分度最关键的题上；
  且同一把歪尺子同时喂 `p_t`（proposer 奖励）、critic 因果奖励、`is_correct`
  和 eval_acc，奖励地形与评测同源污染。
- **修复**：新建 `agents/grader.py`（方案 A2，零第三方依赖）：
  1. 移植 verl/Hendrycks MATH 官方 `strip_string`/`is_equiv` 归一化
     （去 `\left\right`、dfrac/tfrac 统一、sqrt/frac 简写展开、去单位/百分号）；
  2. 数值比较改为 **fullmatch 严格解析**（整个字符串是数/分数才走数值路径，
     支持千分位、`\text{}` 包裹、负号在 `\frac` 外）；
  3. **删除"最后一个数字"兜底**：无法证明等价一律 False；
  4. 判等前轻量清洗：`\boxed{}` 提取、markdown 加粗、`$`、尾部标点、`x \in` 前缀。
  配套 `test_grader.py`（42 条回归用例 + 对称性检查，用例取自真实测试集答案形态）。
- **验收**：8934 条真实答案自反性 100% 通过；作弊策略接受率 66.2% → **8.5%**
  （残余部分主要是合法等价：`118 \text{ dollars}` vs `118` 这类去单位后确实相等）。
- **位置**：`agents/grader.py`（新增）；`agents/agentic_executor.py`（旧三函数删除，
  改为 `from agents.grader import math_equal`）；`train.py`（移除失效 import）。
- **副作用（重要）**：判分器变严后 **eval_acc 与旧实验不可比**（旧数字系统性偏高），
  奖励地形同步改变——这是继 Fix 10 数据重生之后的又一次断代，应与 v2 开跑
  对齐；如需折算历史实验，用新 grader 重跑旧 checkpoint 的 eval。
  纯符号 vs 小数的跨形式等价（`\frac{\pi}{2}` vs `1.5708`）仍判 False，
  属有意为之的保守选择（宁漏判不猜测）。

### 可观测性新增（配合 Fix 7/17）

| 输出 | 位置 | 用途 |
|------|------|------|
| `groups=kept/total` | stdout 每步 | 退化组比例 |
| `[skip] ...` | stdout | 无梯度批次及原因 |
| `group_keep_rate` | wandb | 持续低于 0.4 则需 DAPO 动态采样 |
| `skipped_batches` | wandb | 累计丢弃批次 |

### 小结

按影响层次归类（而非发现顺序）：

| 层次 | # | 问题 | 影响面 | v2 后续动作 |
|------|---|------|--------|----------------|
| **数据/监督信号** | 10 | `\boxed{}` 嵌套截断 | 25% ground truth 错误 | 继承（⚠️ 数据待重生） |
| | 11 | `normalize_answer` 有损比较 | 所有角色奖励的输入信号 | 继承 |
| | 18 | `math_equal` 数字兜底假阳性 | 符号型答案奖励/评测双向失真 | 继承（grader.py，⚠️ eval_acc 断代） |
| | 3 | critic_flagged 恒为 True | critic 四格矩阵输入失真 | 继承；根因由 v2 双槽存储根治 |
| **权重加载** | 9 | SFT 路径静默跳过 | 实际从随机 LoRA 开始训练 | 继承 |
| | 16 | 加载按 requires_grad 过滤 | 只载入 1/4 adapter | 继承 |
| | 15 | adapter dtype 不统一 | F32/bf16 混算 | 继承 |
| **训练正确性** | 6 | optimizer 只收集激活 adapter | 非激活 adapter 梯度跳 step 累加 | 继承 |
| | 14 | KL 项恒为 0 | 策略无约束漂移、熵崩塌风险 | 继承 |
| | 8 | NaN / log_ratio 飞车 | 训练中断或梯度爆炸 | 继承 |
| **credit 分配** | 1 | stop turn 零梯度 | controller 终止决策不可训练 | 继承；v2 中 stop 更成为其唯一职责 |
| | 4 | controller turn 未进 round_records | controller 无 Layer 1 信号 | 继承 |
| | 2 | 单边省轮奖励 | 早停弱占优、avg_turns 坍塌 | 继承（γ 项入 v2 公式） |
| | 5 | anchor 用 round | 组内不可比 | v2 升级为机械 σ |
| **梯度质量/流程** | 7 | 零方差组发零优势 | 稀释有效组梯度（loss→0） | 继承 |
| | 17 | 空步消耗 step 预算 | 200 步中含空转步 | 继承 |
| **评估可信度** | 12 | eval 用采样解码 | eval_acc 带噪、不可复现 | 继承 |

---

## ⚠️ 遗留问题：代码已修，数据未重新生成

> **状态：已解决（2026-08-11，处理过程见本节末尾）。**

Fix 10 修了 `extract_math_answer` 的代码，**但数据文件是修复前生成的，
从未重跑**。时间戳证据：

```
data/math_test_clean.jsonl   Aug 7 10:39:13   ← 数据生成
data/math_train_rl.jsonl     Aug 7 10:39:14   ← 数据生成
data/prepare_data.py         Aug 7 11:07:16   ← 代码修复（比数据晚 28 分钟）
```

而 `configs/data/math.yaml` 仍指向这两个文件：

```yaml
train_path: "data/math_train_rl.jsonl"    # 25.1% ground truth 被截断
test_path:  "data/math_test_clean.jsonl"  # 25.7% 被截断；eval 子集 20.3%
```

### 对已完成实验的影响

`train_20260807_062356.log`（200 步）是在这份污染数据上跑的：

1. **eval_acc 存在结构性上限**：eval 子集 20.3% 的题 ground truth 残缺，
   模型答对也被判错 → **天花板约 79.7% 而非 100%**。
   实测 eval_acc 0.693 对应有效题目上的真实准确率 ≈ 0.693/0.797 ≈ **87%**。
2. **与 Fix 7 叠加**：这 25% 的"死题"模型永远答不对 → 组内 16 个 rollout 全错
   → 零方差 → 被退化组过滤丢弃。这是 `group_keep_rate` 的一个**确定性**
   损耗源，与策略好坏无关。
3. **少量假阳性**：截断后偶尔反而匹配（如 gt `'\frac{1'` 提取为 1.0，
   模型输出 `1` 也得 1.0 → 判对，而正确答案是 0.5）。主要是假阴性，
   但假阳性同样存在。

### ⚠️ 重生流程比想象的麻烦：派生文件无生成脚本

不能只跑 `python data/prepare_data.py` 就事——它**只生成**
`math_train.jsonl` / `math_test.jsonl`，而配置实际使用的是派生文件
`math_train_rl.jsonl` / `math_test_clean.jsonl`，**仓库里没有任何脚本生成它们**
（已 grep 全仓库确认）。

实测派生关系：

```
math_train.jsonl   5586 条  →  math_train_rl.jsonl    5185 条  （剔除 401）
math_test.jsonl    3669 条  →  math_test_clean.jsonl  3663 条  （剔除 6）

✓ 严格子集（question 集合上）
✓ 各 level 按比例减少（L5 2304→2140, L4 1690→1577, L3 1592→1468）
✗ 不是去重（原文件 question 本身无重复）
✗ 不是"剔除非纯数值 answer"（被剔的含纯数值 '6'/'9'/'-30'，
   保留的含 1783 条 LaTeX）
✗ 不是"剔除空 answer"（被剔的 401 条中 0 条为空）
```

**派生规则已破案（2026-08-11 实测）：是 SFT 数据泄漏防护。**

`generate_sft.py` 用 `seed=42` 从 `math_train.jsonl` 采样 500 题生成 SFT 数据
（成功 372 条）。实测剔除的 401 题中：

```
361/401 (90%) ∈ seed42/n500 采样集
290/401 (72%) ∈ SFT 成功的 372 题
test 剔除的 6 条：与 train、SFT 均零重叠（3 条是截断坏答案，疑似手工清理）
```

→ `math_train_rl` 的语义就是 **"RL 训练集 = 主训练集 − SFT 用过的题"**。
未精确吻合的原因：当前 `math_train.jsonl` 是 10:39 重新生成的版本，与当初
SFT 采样时的文件顺序/条数已不同，seed42 无法复现当时的采样——
**精确剔除列表不可复现，但语义规则清楚，可按语义重建**（以 `sft_train.jsonl`
中实际出现的 372 个 question 为剔除依据）。

### ❌ 早先设想的"改配置指向主文件"方案不成立

实测主文件同样污染：`math_train.jsonl` 截断率 **25.1%**（1404/5586）、
`math_test.jsonl` **25.8%**（945/3669）——它们与派生文件是同一时刻（10:39）、
用修复前代码生成的。只改配置指向、不重跑脚本，截断率一点不变，
还会引入 SFT 泄漏（372 题回到 RL 训练集）。

### ✅ 最终处理方案（已执行并验收，2026-08-11）

```
1. 重跑 python data/prepare_data.py（修复后代码）→ 干净主文件
2. 新增 data/derive_rl_split.py（入库，消除"无脚本可复现"悬案）：
   - 过滤空 answer（修复后解析器对不平衡花括号返回空串）
   - 剔除 sft_train.jsonl 中实际出现的 question（防泄漏）
   - 重建 math_train_rl.jsonl
3. test_path 改指 math_test.jsonl（_clean 的6条剔除无泄漏依据，
   坏答案在修复后自然消失，无保留价值）
4. 旧污染文件备份在 data/backup_contaminated_20260807/（留作对照证据）
```

**执行结果（实测）**：

```
math_train.jsonl     5586 条  花括号不平衡 0（修复前 1404/25.1%）、空答案 4
math_test.jsonl      3669 条  花括号不平衡 0（修复前  945/25.8%）、空答案 0
math_train_rl.jsonl  5265 条  ＝ 5586 − 4 空答案 − 317 SFT 泄漏题
RL 集 ∩ SFT 题目 ＝ 0；eval 子集 Level5[:300] 坏答案 ＝ 0（修复前 61/20.3%）
修复样例：'\frac{1' → '\frac{1}{2}'，'\sqrt{2' → '\sqrt{2}+1'
```

注：新 RL 集 5265 条 vs 旧 5185 条——SFT 的 372 题中只有 317 题存在于当前主集
（其余 55 题来自早期版本的主文件），旧的 401 剔除列表本就含不可复现成分，
题目数差异符合预期。

### 对实验可比性的影响（重要）

1. **旧 checkpoint 与旧 eval_acc 均不可与新数据实验跨比**，需重跑 baseline。
2. eval 子集是"Level 5 按文件顺序取前 300"，重新生成会改变文件顺序 →
   **eval 题目集合本身也变了**。数据修复前后的曲线画在同一张图上时必须标注分界。
3. SFT 数据（`sft_train.jsonl`）由外部 API 生成，**不经过 `extract_math_answer`**，
   不受本问题影响，无需重生；且其 question 列表是去泄漏剔除的依据，**不可删改**。

---
---

# 第二版（RACA v2，当前生效纲领）

> 最后更新：2026-08-11
> **核心卖点重定义**：不是"系统里有交互"，而是"交互作为显式动作被因果信用训练出来，
> 且有行为演化证据"。v2 把 agent 间通信建模为带因果 credit 的一等动作——
> 不仅评估说了什么，还评估该不该说、说给谁听、说完有没有用。

---

## 0. v2 相对 v1 的变更总览

| # | 变更点 | v1 | v2 | 动机 |
|---|--------|----|----|------|
| 1 | Controller 定位 | strategy + focus + stop 三职责 | **仅 continue/stop**（元认知终止者） | focus 被 proposer fallback 架空；路由与交互机制冲突 |
| 2 | 路由方式 | controller focus + 自由交互并存 | **交互是唯一路由机制** | 消除双重路由，交互成为 load-bearing |
| 3 | 交互决策奖励 | 无（与 reward 零相关） | **交互因果奖励 + 固定成本 −c** | 无成本求助是 free option，会导致交互率饱和 |
| 4 | 交互响应计分 | 与主 turn 共享 round 奖励（搭便车） | **响应 turn 按角色语义独立计分** | 响应质量需要独立梯度信号 |
| 5 | 响应存储 | 覆盖 role_outputs（污染解析） | **primary / responses 分槽存储** | 根治 _critic_found_errors 类补丁问题 |
| 6 | Layer 2 anchor | (role, controller 输出的 strategy) | **(role, 机械推导 σ, is_response)** | controller 标签随策略漂移，机械标签跨阶段可比 |
| 7 | Anchor 组去重 | 同 episode 同轮多 turn 重复入组 | **按 (episode, round, role) 去重后广播** | 重复样本压低组内方差，稀释真实差异 |
| 8 | Verifier 地位 | 分数不进任何决策（功能悬空） | **stop 的前置闸门** | 校准能力直接影响系统行为，求助 verifier 有因果通路 |
| 9 | Critic 末轮奖励 | 真阳性回退用 c^i（幸存者偏差回流） | **末轮真阳性给固定正分** | 末轮无下一轮可影响，不应与结局耦合 |
| 10 | Critic/Verifier 调用 | 由 focus / 交互随机触发 | **仅按需调用 + ε 强制注入** | 冷启动保护，避免 critic 零训练数据 |
| 11 | 动作集 | none/request/support/challenge | **none/request/challenge**（删 support） | support 不改变任何状态，是纯噪声动作 |

Controller 奖励公式（底薪 + 效率提成 + 对称惩罚，含 γ 项）**不变**；
Proposer 逐轮奖励 p^i_t **不变**；Phase 4 更新流程 **不变**。

---

## 1. 系统设定

模型架构不变：冻结 base LLM（Qwen3-8B）+ 4 个独立 LoRA adapter，
k ∈ {ctrl, prop, crit, verif}。

**角色定位（v2）**

| 角色 | 决策视野 | 唯一职责 | 优势层 |
|------|---------|---------|--------|
| Controller | 宏观（episode） | 终止决策：当前答案值得信任吗？再跑一轮值不值？ | Layer 1 |
| Proposer | 微观（turn） | 生成/改进解法 + 决定是否求助、向谁求助 | Layer 2 |
| Critic | 微观（turn） | 按需审查错误 + 可要求 proposer 修正 | Layer 2 |
| Verifier | 微观（turn） | 按需校准置信度，**把守 stop 闸门** | Layer 2 |

决策层次与优势层一一对应：宏观决策（何时终止）用 episode 级 credit，
微观通信（谁向谁说话）用 step 级 credit + 交互因果奖励。

**新增符号**

| 符号 | 含义 |
|------|------|
| σ^i_t | 第 t 轮开始时从黑板机械推导的上下文标签 ∈ {explore, refine, verify} |
| u^i_t | 1[proposer 第 t 轮发起了交互] |
| Δp^i_t | 交互后本轮内 proposer 答案的修正效果 |
| c_int | 交互固定成本（默认 0.05） |
| λ | 交互奖励合成权重（默认 1.0） |
| ε_t | 强制注入 critic 审查的概率（随训练衰减） |

---

## 2. 交互协议

### 2.1 每轮流程

```
1. Controller 看黑板 → continue | stop
   约束：stop 仅当黑板上存在 verifier 分数；否则强制 continue 或自动触发一次 verify
2. Proposer（固定起点，focus 逻辑删除）：解题/改进
   → 输出答案 + 交互决策 {none | request:critic | request:verifier}
3. 若发起交互 → target 以完整角色格式响应（追加写入，不覆盖）
4. 响应方可再发起一跳（如 critic 发现错误 → request:proposer 要求修正，
   proposer 的修正输出作为新 trace 进入多数投票——这是交互影响最终结果的因果通路）
5. 每轮最多 2 跳；round record 落盘
```

### 2.2 动作集

```
action ∈ {none, request, challenge}     # 删除 support（不改变状态的噪声动作）
target ∈ 另外两个工作角色
```

### 2.3 存储规则

`role_outputs` 分两个槽：
- `primary[role]`：该角色本轮的主输出（解析 p_t / flag / score 只用这个槽）
- `responses[role]`：交互响应列表（独立计分，不参与主输出解析）

### 2.4 冷启动保护（ε 强制注入）

critic/verifier 仅按需调用会导致冷启动死锁：训练初期 proposer 从不求助
→ critic 永远没有训练数据。解法：

```
以概率 ε_t 强制注入一次 critic 审查（ε_0 = 0.3，线性衰减至 0.05）
- 被强制轮次：发起方不计交互决策奖励（决策不是它做的）
- 响应方正常计分（其 token 仍是 on-policy 的，只有"被调用"这件事是强制的）
```

---

## 3. 机械 σ 推导（替代 controller strategy 标签）

每轮开始时从黑板状态确定性推导：

```
σ = explore   若黑板无 trace
  = refine    若最近一条 critic flag 存在且未被后续 trace 处理
  = verify    若已有候选答案且无未处理的 flag
```

**为什么替换**：controller 生成的 strategy 是策略相关的——训练中策略一变，
anchor 分组的语义就漂移。机械推导的 σ 跨 rollout、跨训练阶段严格可比，
且不依赖任何模型输出的解析成功率。

---

## 4. Phase 2（v2）：角色奖励 + 交互奖励

### 4.1 Proposer（不变）

```
r_prop[i, t] = p^i_t
```

### 4.2 交互发起奖励（新增，核心设计）

交互是一个动作，用它引起的状态变化计分，且**每次发起收固定成本 −c_int**：

```
发起（u=1）：r_int = −c_int + { +0.3   p_t=0 且交互后修正为对   （有效求助）
                                0      p_t=0 且交互后仍错       （无效求助）
                               −0.2   p_t=1                    （画蛇添足）}
不发起（u=0）：r_int = { 0     p_t=0   （机会成本已隐含在 r_prop 中，不双重计罚）
                        +0.1   p_t=1   （正确的自信）}
```

效果窗口：交互链在本轮内结束，"修正为对"以本轮结束时 primary/修正 trace 的
正确性判定。

**为什么必须有 −c_int**：若求助零成本且偶尔有用，"永远求助"是弱占优策略，
交互率饱和到 100%，"学会交互"退化为"无脑交互"。加成本后，仅当预期修正收益
超过通信成本时求助才划算，训练出的才是**选择性交互**——预期看到交互率与 p_t
的负相关随训练增强，这是卖点的直接证据。

**合成方式**：交互决策与角色输出在同一 turn 生成，该 turn 奖励为

```
r_turn = r_role + λ · r_int        # λ = 1.0，消融 {0, 0.5, 1.0, 2.0}
```

λ=0 即"有交互但不训练交互"，正是 v1 行为，作为关键对照组。

### 4.3 Critic（四格矩阵保留，末轮修正）

```
         flagged (f=1)                        silent (f=0)
p_t=0    真阳性: 0.3·p_{t+1} + 0.1·(1−p_{t+1})    漏检: 0
p_t=1    假阳性: −0.2                              真阴性: +0.1
```

**末轮修正（v2）**：t = t_stop 时真阳性给固定分 **+0.2**，不再回退用 c^i。
理由：末轮 critic 没有下一轮可影响，用 episode 结局计分会把 v1 批评过的
幸存者偏差从后门带回来——而末轮恰是 critic turn 占比最高的位置。

critic 作为响应方被调用时，同样按此矩阵计分（p 以它实际审查的那个答案计）。

### 4.4 Verifier（校准奖励不变 + 闸门职责）

```
r_verif[i, t] = 1 − |v^i_t − p^i_t|
```

v2 中 verifier 的分数把守 stop 闸门（§2.1），校准好坏直接影响系统终止行为，
不再功能悬空。

### 4.5 Controller（公式不变）

```
r_ctrl[i] = c^i
           + α · c^i · (T_max − t_stop) / T_max          # 效率提成（答对才有）
           − β · (1 − c^i)                                # 答错惩罚
           − γ · (1 − c^i) · (T_max − t_stop) / T_max    # 对称的省轮惩罚
```

α = 0.3，β = 0.2，γ = 0.3。仍仅赋给结束 episode 的那个 controller turn
（stop turn 优先，耗尽轮次则为最后一个工作 turn）。

### 4.6 交互响应方计分

响应 turn 不再复用 round 奖励，按响应方自己的角色语义在响应后状态上评估：

| 响应方 | 计分方式 |
|--------|---------|
| Critic | 四格矩阵，p 以被审查答案计，p_{t+1} 以响应后本轮修正结果计 |
| Proposer（修正响应） | 新答案的 p（即 math_equal(新答案, a*)） |
| Verifier | 校准奖励 1 − \|v − p\| |

---

## 5. Phase 3（v2）：两层优势

### 5.1 Layer 1：Episode 级（仅 Controller，不变）

```
A_E_ctrl[i] = (r_ctrl[i] − μ_ctrl) / max(σ_ctrl, δ)
```

零方差组照旧丢弃该层（不发零梯度、不稀释 total_valid）。

### 5.2 Layer 2：Step 级 anchor（v2 修改）

**Anchor key**：

```
s̃ = (role = k, σ, is_response)      # σ 为机械推导标签，is_response ∈ {0, 1}
```

交互响应 turn 只和交互响应 turn 比，主 turn 只和主 turn 比。

**组内去重（v2 新增）**：同一 episode 同一轮的同角色多个 turn 携带相同 reward
时，只以一个代表样本入组计算 μ/σ，得到的 advantage 再广播回该轮该角色的全部
turn。防止重复样本人为压低组内方差。

```
μ, σ ← mean/std over 去重后的 G_S(k, σ, is_resp)
A_S = (r − μ) / max(σ, δ)
|G_S| < 2 或 σ ≤ δ 时跳过该 anchor
```

### 5.3 路由（不变）

```
A_total[ctrl, i, t] = A_E_ctrl[i]
A_total[k,    i, t] = A_S[k, i, σ_t, is_resp]     # k ∈ {prop, crit, verif}
```

---

## 6. Phase 4：策略更新

PPO clip loss + KL 惩罚，per-turn 立即 backward，
梯度按 total_valid 归一化，4 adapter 经 set_adapter() 天然隔离。
KL 参考：v2.0 对裸 base model；v2.1 起改为对冻结的 SFT 快照（见 §14.4）。

---

## 7. v2 伪代码

```
Algorithm RACA-v2
─────────────────────────────────────────────────────────────────
Params: N=16, T_max=4, clip_ε=0.2, α=0.3, β=0.2, γ=0.3, δ=1e-4,
        c_int=0.05, λ=1.0, ε_0=0.3, max_hops=2
─────────────────────────────────────────────────────────────────
For each training step:

  ▌ Phase 1: Rollout
  For each q, i=1..N:
    For t = 1..T_max:
      σ^i_t ← derive_sigma(blackboard)                  # §3 机械推导
      ctrl ← Controller(blackboard)
      if ctrl == stop and blackboard.has_verifier_score: break
      prop ← Proposer(q, blackboard)                    # 固定起点
      记录 primary 输出、交互决策 u^i_t
      forced ← Bernoulli(ε_t)                           # 冷启动注入
      if u^i_t or forced:
        执行交互链（≤ max_hops），响应追加写入 responses 槽
    记录 {p, f, v, u, Δp, σ, t_stop, c}

  ▌ Phase 2: Rewards
  r_prop  ← p^i_t
  r_int   ← 交互因果矩阵（§4.2），forced 轮次发起方不计
  r_crit  ← 四格矩阵，末轮真阳性固定 +0.2（§4.3）
  r_verif ← 1 − |v − p|
  r_ctrl  ← c + α·c·rem − β·(1−c) − γ·(1−c)·rem
  proposer 主 turn: r_turn ← r_prop + λ·r_int
  响应 turn: 按 §4.6 独立计分

  ▌ Phase 3: Advantages
  Layer 1: A_E_ctrl ← 组内归一化 r_ctrl（零方差丢弃）
  Layer 2: anchor (role, σ, is_response)，组内去重后归一化，广播回 turn

  ▌ Phase 4: Update（同 v1）
─────────────────────────────────────────────────────────────────
```

---

## 8. 证据指标（卖点的实验支撑）

训练全程记录：

| 指标 | 定义 | 预期演化 |
|------|------|---------|
| 交互发起率 | mean(u^i_t) | 从随机 → 收敛到选择性水平（远低于 100%） |
| 交互有效率 | P(wrong→right \| 发起交互) | 持续上升 |
| 选择性 | corr(u^i_t, p^i_t) | 负相关随训练增强（错的时候才求助） |
| 按 action/target 分解频率 | — | request:critic 在 refine 语境占比上升 |
| stop 校准 | P(correct \| stop) vs P(correct \| exhausted) | stop 组显著更高 |

**核心消融（三组对照）**：

```
A. 禁用交互（max_hops=0）
B. 允许交互但 λ=0（= v1 行为：有交互、不训练交互）
C. 完整 v2
```

C > B 即证明"交互被训练"有效；B vs A 分离"交互机制存在"本身的贡献。
附加敏感性：λ ∈ {0, 0.5, 1, 2}，max_hops ∈ {1, 2, 3}，c_int ∈ {0, 0.05, 0.1}。

---

## 9. 超参数汇总（v2 新增部分）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| c_int | 0.05 | 交互固定成本 |
| λ | 1.0 | 交互奖励合成权重 |
| ε_0 → ε_min | 0.3 → 0.05 | 强制注入概率，线性衰减 |
| max_hops | 2 | 每轮交互链上限 |
| 末轮真阳性固定分 | +0.2 | critic 末轮奖励 |

---

## 10. 已知风险与观察点

1. p_t=0 不发起给 0 分而非负分是刻意的（r_prop 已计罚，避免双重计数）；
   若交互率过低，再考虑给"该求助没求助"加小负分。
2. ε 衰减 schedule 需调：衰减太快 critic 训练不充分，太慢污染交互决策分布。
3. 交互决策与角色输出同 turn 合成奖励存在 credit 混叠；若 λ 消融显示信号
   互相干扰，备选方案是把交互决策拆成独立短生成 turn（代价：每轮多一次调用）。
4. stop 闸门依赖 verifier 分数存在，需监控"为了 stop 而空跑 verify"的行为。

---

## 11. 待实现代码改动清单（v2）

> **状态：已全部实现（2026-08-16，实施记录见本节末尾）。**

- [x] `llm/prompt_templates.py`：controller prompt 改为仅 continue/stop；
      proposer/critic/verifier 动作集删 support；proposer 交互决策格式明确化
- [x] `agents/agentic_executor.py`：删除 focus 解析与 start_role 逻辑，proposer 固定起点
- [x] `agents/agentic_executor.py`：role_outputs 分 primary/responses 双槽
- [x] `agents/agentic_executor.py`：实现机械 σ 推导（derive_sigma，落在 envs/blackboard.py）
- [x] `agents/agentic_executor.py`：stop 闸门（无 verifier 分数不得 stop）
- [x] `agents/agentic_executor.py`：ε 强制注入 + forced 标记
- [x] `agents/agentic_executor.py`：交互因果奖励 r_int、响应独立计分、
      critic 末轮固定分、r_turn 合成（落在 agents/raca_rewards.py）
- [x] `training/grpo_trainer.py`：anchor key 改 (role, σ, is_response) + 组内去重
      （落在 training/raca_adv.py）
- [x] `training/grpo_trainer.py`：新增指标日志（交互率/有效率/选择性/stop 校准）
- [x] `configs/agentic/default.yaml`：新增 c_int, lambda_int, eps_force, max_hops
- [x] `train.py`：消融开关（max_hops=0 / lambda_int=0 经 Hydra 覆盖即可）

---

## 12. v2 实施记录（2026-08-16）

### 新增/重写的模块

| 模块 | 职责 | 可测性 |
|------|------|--------|
| `agents/parsing.py` | decision/interaction/reasoning/score 解析（纯正则） | 零 torch 依赖 |
| `agents/raca_rewards.py` | Phase 2 全部奖励（r_int/四格/响应计分/controller） | 零 torch 依赖 |
| `training/raca_adv.py` | Phase 3 两层优势 + anchor 去重广播 | 零 torch/numpy 依赖 |
| `envs/blackboard.py` | 事件序号 + `derive_sigma()` | 零依赖 |
| `agents/agentic_executor.py` | v2 主流程（重写，无 HF-generate 旧路径） | 集成测试用 Fake 引擎 |
| `test_raca_v2.py` | 15 条单测 + 2 条端到端集成（含在 15 内） | 本地 CPU 可跑 |

### 规格未定死处的实施决策

1. **Critic 真阳性效果量 q 三级解析**（合并 §4.3 与 §4.6 语义）：
   本轮有后续修正 → q = 轮末正确性；无修正但有下一轮 → q = 下轮 primary；
   末轮无修正 → 固定 +0.2。
2. **stop 闸门拦截后强制 continue**（规格两选一中选了简单项，未自动触发 verify），
   拦截次数记入 `gate_blocked` 监控（对应 §10 风险 4）。
3. **r_int 只挂在 primary proposer turn**；响应方在链上再发起的下一跳属链机制，
   不单独计奖（避免响应 turn 的 credit 混叠）。
4. **forced 只在 u=0 时注入**；被注入轮发起方 r_int=0，响应方正常计分；
   **eval 时 ε=0**（评测学到的策略本身）。
5. **响应方 prompt = 标准角色格式 + 请求上下文后缀**，而非 v1 的自由格式
   interaction_response —— 保证错误分析/分数字段可解析（配合双槽存储根治 Fix 3 类问题）。
6. **KL 估计器 k1 → k3**（`exp(ref−new) − (ref−new) − 1`，delta clamp≤20）：
   v1 的 k1 在 on-policy 采样下梯度期望为零，无约束力只加噪声；k3 非负且
   梯度方向正确。**日志里 kl 数值量纲与旧实验不可比（k3 恒非负）。**
7. 删除 v1 遗留：HF-generate 无 vLLM 路径（早已不可用）、seq_input_ids/seq_step_ids
   记账（trainer 不消费）、strategy/focus 解析。`run_episode` 保留为 batch 路径的
   薄包装；`evaluate.py` 离线评测现需传入 vLLM 引擎才能运行。

### 新增监控指标（wandb）

**RACA v2 证据指标（§8）**：`int_rate` / `int_effectiveness` /
`int_selectivity`（预期负相关增强）/ `forced_rate` / `stop_rate` /
`stop_acc` vs `exhaust_acc`（stop 校准）/ `gate_blocked` / `eps_force`；
eval 侧新增 `eval_int_rate`、`eval_stop_rate`。

**标准 GRPO 看盘项（补齐 v1 缺失项）**：

| 类别 | 指标 | 健康区间 / 危险信号 |
|------|------|------------------|
| 策略健康 | `entropy` | 缓降正常；**断崖式下跌 = 熵坍塌**（GRPO 最常见死法） |
| | `kl`（k3） | 缓慢爬升正常；指数上升 = 漂移失控 |
| | `clip_frac` | 通常 <0.2；持续走高 = lr 过大或 off-policy 太深 |
| | `ratio_mean` / `ratio_max` | mean 应 ≈1；max 飞了 = rollout/训练权重脱节 |
| | `grad_norm`（stdout） | 有界即可；持续打到 clip 阈值是警报 |
| 信号质量 | `group_keep_rate` | 优势可用组占比（现有） |
| | `all_pass_frac` / `all_fail_frac` | 组内零方差 ⇒ 无梯度；持续上升 = 信号枯竭 |
| | `group_reward_std` | 趋零 = 信号枯竭（考虑 DAPO 动态采样） |
| 行为 | `resp_len` | 暴涨 = reward hacking / 熵坍塌前兆；暴跌 = 学会敷衍 |
| | `parse_rate` | 「最终答案：」字段输出率；格式崩了 reward 再高也是假的 |

**实施细节**：
- `entropy` 必须在训练前向内采集（vLLM 只回 top-20 logprobs，算不出全分布熵）；
  复用 `_compute_loss` 现有 logits，**分块（128 token）累加**——整体 log_softmax
  会开 (n_resp, 151936) float32（n_resp=1024 时 ≈620MB）再加 exp() 又一份，极易 OOM。
- 健康指标统一按 **token 加权**（分母 `n_tok`），避免短 turn 被过度加权；
  均在 `no_grad` 下采集，不入计算图。
- rollout 行为指标抽到 `training/metrics.py`（零 torch 依赖，可 CPU 单测），
  统计对象是**全部** rollout（含被优势过滤掉的 episode）。
- `loss` 本身基本不看（组内 z-normalization 使其期望为零），只在出 NaN 时有信息量。

### 断代提醒

v2 与 v1 实验三重不可比：① Fix 18 判分器变严（eval_acc 旧刻度系统性偏高）；
② 奖励结构变化（r_int/响应计分/末轮固定分）；③ kl 曲线换 k3 估计器。
v1 对照基线：用新 grader 重跑 rl-math-grpo_fixed-3 的 step-200 checkpoint eval。

---

# 第三版（RACA v2.1）：v2.0 首跑失败复盘与奖励矩阵校准

> 记录时间：2026-08-19　依据：`train_v2_full_20260817_052656.log`（79/200 步后终止）

## 13. v2.0 首跑的连锁失效

### 13.1 观测事实

| step | acc | entropy | int_rate | stop_rate | kl |
|------|-----|---------|----------|-----------|-----|
| 1 | 0.35 | 1.233 | **0.58** | **0.45** | 0.748 |
| 20 | 0.51 | 0.924 | **0.02** | 0.21 | 0.422 |
| 50 | 0.68 | 0.516 | **0.00** | 0.06 | 0.226 |
| 79 | 0.59 | 0.445 | **0.01** | 0.04 | 0.178 |

eval 侧（greedy）更极端：`int_rate=0.00`、`stop_rate=0.00`、`avg_turns=8.0`
**三项完全恒定**。8.0 = 4 轮 ×（1 controller + 1 proposer），即**一个
critic/verifier turn 都没产生**。`eval_acc≈0.84` 看似漂亮，但完全来自
「4 轮 proposer + 多数投票」，与 v2 的交互机制无关——卖点机制全程空转。

### 13.2 根因：r_int 矩阵存在数学缺陷

v2.0 §4.2 给「不发起 + 答对」+0.1（"正确的自信"）。这是一笔**无条件补贴**，
与「发起 + 答对」的 −0.2 共同制造 `0.3p` 的固定 gap，而有效求助最多只能赚
`0.3(1−p)`。发起交互的边缘期望：

```
E[发起] − E[不发起] = −c_int − 0.3p + 0.3q(1−p)
```

| p(答对) | q=0 | q=0.5 | q=1.0（求助必成功） |
|---------|-----|-------|---------------------|
| 0.50 | −0.200 | −0.125 | **−0.050** |
| 0.60 | −0.230 | −0.170 | **−0.110** |
| 0.70 | −0.260 | −0.215 | **−0.170** |

**即使 q=1.0，任何 p≥0.35 下期望差都为负** → `int_rate→0` 是数学必然，
不是训练不充分。这与既有教训（"消除补贴时必须保持非负期望"）属同一类错误：
v2.0 在消除 v1 的搭便车问题时，反手给"不发起"加了新的无条件补贴。

### 13.3 连锁失效路径

```
r_int 矩阵使发起永远不划算
  → int_rate 0.58→0.00（step 20 即塌）
  → verifier 几乎从不被调用（ε 注入只注 critic，verifier 无兜底通道）
  → 黑板永无 verifier 分数
  → stop_gate 拦下所有 stop → stop_rate 0.45→0.04
  → 每个 episode 跑满 max_rounds → avg_turns 恒 8.0
  → 系统退化为「多轮多数投票」，v2 两个核心机制全部失效
```

熵从 1.233 跌到 0.445（−64%）是上述行为收敛的伴生现象——策略正确地收敛到
"永远输出 action: none"，属激励结构导致的合理收敛，而非独立的熵坍塌病症。

## 14. v2.1 修正（14.1–14.3 同属一条失效链，必须同时改；14.4 为复查追加）

### 14.1 r_int 矩阵重设计（经期望值校准）

```
发起（u=1）：r_int = −c_int + { +int_gain      p_t=0 且交互后修正为对
                                0             p_t=0 且仍错
                               −int_overkill  p_t=1 }
不发起（u=0）：r_int = 0        # 删除「正确的自信」+0.1 补贴
```

参数：`c_int=0.02`、`int_gain=0.3`、`int_overkill=0.05`。校准后的期望差：

| p(答对) | q=0 | q=0.3 | q=0.5 | q=0.8 | q=1.0 |
|---------|-----|-------|-------|-------|-------|
| 0.30（弱） | −0.035 | **+0.028** | +0.070 | +0.133 | +0.175 |
| 0.60（当前） | −0.050 | −0.014 | **+0.010** | +0.046 | +0.070 |
| 0.85（强） | −0.062 | −0.049 | −0.040 | −0.026 | −0.018 |

设计意图达成：**弱时求助划算、强时不划算、中间取决于求助有效性 q**，
模型的自我置信度成为决定因素（即"选择性交互"）。条件期望同样正确：
p_t=0 时发起 `−0.02+0.3q > 0`（q>1/15 即可）；p_t=1 时发起 `−0.07 < 0`。

### 14.2 Layer 2 anchor 增加 p_t 分层

proposer 主 turn 奖励 = `r_prop(0或1) + λ·r_int(±0.35)`。不分层时组内归一化
后的优势主要反映"答对了吗"（差异 1.0），交互决策（差异 ≤0.35）被淹没成噪声。
anchor key 扩展为 `(role, σ, is_response, layer_key)`，`layer_key = p_t`
（仅 proposer 主 turn 非 None）。分层后组内 `r_prop` 相同、被均值消掉，
优势纯粹反映交互决策优劣。回归测试 `test_layer2_p_t_stratification` 验证：
不分层时"答对却求助"仍获正优势（画蛇添足未被惩罚），分层后被正确判负。

### 14.3 verifier 兜底调用通道

v2.0 的 ε 注入固定注 critic，verifier 无任何被动调用通道，而它恰是 stop
闸门的看门人。v2.1 双管齐下：

1. `gate_blocked`（controller 想停但黑板无分数）→ **强制注入 verifier**，
   直接解锁终止路径（对应 §2.1 原文"或自动触发一次 verify"分支）
2. 其余 ε 注入按 `eps_verifier_share=0.5` 在 critic/verifier 间随机选

### 14.4 KL 锚定修正：参考模型从裸 base 改为冻结 SFT 快照（复查追加）

**证据**：v2.0 日志里 k3 KL 从 0.748（step 1）**单调**降到 0.178（step 79）——
策略在被 `kl_coef=0.04` 持续拉向 base，而 base 从未学过 `<interaction>`、
`decision: stop`、`分数:` 格式——KL 惩罚在系统性抹除恰好是 v2 赖以运转的
SFT 行为，与 int/stop 坍塌高度同步。这是独立于 r_int 矩阵缺陷的第二条
压力通路：即使 14.1 矩阵正确，KL 锚错位也会把交互格式磨掉。

**实现**（`trainable_llm.py` / `grpo_trainer.py`）：
- `load_trainable_models` 在 SFT 权重加载后为每个角色克隆冻结的
  `ref_{role}` adapter（= 训练起点快照）；克隆发生在 resume 回填之前，
  断点续训时 ref 仍是 SFT 起点而非 RL 中间态。
- `as_ref(role)` 切到 `ref_{role}` 做参考前向；无 ref adapter 时回退旧的
  `disable_adapter_layers()`（兼容旧脚本）。
- `lora_parameters()` 排除 `.ref_`：优化器 / clip_grad_norm_ / NaN guard
  都不碰 ref；`save_pretrained` 不存 ref（load 时从 SFT ckpt 重建），
  checkpoint 目录结构不变，`sync_lora` 只认四个角色名也不受影响。
- 无 SFT ckpt 时 LoRA B 零初始化，ref 等价 base，行为与旧实现一致。

**验收信号**：step 1 的 `kl` 应 ≈ 0（策略=SFT 起点=参考），之后缓慢爬升；
v2.0 是 0.75 起步单调下降。若首步 kl 仍在 0.5+，说明 ref 没挂上。

**配套观测修复**：stdout step 行补齐 `eff=`（int_effectiveness，即 q 的在线
估计）、`sel=`（int_selectivity）、`parse=`、`gate=`；缺失时打 `--` 而非 0，
区分“无自发求助样本”与“求助全部无效”——v2.0 日志正因缺这几项无法
事后验证 q（q > c_int/int_gain ≈ 6.7% 是 14.1 矩阵成立的前提）。

## 15. 熵指标的处理决定

不动 `kl_coef`、不加 entropy bonus（KL 参考的更换见 §14.4，那是锚点纠错
而非旋钮调参）。理由：熵下降是 13.3 中行为退化的**结果**
而非独立病因，修复激励结构后交互决策重新具备价值，熵应自然回升。同时改多个
耦合旋钮会破坏归因能力（既有教训）。观察点：v2.1 首跑若 int_rate 回升而 entropy
仍单调跌向 0.2 以下，再考虑独立干预。

## 16. v2.1 与 v2.0 的可比性

奖励矩阵改变 → `reward` 曲线绝对值不可比；KL 锚更换（§14.4）→ `kl` 曲线
不可比（v2.1 应从 ≈0 起步缓升，v2.0 是 0.75 起步单降）；`int_rate` /
`stop_rate` / `entropy` / `eval_acc` 可比（同一 grader、同一 SFT 起点
`checkpoints/sft_v2`）。
v2.0 那 79 步的价值在于**证伪**：它是"交互奖励设计错误会导致机制空转"的
反面证据，可作为论文中 r_int 设计必要性的实证支撑（比单纯的 λ=0 消融更有力）。

---

# 第四版（RACA v2.2）：v2.1 首跑复盘——q≈0，修正跳断链

> 记录时间：2026-08-19　依据：v2.1 首跑前 13 步日志

## 17. v2.1 首跑：两个修复验证生效，但 int_rate 仍塌——真凶是 q≈0

### 17.1 观测事实

| step | int_rate | **eff** | sel | parse | gate | kl | ent |
|------|----------|---------|-----|-------|------|-----|-----|
| 1 | 0.58 | **0.00** | +0.05 | 0.89 | 0 | 0.0000 | 1.29 |
| 10 | 0.15 | **0.01** | +0.02 | 0.75 | 18 | 0.0064 | 1.15 |
| 13 | 0.03 | **0.00** | +0.03 | 0.77 | 32 | 0.0107 | 1.31 |

- **§14.4 KL 锚定验收通过**：kl 从 0.0000 起步缓升（v2.0：0.75 起步单降）；
  ent 稳在 1.0–1.4（v2.0 同期已奔 0.44）——熵未坍塌。
- **gate_blocked 兜底通道工作正常**：gate 0→32，stop_rate 被拉回 0.32。
- **但 int_rate 塌得比 v2.0 更快**（step 13 即 0.03），且 `eff` 恒 ≈ 0：
  自发求助后“错→对”几乎从未发生。q≈0 时 §14.1 矩阵两个分层发起都是
  负期望（p_t=0：−0.02+0.3·0；p_t=1：−0.07）——RL 正确地学会了
  “求助没用，别求了”。这次不是奖励数学错、也不是 KL 拉扯，是**交互
  真实地没有正价值**。

### 17.2 根因：修正跳由“未学会的行为”把守

`eff` 依赖 `corrected_answer`，而修正发生的前提链是：critic 的响应里
**自己再写一个 `<interaction>` request proposer**，第二跳才生成修正 turn。
SFT 数据里 critic 几乎不这么写 → `correction_turns` 恒空 → p_end=p_primary
→ eff 恒 0。`eff=0.00`（而非“低但有”）强烈指向漏斗断在这一跳。

### 17.3 v2.2 修正（两处）

1. **机械触发修正跳**（executor）：critic `flagged=True` → executor 直接
   调度 proposer 修正，不再依赖 critic 输出 `<interaction>`。与
   gate_blocked 强注 verifier 同一设计哲学：**关键因果通路由机制保证，
   角色只学“判断”，不学“走流程”**。critic 若同时自发发起，机械路径
   优先（不重复入队）。
2. **修正漏斗指标**（round_meta + metrics + step 行 `fnl=F/C/W`）：
   flag（critic 标错数）→ corr（修正 turn 数）→ flip（错→对翻转数），
   计入全部轮（含 forced）。判读：flag 高 corr 低 = 修正跳断（已由 1 根治）；
   **corr 高 flip 低 = proposer 拿着反馈也修不对 → 下一步该回 SFT 修
   critic 反馈质量，而非继续调 RL**。

### 17.4 重跑判据

step 30 探针：`fnl` 的 corr 应与 flag 同量级（机械触发生效）；真正的判决
变量是 **flip/corr（即 q 的直接估计）**：>7% 则 §14.1 矩阵成立、int_rate
应企稳；仍 <7% 则停手，问题在 critic/proposer 的修错能力（SFT 数据），
RL 层无解。另监控 parse（v2.1 首跑 0.89→0.74 阴跌，跌破 0.7 需处理）。

---

# 第五版（RACA v2.3）：交互 × acc 互补化——加权投票、双通道优势、零成本冷启动

> 记录时间：2026-08-20　依据：v2.2 首跑前 26 步日志

## 18. v2.2 首跑复盘与结构重设计

### 18.1 观测事实（step 1→26）

- **机械触发验收通过**：fnl 的 corr≈flag 恒成立；flip/corr ≈ 8–13% > 7% 阈值。
- **但 q 的构成暴露选择反向**：q_forced≈9–10%（随机注入命中可救错误），
  而自发 eff≈0–2%（step 1 自发主导时 fnl=59/65/1）；sel 早期恒正。
  模型在救不回的题上求助、可救的题上不求——“何时求助”的判断 SFT 没教对。
- **int_rate 第三次塌（step 16 归零）且成为吸收态**：GRPO 只从组内方差学习，
  归零后 r_int 全 0 → 连负梯度都没有，无法自愈。
- **新病：p_t 全量分层的副作用——主 turn 停训**。int_rate=0 后主 turn 组内
  零方差、整组被 variance floor 丢弃：解题能力从 step ≈14 起未再训练。
  证据：len 120→46（组成效应：进 loss 的只剩短 turn）；acc 横盘 0.4–0.5
  （v2.0 无分层同期 0.35→0.73）。

### 18.2 结构矛盾：多轮投票是交互的免费竞争者

第 1 轮答错后架构提供两条路：(a) 发起审查+修正（付成本、q≈9%）；
(b) 什么都不做，后续轮次免费重采样把错票投掉（self-consistency，
v2.0 零交互时 eval 0.747→0.863 即其功力）。加大轮数/交互预算只会让 (b)
更强。出路是互补化：**让交互产出投票用得上的信息**，三条通道：
① verifier 分数 → 加权投票；② critic 批评 → 改善后续轮次（FLAW 已入黑板）；
③ 修正 turn = 额外一票（警惕：若修正正确率 < 裸重采样，它在稀释投票质量）。

### 18.3 v2.3 修正（四处）

1. **加权投票**（executor `_majority_vote`，`vote_mode: weighted|uniform`）：
   被验证答案用平均分数加权、未验证用先验 0.5，票权 = 票数×权重。
   verifier 校准奖励本就在训，加权投票把它直接兑换成 acc（通道①）。
2. **主 turn 双通道优势**（raca_adv）：adv = z(r_prop，不分层) +
   z(λ·r_int，p_t 分层)。恢复解题信号（修 18.1 停训）；r_int 隔离保留；
   int_rate=0 时 prop 通道仍供梯度（修吸收态）。
3. **零成本冷启动**：c_int=0、int_overkill=0（发起在“没用”时期望相等、
   “有用”时为正；站稳后课程式恢复成本）；eps_force_min 0.05→0.25
   （供给不断）。不发无条件补贴（v2.0 已证伪）。
4. **q 拆分指标**：q_forced（随机干预基准）、int_critic_share；step 行
   `qF=`/`tgtC=`；forced 轮的 target 落盘。eff 与 qF 的 gap = 选择性质量。

### 18.4 第 0 步：三通道离线测量（measure_channels.py，重跑前必做）

在 primary 答错样本上测：a) 裸重采样（基准）、b) critic→修正、
c) 黑板带 FLAW 的下一轮；另测 verifier 分辨力（score|对 − score|错）。
判读：b/c > a → critic 通道有超额价值；b/c ≤ a → 修正票退出投票池、
critic 回 SFT；分辨力 ≤ 0 → 加权投票不成立，verifier 先回 SFT。

### 18.5 重跑判据（step 30 探针）

① int_rate 不归零（零成本下应企稳在 0.2+）；② acc 恢复上升趋势
（双通道生效，对照 v2.0 斜率）；③ eff 向 qF 靠拢（选择性形成）；
④ eval：weighted 相对 uniform 的 Δacc ≥ 0（通道①兑现）。
最终目标不变：完整系统 eval_acc > v2.0 的 0.863（投票-only 基线）。

## 19. SFT v3 二测与止损线执行（2026-08-20）

### 19.1 二测结果（checkpoints/sft_v3，n=300）

| 指标 | SFT v2（一测） | SFT v3（二测） | 判据 | 判决 |
|------|------|------|------|------|
| verifier 分辨力 Δ | +0.012 | **+0.312** | >0.2 | ✅ 通道①解锁 |
| critic flag_rate | 0.12 | **0.47** | — | 检错能力 ×4 |
| b) 修正 − a | −0.062 | −0.018 | b>a | ❌ 收窄但仍负 |
| c) 带 FLAW 下一轮 − a | −0.085 | −0.055 | c>a | ❌ 锚定效应仍在 |

特权蒸馏数据（generate_sft_v3.py，~1400 turns）对判别类能力（verifier/
critic 检错）有效，对生成类能力（拿着批评修对）收效有限——与置信度
预判一致（判断易、生成难）。

### 19.2 按预设止损线执行的三个决定（均留消融开关）

1. **加权投票开启**：`vote_mode: weighted`（Δ=0.312 达标；一测时橡皮图章
   下有害、曾紧急关闭）。
2. **修正票退出投票池**：`correction_in_vote: false`。修正 trace 仍进黑板
   （上下文/σ/r_int 因果信用保留），但不计票——两次测量证明它在稀释
   投票质量。按止损线不再第四次调 r_int 矩阵。
3. **FLAW 不进下一轮 primary prompt**：`flaw_in_primary_prompt: false`
   （Blackboard.to_text(include_flaws=False)）。批评的作用收敛到轮内修正
   链与 σ 推导；锚定伤害线剪断。

叙事收敛：**验证型交互有效（买信息给聚合器），纠错型交互受限于同底座
生成能力**——两个结论都有两轮测量支撑，可直接进论文。

### 19.3 RL 重启配置与探针判据

重启：新 exp_name，`sft_checkpoint=checkpoints/sft_v3`（KL ref 锚到 v3 起点）。
step 30 探针（替代 §18.5）：
① eval：weighted vs uniform 的 Δacc > 0（通道①在线兑现，核心判决）；
② acc 恢复上升斜率（双通道 r_prop 生效，len 回到 120+ 量级）；
③ int_rate 不归零且 tgtC 下降（验证型求助占比上升——verify 求助零成本
且通过加权投票影响 episode 结果，应成为主流）；
④ 最终目标：eval_acc > 0.863（v2.0 投票-only）且 > 同 ckpt 的 uniform 消融。

## 20. v3 首跑复盘：三项修复兑现、通道①被证伪、step 151 崩（2026-08-20）

跑 150 步后崩于 `ValueError: The decoder prompt (length 5036) is longer than
the maximum model length of 4096`。150 步的曲线本身有效，可用。

### 20.1 §18.3 三项工程修复全部兑现

| 修复 | 判据 | 实测 | 判决 |
|------|------|------|------|
| KL 锚到冻结 SFT 快照（§14） | step1 kl≈0 | 0.0000 → 0.0130 缓升 | ✅ |
| 零成本冷启动 | int_rate 不归零 | 全程 0.77–0.87（第四次没塌） | ✅ |
| 主 turn 双通道优势 | groups 不被 floor 丢弃、len 不塌 | 32/32 全程；len 133→251 | ✅ |

`acc_uniform` 0.823 → 0.863（step 90 峰）→ 0.850，**+4.0 点**是真实收益，
且已追平 v2.0 投票-only 基线 0.863。选择性仍未形成（sel 全程 −0.11~+0.04
震荡），但 eff 0.03→0.09 首次追上 qF——§18.5 判据③首次接近。

### 20.2 通道①不兑现（两轮实测，结论比初稿弱）

初稿基于 8/20 单轮写了「系统性有害，p≈0.001」。**8/23 的第二轮推翻了
这个强度表述**，此处以两轮并列为准（同 step 0–90 区间、各 10 次 eval）：

| 运行 | 负 | 正 | 零 | 均值 | 符号检验 |
|------|----|----|----|------|----------|
| 8/20（崩于 151） | 10 | 0 | 0 | −0.0126（≈−3.8 题/300） | p=0.002 |
| 8/23（v3_full） | 4 | 5 | 1 | −0.0039（≈−1.2 题/300） | **p=1.000** |
| 合并 | 14 | 5 | 1 | −0.0082 | p=0.064 |

第二轮符号完全对称、幅度 1.2 题/300，与噪声不可区分。因此成立的表述是
**加权投票不兑现**（两轮都无正收益，幅度上限 ±1.2 题），而不是「有害」。
「有害」是 8/20 单轮的特征，未在第二轮重现。

依然选 uniform，但依据改了：不是「避害」而是「无收益 + 更简单 + 一轮里明显更差」，
选错的代价近于零。

**两轮都扑不破的结论**（可进论文）：离线全覆盖测得的 verifier 分辨力
Δ=+0.312，**在策略自主决定何时调用的在线场景下不转化为聚合收益**。
聚合器分辨力 ≠ 聚合器可用性，中间隔着覆盖率：Δ 是 measure_channels.py 给
**每个**答案都打分时测的，episode 内 verifier 只被稀疏调用（约 32%），
未验证答案吃先验 0.5，大多数票权退化成等权——收益无处可出。

两轮差异的成因尚未定论（不当结论写）。已知差异项：8/23 跑 vLLM V1 引擎；
其 int_rate 平台在 0.64 而非 8/20 的 0.81。后者同时解释了 gate 轨迹差异（见 20.3），
但它与 d_vote 的关系无单调证据：8/23 的主动求助 verifier 比例
`int_rate×(1−tgtC)` 只有 0.19，反而低于 8/20 的 0.29，却没出现系统性负值。
这提示黑板上的分数主要由 ε 强制注入而非策略主动求助喂入（与 20.6 同根）。

### 20.3 step 151 崩溃链

```
parse 0.95→0.80（长 answer / reasoning 缺失时返回整个输出）
  ├→ 无界文本进 responder prompt 与黑板 → 随轮数累积 → 5036>4096 崩
  └→ 垃圾答案各占一票 → 加剧 20.2 的 d_vote 为负

tgtC 0.48→0.75（求助偏向 critic）
  → gate 解锁被 int_rate=0.78 遮蔽 → 黑板永无分数 → gate 拦停 10→193
  → 跑满 max_rounds → 加剧 prompt 膨胀，且压低 verifier 覆盖率
```

**闸门解锁被遮蔽**是这轮最值得记的教训。v2.0 的解锁代码写在
`elif`（proposer 未自发起）分支里：

```python
if u:                       # 自发起：用 proposer 自己选的 target（75% 是 critic）
    ...
elif self.max_hops > 0:     # 解锁逻辑只在这里
    if st["gate_blocked"]: forced, target = True, "verifier"
```

写它时 int_rate→0，所以 `elif` 总能进、解锁总能触发。v3 的零成本冷启动把
int_rate 救到 0.78，78% 的轮次改走 `if u:` → 解锁静默失效。
**前一个修复的成功造成了后一个修复的失效**——分支互斥的兜底逻辑，其可达性
依赖于另一个指标的取值，指标一变兜底就失明。

注：本节初稿曾把 prompt 膨胀归因于 FLAW 累积，错误。`to_text` 只取
`flaws[-1]` 且截断 80 字符，黑板的 flaw 部分不会膨胀。真凶是三个无界文本点
（见 20.4 第 1、2 条）。

### 20.4 v3.1 修正（六处）

1. **parse 硬上限**（parsing.py）：`MAX_ANSWER_CHARS=64`、
   `MAX_REASONING_CHARS=1500`。answer 超限时**回退抽末位数字而非截断**——
   截断会造出一个「看上去像答案」的假票污染投票池，宁可判解析失败。
2. **黑板展示层收敛**（blackboard.py）：`get_distinct_answers` 改
   `dict.fromkeys` 保序（`list(set)` 的顺序依赖字符串 hash，Python 默认
   随机化 → prompt 内容与 `max(key=)` 的平分 tie-break 跨进程不可复现）；
   新增 `_answers_for_display()` 滤空串、限长 64、只留最近 6 个。

   > **v3.2 第六轮更正（2026-08-27）**：上面这两条的取值与理由都已过时，原文
   > 保留作为病灶记录。① `MAX_ANSWER_CHARS` 已 64 → **192**：64 偏紧到会毁
   > 掉合法答案，实测 v2+v3 SFT 的 580 个 proposer turn 命中 1 次（一个 68 字
   > 的正确矩阵被抽成 `'4'`），且 `math_train_rl` 5265 题有 14 题 gold 本身
   > 就超 64（最长 159）。② 「宁可判解析失败」与代码不符：代码做的是
   > `nums[-1]`，即它自己批判的「造一张假票」；这个行为**仍然保留**，因为
   > 改成返回空串更坏——空串是投票池的合法键，加权模式下拿 0.5 先验，两票
   > 空串（1.0）会压过一票 verifier 背书的正确答案（0.9）。③ 展示层限长 64
   > 保持不变（它管 prompt 预算，与解析上限不是一个职责），但两处裸切片已改
   > 为带 `CLIP_MARK` 的 `clip_text`——放宽解析上限之后它们才第一次真正开始
   > 截断，而静默截断正是 v3.2 第四轮扫过的病灶。
3. **prompt 长度保险**（executor `make_prompt`，所有请求的唯一出口）：
   按 token 精确计数，超 `max_prompt_tokens` 取头尾各半、中段省略；
   首次触发打警告并计入 `n_prompt_clipped`。**不提 `max_model_len`**——
   有了 clip 计数，4096 是暴露新无界点的哨兵，抬到 8192 只是把问题推后。
4. **闸门解锁独立批次**（executor 3.5 步，`gate_unlock: true`）：想停但黑板
   无分数时，本轮末独立补一次 verifier 打分，**不占 hop 预算**。已确认
   `verifier_turns` 只用于 verifier 自己的校准奖励 `r = 1 − |score − 对错|`，
   与 `u`/`forced`/`r_int` 解耦 → 机制触发不污染因果归因，还顺带抬覆盖率。
   计数 `n_gate_unlocked` 是健康指标：**应随训练下降**（策略自己学会找 verifier）。
5. **加权投票回退**：`vote_mode: uniform`（20.2 证伪）。同时把消融改成
   **两路无条件计票**——原实现只在 `vote_mode == "weighted"` 时才真算 uniform，
   回退后 d_vote 会恒等于 0，重开 weighted 的判据就此失明。现 d_vote 恒为
   「weighted − uniform」，与生产 mode 无关；weighted 作影子指标持续记录。
6. **指标接入**（train.py）：step 行 `gate=N→M`、`clip_prompt=N`；
   wandb 加 `gate_unlocked` / `prompt_clipped` / `eval_accuracy_weighted`。

测试：新增 3 条（26→29 全过），其中
`test_gate_unlock_when_proposer_self_initiates_to_critic` **带消融对照**——
`gate_unlock=False` 时精确复现 v3 故障（跑满 4 轮、`stopped=False`、verifier
从未被调用），证明是该开关在起作用而非测试自证。

### 20.5 重跑判据（替代 §19.3）

| 读数 | 期望 | 异常含义 |
|------|------|----------|
| `gate=N→M` | M 接近 N | M 远小于 N → 解锁没生效 |
| `gate_unlocked` 趋势 | 随训练下降 | 不降 → 策略没学会主动找 verifier |
| `clip_prompt` | 不出现 | 出现且增长 → 还有文本无界点 |
| `parse` | 可能低于 0.95 | 正常：长答案现在被诚实判为失败 |
| `d_vote`（影子） | 转正后再考虑重开 weighted | 仍为负 → 覆盖率或污染没解决 |

重开 weighted 需两个前提同时满足：覆盖率上来（gate 解锁生效且计数下降）+
污染堵住（parse 稳、clip 不出现），然后 d_vote 连续几次 eval 转正。
在此之前 weighted 只作影子指标，不进生产路径。

### 20.6 三轮未解的主矛盾：选择性不成型，且完整系统未超基线

工程层已经健康，但核心假设连三轮没兑现：

| 读数 | v2.2 | 8/20 | 8/23（step 93） |
|------|------|------|------|
| `sel`（eff − qF，选择性） | 震荡 | −0.11~+0.01 | −0.14~+0.08 |
| `acc_uniform` 峰值 | — | 0.863 | 0.853 |
| v2.0 投票-only 基线 | 0.863 | 0.863 | 0.863 |

三轮下来完整系统一次也没稳定超过投票-only 基线。这不是工程问题，
§20.4 那六项修复一条都治不了它。

一个刺眼的旁证：8/23 轮的 gate 低（想停时黑板常有分数），但主动求助
verifier 的比例 `int_rate×(1−tgtC)` 只有 0.19，低于 8/20 的 0.29。两个指标
方向相反，最可能的解释是：**黑板上的 verifier 分数主要是 ε 随机强制注入
喂进去的，不是策略学会了主动去要。** 这与 `sel≈0`、`eff` 追不上 `qF`
是同一件事的三个侧面。
