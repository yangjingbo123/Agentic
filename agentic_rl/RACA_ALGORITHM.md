# RACA：Role-Aware Credit Assignment

> 本文件是当前实验的算法纲领，所有代码修改均以此为准。
> 最后更新：2026-07-19

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