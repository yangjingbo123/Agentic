你说得对，我之前的版本里低层agent的"交互"只是被动的信息广播，并没有把"与谁交互、交互什么"建模为agent的可学习动作。这不是真正的多智能体交互——真正的交互应该是agent主动选择向谁发送什么信息、是否采纳对方意见等。

让我重新从头构建，把交互本身作为低层agent的显式动作的一部分。

---

# HMARL-SC：面向大语言模型测试时推理的分层多智能体强化学习框架

## 最终完整版

---

## 1. 总论

### 1.1 核心问题

标准Self-Consistency（SC）以固定次数 $K$ 同质采样后多数投票，存在三个根本性缺陷：

1. **盲目采样**：不考虑已有证据，无法根据推理进展动态调整
2. **缺乏证据对抗**：所有采样链地位平等，没有机制主动发现和修正错误
3. **固定预算分配**：简单题和难题消耗相同资源，无法做到预算自适应

### 1.2 核心思想

将测试时多路径推理建模为一个**两层时间尺度的多智能体序贯决策问题**：

- **高层（宏动作时间尺度）**：一个Controller agent决定每一轮的**推理目标和模式**
- **低层（微动作时间尺度）**：三个功能分化的agent在每一轮内部通过**多步交互**协作完成推理目标

与已有工作的本质区别：低层agent之间的交互——包括信息传递、意见征询、证据质疑——被建模为**显式的可学习动作**，而非预设的固定流程。

### 1.3 层级结构

```
┌─────────────────────────────────────────────────────────┐
│  高层 Controller (Agent 0)                               │
│  宏动作时间尺度：每轮一个决策                              │
│  决策内容：本轮推理目标 + 预算分配 + 是否终止               │
├─────────────────────────────────────────────────────────┤
│  低层 三Agent协作体                                      │
│  微动作时间尺度：每轮内部多步交互                          │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Proposer  │  │  Critic  │  │ Verifier │              │
│  │ (Agent 1) │  │(Agent 2) │  │(Agent 3) │              │
│  │ 正向证据  │  │ 负向证据  │  │ 诊断证据  │              │
│  └─────┬────┘  └────┬─────┘  └────┬─────┘              │
│        │            │             │                     │
│        └──────── 交互动作 ────────┘                      │
│        (发送/请求/质疑/采纳/忽略)                         │
├─────────────────────────────────────────────────────────┤
│  共享信息板 (Shared Blackboard)                          │
│  所有agent的交互信息在此汇聚                              │
└─────────────────────────────────────────────────────────┘
```

### 1.4 关键设计选择

| 设计维度   | 选择               | 理由                     |
| ------ | ---------------- | ---------------------- |
| LLM权重  | Frozen           | 训练对象是调度策略，不是LLM本身      |
| 高层时间尺度 | SMDP             | 宏动作持续时间不固定             |
| 低层交互   | 显式动作             | 交互是可学习的策略，不是预设流程       |
| 低层参数   | 共享主干+角色头         | 样本效率，三角色共享"理解推理状态"的能力  |
| 状态特征   | 分层标注计算成本         | 明确方法边界，避免隐式oracle      |
| 奖励     | 全局为主 + 极简shaping | 避免proxy reward hacking |

---

## 2. 符号系统

| 符号                      | 类型  | 含义                                    |
| ----------------------- | --- | ------------------------------------- |
| $x$                     | 文本  | 输入问题                                  |
| $y^*$                   | 文本  | 真实答案（仅训练时可用）                          |
| $\hat{y}$               | 文本  | 系统最终输出                                |
| $t \in \{0,1,2,...\}$   | 整数  | 高层宏动作步数（轮次）                           |
| $k \in \{1,2,...,K_t\}$ | 整数  | 第 $t$ 轮内低层微动作步数                       |
| $K_t$                   | 整数  | 第 $t$ 轮的微动作总步数（由低层交互动态决定）             |
| $s_t$                   | 元组  | 第 $t$ 轮开始时的全局状态                       |
| $\omega_{t,k}$          | 元组  | 第 $t$ 轮第 $k$ 微步的低层局部状态                |
| $a_t^H$                 | 离散  | 高层宏动作                                 |
| $a_{t,k}^i$             | 离散  | agent $i$ 在第 $t$ 轮第 $k$ 微步的微动作（含交互动作） |
| $\mathcal{B}_t$         | 集合  | 第 $t$ 轮的共享信息板内容                       |
| $\mathcal{T}_t$         | 集合  | 已收集的推理轨迹                              |
| $\mathcal{V}_t$         | 集合  | 已收集的验证信号                              |
| $\mathcal{C}_t$         | 集合  | 已收集的批判信号                              |
| $p_t(y)$                | 分布  | 第 $t$ 轮的候选答案经验分布                      |
| $B, B_{\text{rem}}$     | 整数  | 总预算/剩余预算（token计）                      |

---

## 3. 全局形式化

### 3.1 两层MDP元组

$$\mathcal{M} = \langle \mathcal{M}^H, \mathcal{M}^L, \text{Interface} \rangle$$

**高层SMDP**：

$$\mathcal{M}^H = \langle \mathcal{S}^H, \mathcal{A}^H, \mathcal{P}^H, \mathcal{R}^H, \gamma^H \rangle$$

**低层多智能体MDP**（在每个宏动作内部）：

$$\mathcal{M}^L = \langle \Omega, \{1,2,3\}, \{\mathcal{A}^i\}_{i=1}^{3}, \mathcal{P}^L, \mathcal{R}^L, \gamma^L, \mathcal{B} \rangle$$

**Interface**（两层之间的耦合）：

- 高层动作 $a_t^H$ 作为低层MDP的**初始条件**
- 低层交互的最终产出作为高层的**状态转移输入**
- 低层MDP在满足终止条件后控制权返回高层

---

## 4. 高层Controller

### 4.1 高层全局状态 $s_t$

$$s_t = (x, \mathcal{T}_t, p_t, \mathcal{V}_t, \mathcal{C}_t, b_t)$$

各分量定义：

**推理轨迹集合**：

$$\mathcal{T}_t = \{(\tau_j, y_j, \text{src}_j)\}_{j=1}^{n_t}$$

- $\tau_j$：推理链文本（或其embedding）
- $y_j$：候选答案
- $\text{src}_j \in \{\text{init}, \text{refine}, \text{counter}\}$：来源标记（初始生成/修补/反驳后新生成）

**候选答案分布**：

$$p_t(y) = \frac{\sum_{j=1}^{n_t} \mathbb{1}[y_j=y]}{n_t}$$

**验证信号集合**：

$$\mathcal{V}_t = \{(y_k^{\text{ver}}, \text{score}_k)\}_k, \quad \text{score}_k \in [0,1]$$

**批判信号集合**：

$$\mathcal{C}_t = \{(\text{target}_m, \text{flaw}_m, \text{survived}_m)\}_m$$

- $\text{target}_m$：被批判的答案
- $\text{flaw}_m$：发现的缺陷（文本摘要或类别标签）
- $\text{survived}_m \in \{0,1\}$：该批判是否"成立"（由后续验证或反驳判定）

**预算状态**：

$$b_t = (B_{\text{rem}}, c_{\text{cum}}, t)$$

### 4.2 高层状态特征化 $\phi(s_t)$

将 $s_t$ 压缩为定长向量。**严格分层标注计算成本**：

#### 零成本层（直接计数/统计，无LLM调用）

| 符号                  | 定义                          | 语义            |
| ------------------- | --------------------------- | ------------- |
| $n_t$               | $                           | \mathcal{T}_t |
| $p_t^{(1)}$         | $\max_y p_t(y)$             | 领先答案支持率       |
| $\Delta_{12}(t)$    | $p_t^{(1)} - p_t^{(2)}$     | top-1与top-2差距 |
| $H_t$               | $-\sum_y p_t(y)\log p_t(y)$ | 答案分布熵         |
| $K_t^{\text{dist}}$ | 不同答案的个数                     | 答案多样性         |
| $n_t^{\text{ver}}$  | $                           | \mathcal{V}_t |
| $n_t^{\text{crit}}$ | $                           | \mathcal{C}_t |
| $n_t^{\text{surv}}$ | $\sum_m \text{survived}_m$  | 成立的批判数        |
| $\rho_t^B$          | $c_{\text{cum}}/B$          | 预算消耗比例        |
| $t$                 | 当前轮次                        | 进度            |

#### 低成本层（利用已有Verifier结果，无额外调用）

| 符号                           | 定义                                               | 语义        |
| ---------------------------- | ------------------------------------------------ | --------- |
| $\bar{v}_t^{(1)}$            | 领先答案的平均验证分                                       | 领先答案的诊断信心 |
| $\bar{v}_t^{(2)}$            | 第二答案的平均验证分                                       | 竞争答案的诊断信心 |
| $\text{cov}_t^{\text{ver}}$  | 被验证过的distinct答案比例                                | 验证覆盖度     |
| $\text{cov}_t^{\text{crit}}$ | 被批判过的distinct答案比例                                | 批判覆盖度     |
| $f_t^{\text{surv}}$          | $n_t^{\text{surv}} / \max(n_t^{\text{crit}}, 1)$ | 批判成立率     |

#### 一次性成本层（问题编码，Episode开始时计算一次）

| 符号                              | 定义             | 语义        |
| ------------------------------- | -------------- | --------- |
| $\mathbf{e}_x \in \mathbb{R}^d$ | 问题文本的embedding | 问题难度/类型信号 |

**完整特征向量**：

$$\phi(s_t) = \Big[\underbrace{n_t, p_t^{(1)}, \Delta_{12}, H_t, K_t^{\text{dist}}}_{\text{答案分布（零成本）}},\ \underbrace{n_t^{\text{ver}}, n_t^{\text{crit}}, n_t^{\text{surv}}}_{\text{活动计数（零成本）}},\ \underbrace{\bar{v}_t^{(1)}, \bar{v}_t^{(2)}, \text{cov}_t^{\text{ver}}, \text{cov}_t^{\text{crit}}, f_t^{\text{surv}}}_{\text{诊断统计（低成本）}},\ \underbrace{\rho_t^B, t}_{\text{预算进度}},\ \underbrace{\mathbf{e}_x}_{\text{问题编码}}\Big]$$

维度：$15 + d$（$d$ 为embedding维度）。

### 4.3 高层动作空间 $\mathcal{A}^H$

高层每步选择一个宏动作，定义本轮低层多agent协作的**目标和约束**：

$$a_t^H = \begin{cases} (\text{goal}_t, \text{focus}_t, \delta B_t) & \text{推理动作} \\ \texttt{STOP}(\text{agg}) & \text{终止动作} \end{cases}$$

#### 4.3.1 推理动作的三个分量

**goal（本轮推理目标）**：

| goal                 | 语义       | 低层内部的主要工作               |
| -------------------- | -------- | ----------------------- |
| $\texttt{EXPLORE}$   | 生成新的候选证据 | Proposer为主，其他agent辅助评估  |
| $\texttt{CHALLENGE}$ | 挑战现有某个假设 | Critic为主，Proposer可能修补   |
| $\texttt{DIAGNOSE}$  | 深度验证某个假设 | Verifier为主，Critic辅助交叉检验 |

**focus（目标焦点）**：

| 当goal为    | focus取值             | 语义                |
| --------- | ------------------- | ----------------- |
| EXPLORE   | $\texttt{open}$     | 自由探索，不限方向         |
| EXPLORE   | $\texttt{minority}$ | 为少数派答案寻找支持        |
| CHALLENGE | $y^{(k)} \in$ 答案集   | 攻击目标答案（pointer选择） |
| DIAGNOSE  | $y^{(k)} \in$ 答案集   | 验证目标答案（pointer选择） |

**$\delta B_t$（本轮预算上限）**：

$$\delta B_t \in \{\texttt{light}, \texttt{standard}, \texttt{heavy}\}$$

对应本轮最多允许消耗的token预算比例：

| 等级                  | 预算上限            | 含义        |
| ------------------- | --------------- | --------- |
| $\texttt{light}$    | $0.05 \times B$ | 快速轻量轮次    |
| $\texttt{standard}$ | $0.15 \times B$ | 标准轮次      |
| $\texttt{heavy}$    | $0.30 \times B$ | 深度探索/验证轮次 |

> **预算分配是高层动作的一部分**：这使高层不仅决定"做什么"，还决定"投入多少资源"。低层agent的gate决策受此约束——当轮内预算耗尽时，剩余微步强制skip。

#### 4.3.2 终止动作

$$\texttt{STOP}(\text{agg}), \quad \text{agg} \in \{\texttt{majority}, \texttt{weighted}\}$$

#### 4.3.3 动作空间大小

$$|\mathcal{A}^H| = \underbrace{2}_{\text{EXPLORE}} + \underbrace{|\mathcal{Y}_t|}_{\text{CHALLENGE targets}} + \underbrace{|\mathcal{Y}_t|}_{\text{DIAGNOSE targets}}) \times \underbrace{3}_{\delta B} + \underbrace{2}_{\text{STOP}} $$

典型规模：当有3-5个distinct答案时，约 $(2+4+4) \times 3 + 2 = 32$ 个动作。PPO完全可处理。

### 4.4 高层策略网络

分解式输出：

$$\pi_H(a_t^H | \phi(s_t); \theta_H) = \pi_H^{\text{goal}}(\text{goal}_t | \phi(s_t)) \cdot \pi_H^{\text{focus}}(\text{focus}_t | \text{goal}_t, \phi(s_t)) \cdot \pi_H^{\text{budget}}(\delta B_t | \text{goal}_t, \phi(s_t))$$

- $\pi_H^{\text{goal}}$：5-way softmax（EXPLORE-open, EXPLORE-minority, CHALLENGE, DIAGNOSE, STOP）
- $\pi_H^{\text{focus}}$：当goal需要target时，pointer attention在当前distinct答案集上选择
- $\pi_H^{\text{budget}}$：3-way softmax

```
网络结构：

Input: φ(s_t) ∈ R^{15+d}
       ↓
SharedTrunk: Linear(15+d, 256) → ReLU → Linear(256, 256) → ReLU
       ↓
       ├→ GoalHead: Linear(256, 5) → softmax     // EXPLORE-open, EXPLORE-min, CHALLENGE, DIAGNOSE, STOP
       ├→ FocusHead: PointerAttention(256, answer_embeddings)  // 在distinct answers上选
       └→ BudgetHead: Linear(256, 3) → softmax   // light, standard, heavy
```

---

## 5. 低层多智能体交互系统

### 5.1 核心理念

低层三个agent在每个宏动作内部运行一个**有限步数的多智能体MDP**。每个微步中，**所有三个agent同时选择动作**（包括"做什么推理工作"和"与谁怎样交互"），然后同时执行，结果写入共享信息板。

```
宏动作第t轮内部时间线：

    微步1          微步2          微步3        ...    微步K_t
  ┌─────────┐   ┌─────────┐   ┌─────────┐        ┌─────────┐
  │P: a_{1,1}│   │P: a_{1,2}│   │P: a_{1,3}│        │P: a_{1,K}│
  │C: a_{2,1}│   │C: a_{2,2}│   │C: a_{2,3}│  ...   │C: a_{2,K}│
  │V: a_{3,1}│   │V: a_{3,2}│   │V: a_{3,3}│        │V: a_{3,K}│
  └────┬────┘   └────┬────┘   └────┬────┘        └────┬────┘
       ↓              ↓              ↓                  ↓
    信息板更新      信息板更新      信息板更新          信息板更新
       ↓              ↓              ↓                  ↓
      ω_{t,2}       ω_{t,3}       ω_{t,4}         宏动作产出
```

### 5.2 共享信息板 $\mathcal{B}_{t,k}$

信息板是低层agent交互的**中介**。它不是简单的结果缓存，而是一个结构化的**消息空间**，记录了所有agent已发出的消息和工作产出。

$$\mathcal{B}_{t,k} = \{\text{msg}_1, \text{msg}_2, ..., \text{msg}_{|\mathcal{B}_{t,k}|}\}$$

每条消息的格式：

$$\text{msg} = (\text{sender},\ \text{type},\ \text{target\_agent},\ \text{content},\ \text{step})$$

| 字段            | 取值                 | 含义            |
| ------------- | ------------------ | ------------- |
| sender        | $\in \{1,2,3\}$    | 发送者agent      |
| type          | 见下表                | 消息类型          |
| target\_agent | $\in \{0, 1,2,3\}$ | 接收者（0=广播给所有人） |
| content       | 结构化数据              | 消息内容          |
| step          | $k$                | 发送时的微步        |

**消息类型**：

| type                 | 语义            | 典型sender | 典型content                                    |
| -------------------- | ------------- | -------- | -------------------------------------------- |
| $\texttt{TRACE}$     | 提交一条新推理链      | Proposer | $(\tau, y)$                                  |
| $\texttt{SCORE}$     | 提交一个验证评分      | Verifier | $(y, \text{score}, \text{rationale})$        |
| $\texttt{FLAW}$      | 提交一个缺陷报告      | Critic   | $(y_{\text{target}}, \text{flaw\_desc})$     |
| $\texttt{REQUEST}$   | 请求某agent执行某任务 | 任意       | $(\text{requested\_action}, \text{reason})$  |
| $\texttt{CHALLENGE}$ | 质疑某条已有消息      | 任意       | $(\text{msg\_id}, \text{challenge\_reason})$ |
| $\texttt{ENDORSE}$   | 支持/采纳某条已有消息   | 任意       | $(\text{msg\_id})$                           |
| $\texttt{IDLE}$      | 本步无操作         | 任意       | $\varnothing$                                |

> **信息板的设计理由**：直接的agent-to-agent通信在3-agent系统中会产生 $3 \times 2 = 6$ 条通信通道，且消息格式难以统一。信息板模式将所有交互统一为"向公共空间写消息"，简化了通信拓扑。`target_agent`字段允许定向消息（如Critic专门请求Verifier检查某条链），`target_agent=0`则为广播。

### 5.3 低层局部状态 $\omega_{t,k}$

第 $t$ 轮第 $k$ 微步的低层公共局部状态：

$$\omega_{t,k} = (\text{goal}_t,\ \text{focus}_t,\ \delta B_t^{\text{rem}},\ k,\ \mathcal{B}_{t,k})$$

- goal, focus：来自高层宏动作
- $\delta B_t^{\text{rem}}$：本轮剩余预算
- $k$：当前微步
- $\mathcal{B}_{t,k}$：信息板当前内容

### 5.4 低层agent观测 $o_{t,k}^i$

Agent $i$ 在微步 $k$ 的观测：

$$o_{t,k}^i = (\text{role}=i,\ \omega_{t,k},\ h_{t,k}^i)$$

其中 $h_{t,k}^i$ 是agent $i$ 的**角色特定附加信息**：

| Agent        | $h_{t,k}^i$ 的内容                                        |
| ------------ | ------------------------------------------------------ |
| Proposer (1) | 已有轨迹摘要、当前答案分布 $p_t$、收到的REQUEST消息列表                     |
| Critic (2)   | 被focus指定的答案的支持链列表、收到的REQUEST消息列表、已发出的FLAW被CHALLENGE的记录 |
| Verifier (3) | 待验证答案的支持链列表、收到的REQUEST消息列表、已发出的SCORE被CHALLENGE的记录      |

### 5.5 低层微动作空间 $\mathcal{A}^i$

每个agent的微动作由两个分量组成：

$$a_{t,k}^i = (\text{work}_{t,k}^i,\ \text{comm}_{t,k}^i)$$

- $\text{work}$：推理工作动作（可能触发LLM调用）
- $\text{comm}$：通信/交互动作（写入信息板）

**一个微步中agent同时选择work和comm**。

#### 5.5.1 Proposer (Agent 1) 的动作空间

**Work动作**：

| work                                       | 语义                  | 是否调用LLM | token成本 |
| ------------------------------------------ | ------------------- | ------- | ------- |
| $\texttt{generate}$                        | 生成一条新推理链            | 是       | ~500    |
| $\texttt{generate-diverse}$                | 高温/换模板生成            | 是       | ~500    |
| $\texttt{refine}(\text{msg\_id})$          | 基于信息板上某条FLAW消息修补对应链 | 是       | ~400    |
| $\texttt{counter-propose}(\text{msg\_id})$ | 针对某条FLAW提出替代方案      | 是       | ~500    |
| $\texttt{work-idle}$                       | 不执行推理工作             | 否       | 0       |

**Comm动作**：

| comm                                 | 语义                | 写入信息板的消息type |
| ------------------------------------ | ----------------- | ------------ |
| $\texttt{submit-trace}$              | 提交work产出的链到信息板    | TRACE        |
| $\texttt{request}(j, \text{task})$   | 请求agent $j$ 执行某任务 | REQUEST      |
| $\texttt{challenge}(\text{msg\_id})$ | 质疑信息板上某条消息        | CHALLENGE    |
| $\texttt{endorse}(\text{msg\_id})$   | 支持信息板上某条消息        | ENDORSE      |
| $\texttt{comm-idle}$                 | 不发送消息             | IDLE         |

> **约束**：$\texttt{submit-trace}$ 只在本步有work产出时才有效。$\texttt{refine}$和$\texttt{counter-propose}$需要指定信息板上一条FLAW消息的ID。

#### 5.5.2 Critic (Agent 2) 的动作空间

**Work动作**：

| work                                        | 语义                  | 是否调用LLM | token成本 |
| ------------------------------------------- | ------------------- | ------- | ------- |
| $\texttt{critique-logic}(\text{trace\_id})$ | 对指定链做逻辑分析           | 是       | ~300    |
| $\texttt{find-counterexample}(y)$           | 对指定答案构造反例           | 是       | ~400    |
| $\texttt{cross-examine}(\text{msg\_id})$    | 对信息板上某条SCORE消息做交叉检验 | 是       | ~300    |
| $\texttt{work-idle}$                        | 不执行批判工作             | 否       | 0       |

**Comm动作**：

| comm                                 | 语义              | 写入信息板的消息type |
| ------------------------------------ | --------------- | ------------ |
| $\texttt{submit-flaw}$               | 提交发现的缺陷         | FLAW         |
| $\texttt{request}(j, \text{task})$   | 请求agent $j$ 做某事 | REQUEST      |
| $\texttt{challenge}(\text{msg\_id})$ | 质疑某条消息          | CHALLENGE    |
| $\texttt{endorse}(\text{msg\_id})$   | 支持某条消息          | ENDORSE      |
| $\texttt{comm-idle}$                 | 不发送消息           | IDLE         |

#### 5.5.3 Verifier (Agent 3) 的动作空间

**Work动作**：

| work                                     | 语义        | 是否调用LLM | token成本 |
| ---------------------------------------- | --------- | ------- | ------- |
| $\texttt{quick-verify}(y)$               | 对指定答案快速评分 | 是       | ~200    |
| $\texttt{step-verify}(\text{trace\_id})$ | 对指定链逐步核验  | 是       | ~400    |
| $\texttt{work-idle}$                     | 不执行验证工作   | 否       | 0       |

**Comm动作**：

| comm                                 | 语义              | 写入信息板的消息type |
| ------------------------------------ | --------------- | ------------ |
| $\texttt{submit-score}$              | 提交验证评分          | SCORE        |
| $\texttt{request}(j, \text{task})$   | 请求agent $j$ 做某事 | REQUEST      |
| $\texttt{challenge}(\text{msg\_id})$ | 质疑某条消息          | CHALLENGE    |
| $\texttt{endorse}(\text{msg\_id})$   | 支持某条消息          | ENDORSE      |
| $\texttt{comm-idle}$                 | 不发送消息           | IDLE         |

#### 5.5.4 动作空间复杂度分析

每个agent每微步的动作 = work × comm。

work动作数：$\sim$3-5（含idle），其中部分需要pointer（选择msg\_id或trace\_id或answer）。
comm动作数：$\sim$5（含idle），其中部分需要pointer（选择msg\_id或agent\_id）。

组合空间 $\sim 5 \times 5 = 25$ 每agent，可通过分解式输出避免组合：

$$\pi_i(a_{t,k}^i | o_{t,k}^i; \theta_{\text{low}}) = \pi_i^{\text{work}}(\text{work} | o_{t,k}^i) \cdot \pi_i^{\text{comm}}(\text{comm} | o_{t,k}^i, \text{work})$$

### 5.6 低层策略网络（共享主干 + 角色头）

```
Input: [ω_{t,k}, role_embedding_i, h_{t,k}^i, blackboard_encoding]
                  ↓
SharedTrunk:  Linear → ReLU → Linear → ReLU        (shared θ_low)
                  ↓
         ┌────────┼────────┐
         ↓        ↓        ↓
      Role=1    Role=2    Role=3
      头组       头组       头组
    ┌──────┐  ┌──────┐  ┌──────┐
    │WorkHd│  │WorkHd│  │WorkHd│     // 各角色独立的work动作头
    │CommHd│  │CommHd│  │CommHd│     // 各角色独立的comm动作头
    │PtrHd │  │PtrHd │  │PtrHd │     // 各角色的pointer头（选msg_id/trace_id等）
    └──────┘  └──────┘  └──────┘
```

**信息板编码** $\text{blackboard\_encoding}$：

$$\text{bb\_enc} = \text{Attention}\Big(\text{query}=\text{role\_state},\ \text{keys}=\{\text{msg\_enc}(\text{msg}_j)\}_{j=1}^{|\mathcal{B}_{t,k}|}\Big)$$

每条消息编码为：

$$\text{msg\_enc}(\text{msg}) = \text{MLP}([\text{sender\_emb}, \text{type\_emb}, \text{target\_emb}, \text{content\_emb}, \text{step\_emb}])$$

Agent通过attention机制"阅读"信息板，注意力权重自然反映了它关注哪些消息。

### 5.7 低层微步转移

每微步中三个agent**同时**选择动作，然后：

1. **Work执行**：所有选择了非idle work动作的agent同时调用LLM（可并行）
2. **Comm执行**：所有agent的comm动作写入信息板

$$\mathcal{B}_{t,k+1} = \mathcal{B}_{t,k} \cup \bigcup_{i=1}^{3} \text{ToMessage}(\text{comm}_{t,k}^i, \text{work\_result}_{t,k}^i)$$

3. **预算更新**：

$$\delta B_t^{\text{rem}} \leftarrow \delta B_t^{\text{rem}} - \sum_{i=1}^{3} \text{cost}(\text{work}_{t,k}^i)$$

### 5.8 低层轮内终止条件

一个宏动作（一轮低层交互）在以下任一条件满足时终止：

1. **达到最大微步数** $K_{\max}$：硬上限，防止低层交互无限循环。$K_{\max}$ 随 $\delta B_t$ 设定：light→3, standard→5, heavy→8
2. **轮内预算耗尽**：$\delta B_t^{\text{rem}} \leq 0$
3. **全员idle**：三个agent在同一微步都选择了work-idle且comm-idle，表示它们认为本轮工作已完成
4. **全员endorse同一消息**：三个agent都endorse了信息板上同一条消息，表示达成共识

> **终止条件3和4是低层agents通过动作选择隐式"投票"终止的机制**。这使得轮内交互的步数也是学习得到的，而非固定值。

### 5.9 宏动作产出

一轮低层交互结束后，从信息板 $\mathcal{B}_{t,K_t}$ 中提取结构化产出：

$$\text{macro\_output}_t = \text{Extract}(\mathcal{B}_{t,K_t})$$

$$\text{new\_traces} = \{(\tau, y) : \exists\ \texttt{TRACE}(\tau, y) \in \mathcal{B}_{t,K_t}\}$$

$$\text{new\_scores} = \{(y, s) : \exists\ \texttt{SCORE}(y, s, \_) \in \mathcal{B}_{t,K_t}\}$$

$$\text{new\_critiques} = \{(y, f, \text{surv}) : \exists\ \texttt{FLAW}(y, f) \in \mathcal{B}_{t,K_t}\}$$

其中 $\text{surv}$（survived）的判定：

$$\text{surv}(m) = \begin{cases} 0 & \text{if } \exists\ \texttt{CHALLENGE}(\text{msg\_id}=m) \in \mathcal{B}_{t,K_t} \text{ 且无后续ENDORSE} \\ 1 & \text{otherwise} \end{cases}$$

> **survived机制**：Critic发出的FLAW可以被其他agent CHALLENGE。如果一条FLAW被CHALLENGE且没有被第三方ENDORSE，则标记为"未成立"（survived=0）。这让系统自然地过滤掉"瞎挑刺"——如果Critic的批判经不起其他agent的检验，它不会影响全局状态。

---

## 6. 全局状态转移

高层第 $t$ 步到第 $t+1$ 步：

$$s_{t+1} = \mathcal{P}^H(s_t, a_t^H, \text{macro\_output}_t)$$

$$\mathcal{T}_{t+1} = \mathcal{T}_t \cup \text{new\_traces}$$
$$\mathcal{V}_{t+1} = \mathcal{V}_t \cup \text{new\_scores}$$
$$\mathcal{C}_{t+1} = \mathcal{C}_t \cup \text{new\_critiques}$$
$$b_{t+1} = (B_{\text{rem}} - \text{total\_cost}_t,\ c_{\text{cum}} + \text{total\_cost}_t,\ t+1)$$

其中 $\text{total\_cost}_t = \sum_{k=1}^{K_t}\sum_{i=1}^{3} \text{cost}(\text{work}_{t,k}^i)$

### 6.1 全局终止与聚合

**终止条件**：

1. 高层选择 $\texttt{STOP}(\text{agg})$
2. $B_{\text{rem}} \leq \text{min\_cost\_estimate}$：全局预算耗尽
3. $t \geq T_{\max}$：达到最大轮次

**聚合方式**：

**majority**：

$$\hat{y} = \arg\max_y\ p_T(y)$$

**weighted**：

$$\hat{y} = \arg\max_y\ w(y)$$

$$w(y) = p_T(y) \cdot \bar{v}(y) \cdot (1 - \text{surv\_attack}(y))$$

其中：

- $\bar{v}(y)$：答案 $y$ 的平均验证分。未被验证的答案取 $\bar{v}(y) = 0.5$
- $\text{surv\_attack}(y)$：针对答案 $y$ 的"成立攻击"比例：

$$\text{surv\_attack}(y) = \frac{|\{m \in \mathcal{C}_T : m.\text{target}=y \land m.\text{survived}=1\}|}{\max(|\{m \in \mathcal{C}_T : m.\text{target}=y\}|, 1)}$$

> **weighted聚合综合了三种证据**：频率（$p_T$）代表正向支持、验证分（$\bar{v}$）代表诊断确认、成立攻击率（$\text{surv\_attack}$）代表负向证伪。一个答案要获得高权重，需要同时具备高支持率、高验证分、低成立攻击率。

### 6.2 预算定义

$$\text{cost} = \text{input\_tokens} + \text{output\_tokens}$$

$$B = K_{\text{equiv}} \times \bar{c}_{\text{single}}$$

$K_{\text{equiv}}$ 是等价SC采样次数，$\bar{c}_{\text{single}}$ 是单次标准CoT的平均token数。所有实验以 $K_{\text{equiv}}$ 为横轴报告，使结果可与SC-$K$ 直接比较。

---

## 7. 奖励函数

### 7.1 设计原则

1. **全局奖励为主**：正确性信号只来自 $\mathbb{1}[\hat{y}=y^*]$
2. **步级shaping极简**：只惩罚成本，不构造正向proxy
3. **低层唯一例外**：Verifier的校准奖励（因为有无歧义的局部ground truth）
4. **交互行为不设独立奖励**：交互动作的价值通过全局奖励间接学习

### 7.2 全局奖励（Episode结束时，所有agent共享）

$$R^{\text{global}} = \mathbb{1}[\hat{y}=y^*] - \alpha \cdot \rho_T^B$$

### 7.3 高层步奖励

$$r_t^H = -\beta^H \cdot \text{total\_cost}_t$$

> 每个宏动作的成本 = 该轮内所有LLM调用的token总和。高层通过学习在"继续投入资源探索"和"及时停止节省预算"之间权衡。

### 7.4 低层微步奖励

**所有agent共享的微步奖励**：

$$r_{t,k}^{\text{low}} = -\beta^L \cdot \sum_{i=1}^{3} \text{cost}(\text{work}_{t,k}^i)$$

微步级成本惩罚，鼓励低层agent学会在不需要时选择idle。

**Verifier额外辅助奖励**（仅训练时使用 $y^*$）：

$$r_{t,k}^{\text{Ver-aux}} = \beta^V \cdot (1 - |\text{score}_k - \mathbb{1}[y_k^{\text{ver}}=y^*]|) \quad \text{（仅当Verifier在微步$k$执行了verify动作时）}$$

**其他agent无额外奖励。Critic和Proposer的行为完全由全局奖励塑造。**

### 7.5 总回报

**高层**：

$$G^H = \sum_{t=0}^{T-1} (\gamma^H)^t \cdot r_t^H + R^{\text{global}}$$

**低层agent $i$ 在第 $t$ 轮**：

$$G_{t}^{i} = \sum_{k=1}^{K_t} (\gamma^L)^{k-1} \cdot r_{t,k}^{i} + \eta \cdot R^{\text{global}}$$

其中 $r_{t,k}^{i} = r_{t,k}^{\text{low}} + \mathbb{1}[i=3] \cdot r_{t,k}^{\text{Ver-aux}}$

> **两层折扣因子**：$\gamma^H$ 用于跨轮折扣（高层），$\gamma^L$ 用于轮内微步折扣（低层）。$\eta$ 控制全局奖励对低层的影响权重。

---

## 8. 训练框架

### 8.1 CTDE范式

| 阶段     | 高层信息                                                                   | 低层信息                                                     |
| ------ | ---------------------------------------------------------------------- | -------------------------------------------------------- |
| **训练** | $\phi(s_t)$ 用于Policy和Value；$y^*$ 用于 $R^{\text{global}}$ 和 Verifier辅助奖励 | $o_{t,k}^i$ 用于Policy；全局状态用于集中式Value；$y^*$ 用于Verifier辅助奖励 |
| **执行** | $\phi(s_t)$，无 $y^*$                                                    | $o_{t,k}^i$，无 $y^*$，Verifier仅输出score不参考答案                |

### 8.2 价值网络

**三套独立的Value网络**：

| 网络                                                            | 输入                     | 用途                      |
| ------------------------------------------------------------- | ---------------------- | ----------------------- |
| $V_{\psi_H}^H(\phi(s_t))$                                     | 高层全局状态特征               | 高层GAE advantage估计       |
| $V_{\psi_L}^L(o_{t,k}^{\text{joint}})$                        | 低层联合观测（所有agent观测+全局信息） | 低层集中式Critic的advantage估计 |
| $V_{\psi_{\text{mix}}}(\phi(s_t), \text{round\_summary}_{t})$ | 混合信息                   | 连接高低层的辅助Value估计（可选）     |

> **集中式低层Critic**：训练时，低层Value网络可以观测所有agent的联合观测 $o_{t,k}^{\text{joint}} = (o_{t,k}^1, o_{t,k}^2, o_{t,k}^3)$，用于更准确的advantage估计。执行时仅需要各agent的Policy网络。

### 8.3 训练Pipeline

```
================================================================
Stage 0: 探索性数据收集 + 行为克隆初始化
================================================================

0a. 规则策略库

  高层规则（随机选择一条执行）：

    Rule-Simple:
      EXPLORE(open, standard) ×3 → STOP(majority)

    Rule-Verify:
      EXPLORE(open, standard) ×2 → DIAGNOSE(top-1, standard) → STOP(weighted)

    Rule-Challenge:
      EXPLORE(open, standard) ×2 → CHALLENGE(top-1, standard) 
        → EXPLORE(open, light) → STOP(weighted)

    Rule-Full:
      EXPLORE(open, standard) → DIAGNOSE(top-1, light) 
        → CHALLENGE(top-1, standard) → EXPLORE(minority, standard) 
        → DIAGNOSE(top-1, light) → STOP(weighted)

  低层规则（在各goal下的固定行为）：

    EXPLORE轮内：
      微步1: P=generate, C=work-idle, V=work-idle
              P→submit-trace, C→comm-idle, V→comm-idle
      微步2: P=work-idle, C=work-idle, V=quick-verify(new_answer)
              P→comm-idle, C→comm-idle, V→submit-score
      微步3: 全员idle → 触发终止

    CHALLENGE轮内：
      微步1: P=work-idle, C=critique-logic(focus_chain), V=work-idle
              P→comm-idle, C→submit-flaw, V→comm-idle
      微步2: P=refine(flaw_msg) if severity>low else work-idle, C=work-idle, V=work-idle
              P→submit-trace if worked, C→comm-idle, V→comm-idle
      微步3: P=work-idle, C=work-idle, V=quick-verify(affected_answer)
              全员submit/idle
      微步4: 全员idle → 终止

    DIAGNOSE轮内：
      微步1: P=work-idle, C=work-idle, V=step-verify(focus_answer)
              V→submit-score
      微步2: P=work-idle, C=cross-examine(score_msg), V=work-idle
              C→submit-flaw or comm-idle
      微步3: 全员idle → 终止

0b. 数据收集

  对每个训练问题 x：
    随机选择高层规则
    执行完整episode
    记录完整trajectory：
      高层: {(φ(s_t), a_t^H, r_t^H)}_t ∪ R_global
      低层: {(o_{t,k}^i, a_{t,k}^i, r_{t,k}^i)}_{t,k,i} ∪ R_global

  收集 ~5000 episodes

0c. 行为克隆

  用成功episode（R_global > 0）做BC：
    - 初始化 θ_H（高层策略）
    - 初始化 θ_low（低层共享策略）
    - 初始化 V_ψ_H, V_ψ_L（价值网络，用trajectory回报做回归）


================================================================
Stage 1: 交替冻结训练
================================================================

  超参数：K_rounds=5, N_A=2000 steps, N_B=2000 steps
  学习率：lr_policy=3e-4, lr_value=1e-3
  PPO参数：clip_ε=0.1, λ_GAE=0.95

  for round = 1 to K_rounds do

    ──────── Phase A: 训练低层，冻结高层 ────────

    θ_H frozen
    for step = 1 to N_A do
      收集batch of episodes（高层用当前frozen θ_H）

      对低层trajectory：
        计算低层GAE advantage（使用V_ψ_L）
        PPO更新 θ_low
        回归更新 V_ψ_L

      监控：
        - 各角色的work-idle比例（不应全为idle或全为active）
        - 各角色的comm类型分布（不应退化为全idle）
        - Verifier score的分布（不应退化为常数）
    end for

    ──────── Phase B: 训练高层，冻结低层 ────────

    θ_low frozen
    for step = 1 to N_B do
      收集batch of episodes（低层用当前frozen θ_low）

      对高层trajectory：
        计算高层GAE advantage（使用V_ψ_H）
        PPO更新 θ_H
        回归更新 V_ψ_H

      监控：
        - 各goal的选择比例
        - STOP时机的分布
        - 预算使用曲线
    end for

    ──────── 衰减交替间隔 ────────
    N_A = max(N_A × 0.8, 500)
    N_B = max(N_B × 0.8, 500)

  end for


================================================================
Stage 2: 联合微调
================================================================

  所有参数同时更新
  学习率降至 Stage 1 的 1/3

  for step = 1 to N_joint do
    收集batch of episodes

    同时更新：θ_H, θ_low, V_ψ_H, V_ψ_L

    退化监控（每100步检查一次）：

      CHECK 1: 角色活跃度
        若任一角色的 work-idle 比例 > 95%（几乎从不工作）
        → 增加该角色的entropy bonus系数 ×2

      CHECK 2: 交互活跃度
        若所有agent的 comm-idle 比例 > 90%（几乎不交互）
        → 增加comm动作的entropy bonus

      CHECK 3: 角色区分度
        计算三个角色的work action分布的pairwise KL散度
        若平均KL < 0.1（角色行为趋同）
        → 增加角色特化的辅助loss

      CHECK 4: Verifier退化
        若Verifier score的标准差 < 0.05（退化为常数输出）
        → 增加Verifier辅助奖励系数 β^V ×2

  end for


================================================================
Stage 3: 评估
================================================================

  不使用y*
  高层用 π_H(θ_H) 贪心或低温采样
  低层用 π_low(θ_low) 贪心或低温采样
  Verifier仅基于输入问题和推理链输出score
```

### 8.4 PPO具体公式

#### 高层PPO

$$\mathcal{L}^H(\theta_H) = \mathbb{E}\Big[\min\big(\rho_t^H \hat{A}_t^H,\ \text{clip}(\rho_t^H, 1-\epsilon, 1+\epsilon)\hat{A}_t^H\big)\Big] + \lambda_{\text{ent}}^H \cdot \mathcal{H}[\pi_H]$$

$$\hat{A}_t^H = \sum_{l=0}^{T-t-1} (\gamma^H \lambda)^l \delta_{t+l}^H, \quad \delta_t^H = r_t^H + \gamma^H V_{\psi_H}(s_{t+1}) - V_{\psi_H}(s_t)$$

#### 低层PPO

$$\mathcal{L}^L(\theta_{\text{low}}) = \sum_{i=1}^{3} \mathbb{E}\Big[\min\big(\rho_{t,k}^i \hat{A}_{t,k}^i,\ \text{clip}(\rho_{t,k}^i, 1-\epsilon, 1+\epsilon)\hat{A}_{t,k}^i\big)\Big] + \sum_{i=1}^{3} \lambda_{\text{ent}}^i \cdot \mathcal{H}[\pi_i]$$

$$\hat{A}_{t,k}^i = \sum_{l=0}^{K_t-k} (\gamma^L \lambda)^l \delta_{t,k+l}^i, \quad \delta_{t,k}^i = r_{t,k}^i + \gamma^L V_{\psi_L}(o_{t,k+1}^{\text{joint}}) - V_{\psi_L}(o_{t,k}^{\text{joint}})$$

#### 预算拉格朗日约束

$$\mu \leftarrow \max(0,\ \mu + \eta_\mu(\bar{C}_{\text{batch}} - B))$$

$$\mathcal{L}^H_{\text{constrained}} = \mathcal{L}^H - \mu \cdot \mathbb{E}[\text{total\_cost}]$$

---

## 9. 一个完整Episode的详细运行示例

**问题**：$x$ = "一个水池有两个进水管A和B，A管单独注满需6小时，B管单独注满需8小时。两管同时打开，多长时间能注满？"

**真实答案**：$y^* = \frac{24}{7} \approx 3.43$ 小时

```
================================================================
Episode开始
================================================================

初始状态：
  T_0 = ∅, p_0 = ∅, V_0 = ∅, C_0 = ∅
  B = 5000 tokens (≈ SC-10等价预算)
  φ(s_0) = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e_x]

────────────────────────────────────────
轮次 t=0: 高层选择 EXPLORE(open, standard)
────────────────────────────────────────

  轮内预算: δB_0 = 0.15 × 5000 = 750 tokens
  K_max = 5

  ─── 微步 k=1 ───

  信息板: B_{0,1} = {}

  Proposer观测: (e_x, goal=EXPLORE, focus=open, B中无内容)
    work决策: generate (调用LLM)
    comm决策: submit-trace
    → LLM返回: τ₁="1/6+1/8=7/24, 所以24/7≈3.43小时", y₁=24/7
    → 消耗: 180 tokens
    → 写入信息板: msg_1 = (sender=P, type=TRACE, target=0, content=(τ₁,24/7), step=1)

  Critic观测: (e_x, goal=EXPLORE, focus=open, B中无内容)
    work决策: work-idle (EXPLORE轮首步, 还没有可批判的内容)
    comm决策: comm-idle

  Verifier观测: (e_x, goal=EXPLORE, focus=open, B中无内容)
    work决策: work-idle (还没有可验证的答案)
    comm决策: comm-idle

  信息板更新: B_{0,2} = {msg_1}
  δB_0^rem = 750 - 180 = 570

  ─── 微步 k=2 ───

  Proposer观测: (信息板有1条TRACE)
    work决策: generate-diverse (生成不同路径)
    comm决策: submit-trace
    → LLM返回: τ₂="每小时A注1/6，B注1/8，合计1/6+1/8=4/24+3/24=7/24，
                   注满需24/7小时", y₂=24/7
    → 消耗: 200 tokens
    → 写入: msg_2 = (P, TRACE, 0, (τ₂,24/7), 2)

  Critic观测: (信息板有1条TRACE msg_1)
    work决策: work-idle (只有一条链, 太少, 批判价值不高)
    comm决策: request(Proposer, "再生成一条不同方法的链")
    → 写入: msg_3 = (C, REQUEST, target=P, "请尝试用不同方法解题", 2)

  Verifier观测: (信息板有1条TRACE msg_1)
    work决策: quick-verify(y=24/7)
    comm决策: submit-score
    → LLM返回: score=0.85, rationale="方法正确,算术一致"
    → 消耗: 120 tokens
    → 写入: msg_4 = (V, SCORE, 0, (24/7, 0.85, "方法正确"), 2)

  信息板更新: B_{0,3} = {msg_1, msg_2, msg_3, msg_4}
  δB_0^rem = 570 - 200 - 120 = 250

  ─── 微步 k=3 ───

  Proposer观测: (收到Critic的REQUEST msg_3; 信息板有2条TRACE都指向24/7)
    work决策: generate-diverse (响应Critic的请求, 尝试不同方法)
    comm决策: submit-trace
    → LLM返回: τ₃="设需x小时。x/6+x/8=1, 4x/24+3x/24=1, 7x/24=1, x=24/7", y₃=24/7
    → 消耗: 190 tokens
    → 写入: msg_5 = (P, TRACE, 0, (τ₃,24/7), 3)

  Critic观测: (信息板有2条TRACE, 1条SCORE, 都指向24/7)
    work决策: find-counterexample(y=24/7)
    comm决策: submit-flaw
    → LLM返回: "尝试构造反例但未能找到矛盾。代入验证：(24/7)/6+(24/7)/8=4/7+3/7=1，正确。"
    → 实际上没有发现有效缺陷
    → 消耗: 150 tokens
    → 写入: msg_6 = (C, FLAW, 0, (24/7, "未发现有效缺陷", severity=none), 3)

  Verifier观测: (信息板有msg_6表示Critic未找到缺陷)
    work决策: work-idle (已有验证, Critic也没找到问题)
    comm决策: endorse(msg_4) (支持自己之前的验证)
    → 写入: msg_7 = (V, ENDORSE, 0, msg_id=4, 3)

  信息板更新: B_{0,4} = {msg_1,...,msg_7}
  δB_0^rem = 250 - 190 - 150 = -90 < 0

  → 轮内预算耗尽, 宏动作终止

  宏动作产出:
    new_traces = {(τ₁,24/7), (τ₂,24/7), (τ₃,24/7)}
    new_scores = {(24/7, 0.85)}
    new_critiques = {(24/7, "未发现有效缺陷", survived=0)} 
      // severity=none, 视为无效批判

全局状态更新:
  T_1 = {(τ₁,24/7,init), (τ₂,24/7,init), (τ₃,24/7,init)}
  p_1(24/7) = 1.0
  V_1 = {(24/7, 0.85)}
  C_1 = ∅ (无有效批判)
  b_1 = (5000-840, 840, 1)
  B_rem = 4160

高层步奖励: r_0^H = -β × 840

────────────────────────────────────────
轮次 t=1: 高层选择 STOP(weighted)
────────────────────────────────────────

  φ(s_1) 显示:
    p_1^(1) = 1.0 (唯一答案全票支持)
    H_1 = 0 (零熵)
    v_bar^(1) = 0.85 (高验证分)
    ρ_B = 0.168 (仅用17%预算)

  高层判断: 证据已充分一致, 停止

  聚合(weighted):
    w(24/7) = 1.0 × 0.85 × (1-0) = 0.85
    ŷ = 24/7 ✓

  R_global = 1[24/7 == 24/7] - α × 0.168 = 1 - α×0.168

================================================================
Episode结束
================================================================

总消耗: 840 tokens (≈ SC-1.7等价)
结果: 正确
轮次: 2 (1轮探索 + 1轮STOP)

特征: 简单题 → 系统快速收敛 → 早停节省预算
     Critic尝试了反驳但未成功 → 增强了答案置信度
     Proposer响应了Critic的REQUEST → agent间有效交互
```

---

## 10. 与现有方法的完整对比

| 维度       | SC   | BoN          | ReMA    | HMARL-SC             |
| -------- | ---- | ------------ | ------- | -------------------- |
| 底层agent数 | 0    | 0            | 1       | **3（功能分化）**          |
| agent间交互 | 无    | 无            | 无       | **显式交互动作**           |
| 交互是否可学习  | N/A  | N/A          | N/A     | **是**                |
| 采样次数     | 固定K  | 固定K          | 固定turns | **动态（高层控制）**         |
| 证据类型     | 仅正向  | 仅正向          | 仅正向     | **正向+负向+诊断**         |
| 证据过滤     | 无    | Reward model | 无       | **survived机制**       |
| 停止策略     | 无    | 无            | 无       | **学习的STOP策略**        |
| 聚合方式     | 多数投票 | 选最高分         | 投票      | **频率×验证分×攻击存活率**     |
| 预算感知     | 无    | 无            | 无       | **两层预算分配**           |
| 时间尺度     | 单层   | 单层           | 双层      | **双层（宏+微）**          |
| 训练目标     | 无    | Reward model | LLM权重   | **调度策略（LLM frozen）** |
| 成本自适应    | 固定   | 固定           | 固定      | **高层预算分配 + 低层gate**  |

---

## 11. 实验设计

### 11.1 分阶段验证

| 阶段     | 配置                                   | 新增组件                      | 验证目标        |
| ------ | ------------------------------------ | ------------------------- | ----------- |
| **E1** | 仅Proposer + Controller（EXPLORE/STOP） | 基础框架                      | 动态停止 vs 固定K |
| **E2** | E1 + Verifier                        | 验证证据 + weighted聚合         | 验证信号改善聚合？   |
| **E3** | E2 + Critic                          | 负向证据 + survived机制         | 批判改善结果？     |
| **E4** | E3 + 完整交互动作                          | REQUEST/CHALLENGE/ENDORSE | agent交互有价值？ |
| **E5** | E4 + 预算分配 $\delta B_t$               | 资源调度                      | 预算分配改善效率？   |

### 11.2 消融实验

| ID  | 对比                      | 回答的问题           |
| --- | ----------------------- | --------------- |
| A1  | 学习策略 vs 最佳固定规则          | RL是否必要？         |
| A2  | 有交互动作 vs 无交互（agent独立执行） | 交互是否有价值？        |
| A3  | 有REQUEST vs 无REQUEST    | 主动请求是否有价值？      |
| A4  | 有CHALLENGE/ENDORSE vs 无 | 证据质疑/支持是否有价值？   |
| A5  | 有survived过滤 vs 无        | Critic过滤机制是否有效？ |
| A6  | 参数共享 vs 独立网络            | 共享是否导致角色塌缩？     |
| A7  | 角色行为分布分析                | 角色是否真的分工？       |
| A8  | 有Verifier辅助奖励 vs 无      | 局部shaping是否必要？  |
| A9  | 不同 $\alpha$ 下Pareto曲线   | accuracy-cost权衡 |
| A10 | BC warm-start vs 随机初始化  | warm-start是否关键？ |
| A11 | 交替冻结 vs 联合训练            | 训练策略对比          |

### 11.3 关键分析

| 分析         | 内容                                     |
| ---------- | -------------------------------------- |
| 交互行为分析     | 统计各类comm动作频率、REQUEST的响应率、CHALLENGE的成功率 |
| 停止策略分析     | 按题目难度分析停止时机，对比"答对题用了多少预算"vs"答错题用了多少预算" |
| 角色贡献分析     | 各角色的work-idle率、各角色产出对最终答案的影响           |
| Case Study | 展示完整episode的信息板内容，说明agent间如何通过交互改进推理   |
| 失败分析       | 分析系统仍然答错的case，识别主要failure mode         |

---

## 12. 超参数汇总

| 超参数              | 符号                             | 默认值            | 含义              |
| ---------------- | ------------------------------ | -------------- | --------------- |
| 全局折扣             | $\gamma^H$                     | 0.99           | 高层跨轮折扣          |
| 微步折扣             | $\gamma^L$                     | 0.95           | 低层轮内折扣          |
| 成本权重（全局）         | $\alpha$                       | 0.2            | accuracy-cost权衡 |
| 高层步成本系数          | $\beta^H$                      | $1/B$          | 归一化成本惩罚         |
| 低层步成本系数          | $\beta^L$                      | $1/(3\bar{c})$ | 归一化微步成本惩罚       |
| Verifier辅助系数     | $\beta^V$                      | 0.1            | Verifier校准奖励权重  |
| 全局奖励共享系数         | $\eta$                         | 0.5            | 全局奖励传递给低层的权重    |
| PPO clip         | $\epsilon$                     | 0.1            | 近端约束            |
| GAE lambda       | $\lambda$                      | 0.95           | 优势估计偏差-方差权衡     |
| 高层entropy bonus  | $\lambda_{\text{ent}}^H$       | 0.01           | 探索鼓励            |
| 低层entropy bonus  | $\lambda_{\text{ent}}^{1,2,3}$ | 0.02           | 探索鼓励（可按角色动态调整）  |
| 最大轮次             | $T_{\max}$                     | 10             | episode最大高层步数   |
| 轮内最大微步（standard） | $K_{\max}$                     | 5              | 一轮内最大微步数        |
| 拉格朗日学习率          | $\eta_\mu$                     | 0.01           | 预算约束更新速率        |
| 交替轮数             | $K_{\text{rounds}}$            | 5              | Stage 1的交替训练轮数  |

---

## 13. 理论定位说明

本框架的理论定位是**一套具备合理设计动机的实用系统**，而非一套带有严格收敛保证的理论方法。

1. **交替冻结训练**受block coordinate optimization启发，但在完整随机系统（含外部LLM噪声）中不提供严格单调改进保证。其有效性通过经验性的训练曲线验证。

2. **低层交互协议**的最优性无理论保证。agent学习到的交互模式（何时REQUEST、何时CHALLENGE）是经验性涌现行为，而非理论最优解。

3. **survived过滤机制**是启发式的证据可靠性估计，不是形式化的贝叶斯推断。

4. **weighted聚合公式** $w(y) = p_T(y) \cdot \bar{v}(y) \cdot (1 - \text{surv\_attack}(y))$ 是设计选择而非推导结果，其有效性由实验验证。

---

## 14. 扩展路径

| 扩展              | 触发条件                         | 做法                 |
| --------------- | ---------------------------- | ------------------ |
| 更多Proposer mode | E4中diverse模式被频繁选择且有增益        | 扩展为4-6种mode        |
| Deliberate聚合    | weighted聚合的错误case中>20%是验证分误导 | 加入第三种STOP          |
| 低层自主协议          | 固定协议（谁先执行）成为瓶颈               | 引入turn-order作为学习动作 |
| 并行度扩展           | 计算资源充裕时                      | 允许同一角色多实例          |
| 跨episode元学习     | 积累足够多episode后                | 问题特征 → 初始策略映射      |
