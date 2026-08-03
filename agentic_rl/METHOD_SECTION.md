# 3. Method

## 3.1 Problem Formulation

We consider the task of mathematical reasoning, where an agent must solve a problem $q$ and produce a final answer $\hat{a}$ that matches the ground truth $a^*$. Standard single-agent approaches generate a single chain-of-thought and commit to a final answer. We instead model this as a **multi-round collaborative reasoning** problem: multiple specialized agents communicate over a shared memory structure across $T$ rounds, collectively refining a solution before committing to a final answer.

Formally, the system takes a question $q$ as input and produces a trajectory $\tau = \{(s_t, a_t^{k_t})\}_{t=1}^{T_{stop}}$, where $s_t$ is the shared state at step $t$, $a_t$ is the text action produced, and $k_t \in \mathcal{K}$ denotes the active role. The final answer $\hat{a}$ is determined by majority vote over all candidate answers generated across rounds. The objective is to train the system's policies $\{\pi_{\theta_k}\}_{k \in \mathcal{K}}$ to maximize the probability of producing a correct final answer.

---

## 3.2 Multi-Agent Collaborative Reasoning Framework

### 3.2.1 Role Design

We propose a **four-role cooperative architecture**, where each role is specialized for a distinct cognitive function in the reasoning process:

- **Controller** ($\pi_{\theta_\text{ctrl}}$): A high-level meta-reasoner that observes the current blackboard state each round and decides the reasoning strategy for that round. It outputs a structured *meta-plan* specifying a strategy from $\{\texttt{explore}, \texttt{refine}, \texttt{verify}, \texttt{stop}\}$ and a focus agent from $\{\texttt{proposer}, \texttt{critic}, \texttt{verifier}, \texttt{balanced}\}$. The controller acts as an orchestrator, directing the team's attention and deciding when sufficient confidence has been achieved to stop.

- **Proposer** ($\pi_{\theta_\text{prop}}$): A solution generator that produces or refines a step-by-step reasoning chain and a candidate answer. The proposer is the primary answer-generating agent and is invoked in every round. It can additionally initiate peer interactions by requesting critique or verification.

- **Critic** ($\pi_{\theta_\text{crit}}$): An adversarial error-detector that reviews the proposer's most recent reasoning chain and identifies logical or computational errors. When errors are found, the critic describes them explicitly; when none are found, it outputs a "no error" signal. The critic can also initiate follow-up interactions.

- **Verifier** ($\pi_{\theta_\text{verif}}$): A confidence estimator that independently validates a candidate answer and assigns a continuous score $v \in [0, 1]$ reflecting its confidence in the answer's correctness. The verifier provides a calibrated signal without access to the ground truth at inference time.

### 3.2.2 Shared Blackboard Memory

Agents communicate through a **structured shared blackboard** $\mathcal{B}$, which maintains four typed message queues updated incrementally across rounds:

$$\mathcal{B} = \{\underbrace{\mathcal{T}}_{\text{reasoning traces}},\ \underbrace{\mathcal{F}}_{\text{identified flaws}},\ \underbrace{\mathcal{S}}_{\text{confidence scores}},\ \underbrace{\mathcal{I}}_{\text{interaction log}}\}$$

- $\mathcal{T}$: a list of (reasoning chain, candidate answer) pairs contributed by the proposer
- $\mathcal{F}$: error descriptions contributed by the critic
- $\mathcal{S}$: (answer, score) pairs contributed by the verifier
- $\mathcal{I}$: a log of peer interaction events (who requested what from whom)

Each agent observes a text rendering of the current blackboard state $\mathcal{B}$ as part of its input context. This design allows agents to build on each other's contributions across rounds without requiring direct message passing, and enables the controller to make informed decisions about the overall progress.

### 3.2.3 Round Structure and Interaction Protocol

Each reasoning **round** $t$ proceeds in three stages:

**Stage 1 — Meta-Planning (Controller).** The controller observes the question $q$ and the current blackboard state $\mathcal{B}$ and outputs a meta-plan:
$$m_t \leftarrow \pi_{\theta_\text{ctrl}}(q, \mathcal{B}_t)$$
The meta-plan specifies a strategy (e.g., *refine* if flaws exist, *verify* if a confident answer is available, *stop* if the answer is sufficiently reliable). If the strategy is \texttt{stop}, the episode terminates.

**Stage 2 — Primary Response.** Based on the focus specified in $m_t$, a primary agent (proposer, critic, or verifier) is invoked. Its output is written to the corresponding queue of $\mathcal{B}$.

**Stage 3 — Peer Interaction.** Each agent's output may contain an `<interaction>` block specifying a request directed at another agent (e.g., the proposer requests critique from the critic). Up to $I_{\max}$ interaction turns are executed per round, each involving one agent responding to another's output. All interaction responses are also recorded on the blackboard.

After Stage 3, if the proposer has not yet been invoked in this round (e.g., because the controller focused on critic or verifier), a proposer **fallback** is triggered to ensure a fresh candidate answer is always available.

**Final Answer Selection.** After all rounds conclude (either by controller stopping or reaching $T_{\max}$ rounds), the final answer $\hat{a}$ is determined by majority vote over all candidate answers in $\mathcal{T}$:
$$\hat{a} = \arg\max_{a} \sum_{(\cdot, a') \in \mathcal{T}} \mathbf{1}[a' = a]$$

### 3.2.4 Model Architecture

All four agents share a single frozen pretrained LLM (Qwen3-8B) as their backbone, with **four independent LoRA adapters** $\{\theta_k\}_{k \in \mathcal{K}}$ for role specialization. Formally:

$$\pi_{\theta_k}(\cdot | s) = \text{LLM}_\text{base}(\cdot | s;\ \Delta W_k), \quad k \in \mathcal{K}$$

where $\Delta W_k$ are the trainable LoRA parameters for role $k$ and $\text{LLM}_\text{base}$ is the frozen backbone. This design allows efficient role specialization at a fraction of the cost of maintaining four separate models, while ensuring that role-specific behaviors are learned independently (gradients for $\theta_k$ only accumulate on turns where role $k$ was active).

---

## 3.3 Training: RACA (Role-Aware Credit Assignment)

Standard GRPO assigns the same episode-level advantage to every token in a trajectory, regardless of which role produced it. In our multi-agent setting, this is particularly harmful because: (1) different roles have fundamentally different reward semantics, and (2) episode-level normalization conflates the diverse reward distributions across roles, causing gradient instability (Feng et al., 2026). We address this with **RACA (Role-Aware Credit Assignment)**, a training algorithm that provides each role with precise, causally grounded credit signals.

### 3.3.1 Phase 1: Rollout Collection

For each question $q$ in a training batch, we generate $N$ independent episode rollouts $\{\tau^i\}_{i=1}^N$ using the current policies. We record per-round metadata for each rollout: proposer correctness $p_t^i = \mathbf{1}[\hat{a}_t^i = a^*]$, critic flag $f_t^i$, verifier score $v_t^i$, the stopping round $t_\text{stop}^i$, and episode correctness $c^i$.

### 3.3.2 Phase 2: Role-Specific Reward Design

Rather than using a single outcome reward for all roles, we define role-specific per-turn rewards that reflect each role's actual functional contribution.

**Proposer.** The proposer is rewarded for each round in which its solution is correct:
$$r_{\text{prop},t}^i = p_t^i$$

This provides a dense signal at every round, eliminating the sparsity of the original design (which only rewarded the final-round proposer).

**Critic.** The critic is rewarded based on a causal four-cell matrix. Rather than conditioning on the final episode outcome (which introduces survivorship bias), we condition on whether the critic's intervention causally led to proposer improvement in the *next* round:
$$r_{\text{crit},t}^i = \begin{cases} 0.3 \cdot p_{t+1}^i + 0.1 \cdot (1 - p_{t+1}^i) & f_t^i = 1,\ p_t^i = 0 \quad \text{(true positive)} \\ -0.2 & f_t^i = 1,\ p_t^i = 1 \quad \text{(false positive)} \\ +0.1 & f_t^i = 0,\ p_t^i = 1 \quad \text{(true negative)} \\ 0 & f_t^i = 0,\ p_t^i = 0 \quad \text{(missed error)} \end{cases}$$
where $p_{t+1}^i$ is replaced by $c^i$ at the final round. This design forces the critic to produce *actionable* feedback that the proposer can act upon; a mathematically correct critique that the proposer fails to use yields only partial reward.

**Verifier.** The verifier is rewarded for calibration accuracy:
$$r_{\text{verif},t}^i = 1 - |v_t^i - p_t^i|$$

This encourages the verifier to output scores that faithfully reflect the proposer's actual correctness, rewarding both confident correct predictions and appropriately uncertain ones.

**Controller.** The controller is rewarded for efficiency, with a base correctness reward plus an efficiency bonus:
$$r_{\text{ctrl}}^i = c^i + \alpha \cdot c^i \cdot \frac{T_{\max} - t_\text{stop}^i}{T_{\max}} - \beta \cdot (1 - c^i)$$
This reward is assigned only to the last controller turn per episode. The decomposition ensures that correctness always dominates over efficiency (gradient gap $\geq 1 + \beta$), while the efficiency bonus incentivizes the controller to stop early when the answer is reliable.

### 3.3.3 Phase 3: Two-Level Advantage Computation

RACA computes advantages at two complementary granularities, routed by role type.

**Layer 1 — Episode-Level Advantage (Controller).** Since the controller's decisions affect the entire episode, we normalize its reward at the episode level across the $N$ rollouts:
$$A_{E,\text{ctrl}}^i = \frac{r_\text{ctrl}^i - \mu_\text{ctrl}}{\max(\sigma_\text{ctrl},\ \delta)}$$
where $\mu_\text{ctrl}$, $\sigma_\text{ctrl}$ are the mean and standard deviation of $\{r_\text{ctrl}^i\}$ across the group, and $\delta = 10^{-4}$ is a variance floor that prevents discarding low-variance groups.

**Layer 2 — Step-Level Advantage (Proposer, Critic, Verifier).** For the three local roles, we introduce a *role-strategy anchor state* grouping mechanism inspired by GiGPO (Feng et al., 2025). The key observation is that the controller explicitly labels each round with a cognitive strategy $\sigma_t \in \{\texttt{explore}, \texttt{refine}, \texttt{verify}\}$, providing a natural indicator of the task context each role faces in that round. We define the anchor state as $\tilde{s}_{k,t} = (\text{role}=k,\ \text{strategy}=\sigma_t)$ and construct an anchor group:
$$G_S(k, \sigma) = \{(i,\ r_{k,t}^i) \mid \text{rollout } i \text{ has role } k \text{ invoked under strategy } \sigma\}$$
Step-level advantages are then computed by normalizing within each anchor group:
$$A_{S,k,\sigma}^i = \frac{r_{k,t}^i - \mu_{k,\sigma}}{\max(\sigma_{k,\sigma},\ \delta)}$$

This differs from both GiGPO and a naive round-based grouping in important ways. GiGPO requires literal environment state identity across trajectories—a condition that does not hold in text-based reasoning where blackboard states diverge rapidly. A round-index anchor ($\text{round}=t$) is weaker: by round $t \geq 2$, different rollouts have accumulated different blackboard contents (some have critic feedback, others do not), violating the controlled-comparison assumption. The strategy anchor is strictly stronger: all turns in the same $(k, \sigma)$ group were invoked by the controller under the same meta-level decision (e.g., all proposers in the \texttt{refine} group were asked to fix a flagged solution), creating a meaningful controlled comparison of action quality within a shared cognitive context. Furthermore, the group reward $r_{k,t}^i$ is the verifiable per-round correctness signal rather than a discounted sum, yielding an unbiased estimator.

**Advantage Routing.** The final advantage for each turn is determined by role:
$$A^i_\text{total}(k, t) = \begin{cases} A^i_{E,\text{ctrl}} & k = \text{controller} \\ A^i_{S,k,t} & k \in \{\text{proposer, critic, verifier}\} \end{cases}$$

This routing prevents cross-role noise contamination: the proposer's per-round correctness signal is not diluted by the controller's episode-level outcome, and vice versa.

### 3.3.4 Phase 4: Policy Update

For each turn $(k, i, t)$ with advantage $A^i_\text{total}(k, t)$, we apply a PPO clipped surrogate objective:
$$\mathcal{L}(k, t) = \max\!\Bigl(-A^i_\text{total} \cdot \rho_\theta(k,t),\ -A^i_\text{total} \cdot \text{clip}\bigl(\rho_\theta(k,t),\ 1 \pm \epsilon\bigr)\Bigr)$$
where $\rho_\theta = \pi_{\theta_k}(a_t^i | s_t^i) / \pi_{\theta_k^\text{old}}(a_t^i | s_t^i)$ is the importance sampling ratio. Gradients are accumulated across all valid turns in the batch and normalized by the total turn count $N_\text{valid}$. Since each role's adapter is activated via `set_adapter()` before its forward pass, the four LoRA adapters receive gradients independently with no interference.

---

## 3.4 Summary of Contributions

Table 1 summarizes RACA's relationship to prior work.

| Component | Prior Work | This Work |
|-----------|-----------|-----------|
| Per-agent normalization | Dr. MAS (Feng et al., 2026) | Role-specific $R^i_k$ definitions, not just normalization statistics |
| Step-level anchor grouping | GiGPO (Feng et al., 2025) | Structural role-round anchor; verifiable correctness reward replaces discounted return |
| Critic reward | — | Causal four-cell matrix conditioning on $p_{t+1}^i$ (intervention effect) |
| Controller reward | — | Correctness-base + efficiency bonus, resolving dead-zone in naive efficiency rewards |
| Advantage routing | — | Role-type-conditioned routing (episode-level for orchestrator, step-level for workers) |

The key insight underlying RACA is that **the verifiable nature of mathematical reasoning enables exact intermediate credit signals at zero additional cost**: unlike web or tool-use settings (where intermediate state verification requires expensive re-execution or LLM judges), the mathematical domain allows us to compute $p_t^i$ directly from symbolic comparison, making per-round credit assignment both exact and computationally free.
