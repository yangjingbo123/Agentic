"""RACA v2 Phase 3：两层优势计算（纯 Python，零 torch/numpy 依赖，可单测）。

- Layer 1（Controller）：episode 级组内归一化，零方差组丢弃（继承 v1.x Fix 7）
- Layer 2（Prop/Crit/Verif）：anchor key = (role, σ, is_response, channel, layer_key)
  - Proposer primary 使用 token-routed 双通道：solution = z(r_prop)，interaction =
    λ·z(r_int)。**权重必须在标准化后乘**；旧版 z(λr)=z(r) 使任何正 λ 都失效。
    两个值以结构化 dict 返回，训练侧分别路由到推理/答案 token 与末尾 interaction
    block token，不再先相加后广播整段。r_prop 不按 p_t 分层（否则组内恒定），
    r_int 保留 p_t 分层（隔离「答对仍问」与「答错不问」两种决策）。
  - 副作用修复：int_rate=0 时 r_int 通道自然失活，但 r_prop 通道仍供梯度
    ——主 turn 不再整组消失、**解题能力**可继续训练。注意：此前把这句话写成
    “int_rate=0 不再是吸收态”说过头了；恢复的是 r_prop，不是交互动作本身。
    08-28 两跑实测 int_rate 到 0 后 160 步不恢复，交互决策仍是吸收态。
  - v3.2 第十四轮：forced 轮的 r_int_w=None，完全退出 int anchor group。
    forced 的数值 0 在组相对归一化里并非零梯度：它会高于 −int_miss，并给模型
    实际输出的 action:none 正优势。forced 仍保留 r_prop、响应角色奖励和 q_forced。
  - 旧格式兼容：无 r_prop/r_int_w 字段的 turn 走单通道（reward + layer_key）。
- 组内去重（v2 §5.2）：同一 episode 同一轮的同角色多 turn 携带相同 reward 时，
  只以一个代表样本参与 μ/σ 计算，优势再广播回全部 turn——防重复样本
  人为压低组内方差。
"""

import math
from collections import defaultdict


def _mean_std(values: list) -> tuple:
    n = len(values)
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n   # 总体方差，与 np.std 默认一致
    return mu, math.sqrt(var)


def token_credit_components(advantage_spec, interaction_span, original_positions):
    """返回 ``[(name, advantage, local_token_indices)]``（纯 Python，可 CPU 单测）。

    旧 scalar advantage 仍广播整段。新 proposer primary 的 solution 只覆盖块前
    token，interaction 只覆盖严格的 ``[start,end)`` 块区间；end 后可能存在的
    EOS/special token 不收 PG credit（仍可收 KL）。结构化 spec 没有可证明边界时
    返回空列表，fail-closed 跳过该 turn，避免猜测性路由。
    """
    positions = list(original_positions)
    if not isinstance(advantage_spec, dict):
        return [("default", float(advantage_spec), list(range(len(positions))))]
    if (not isinstance(interaction_span, (tuple, list))
            or len(interaction_span) != 2):
        return []
    start, end = interaction_span
    if (not (isinstance(start, int) and isinstance(end, int) and 0 < start < end)
            or end > len(positions)):
        return []
    solution_idx = [i for i, pos in enumerate(positions) if pos < start]
    interaction_idx = [i for i, pos in enumerate(positions) if start <= pos < end]
    components = []
    if "solution" in advantage_spec and solution_idx:
        components.append(("solution", float(advantage_spec["solution"]), solution_idx))
    if "interaction" in advantage_spec and interaction_idx:
        components.append(("interaction", float(advantage_spec["interaction"]),
                           interaction_idx))
    return components


def compute_raca_advantages(turn_data_list: list, delta: float = 1e-4) -> list:
    """对同一问题的 N 个 rollout 计算逐 turn 优势。

    turn_data_list: N 个 raca_turn_data（{tid: {role, round, sigma,
                    is_response, reward}}）。
    Returns: N 个 {tid: advantage_spec}。普通 turn 的 spec 是 float；新格式
    proposer primary 的 spec 是 {"solution": float, "interaction": float} 的子集，
    供训练侧按 token span 路由。无任何有效通道的 turn 不出现。
    """
    N = len(turn_data_list)

    # ── Layer 1: controller episode 级 ───────────────────────────────────────
    ctrl_rewards = []
    for td in turn_data_list:
        ctrl_rewards.append(
            sum(v["reward"] for v in td.values() if v["role"] == "controller")
        )
    mu_c, sig_c = _mean_std(ctrl_rewards)
    # 零方差组不发零优势：直接丢弃该层（不消耗前向、不稀释 total_valid）
    ctrl_adv = (
        [(r - mu_c) / sig_c for r in ctrl_rewards] if sig_c > delta else None
    )

    # ── Layer 2: anchor (role, σ, is_response, channel, layer_key) + 去重 ───
    # entry: (ep_idx, tid, raw_reward, dedup_key, credit_name, post_z_weight)
    anchor_groups: dict = defaultdict(list)
    token_credit_turns = set()
    for ep_idx, td in enumerate(turn_data_list):
        for tid, v in td.items():
            if v["role"] == "controller":
                continue
            sigma   = v.get("sigma", "explore")
            is_resp = bool(v.get("is_response", False))
            if v["role"] == "proposer" and not is_resp and "r_prop" in v:
                structured = bool(v.get("token_credit", False))
                if structured:
                    token_credit_turns.add((ep_idx, tid))
                    chans = [
                        (("proposer", sigma, False, "prop", None),
                         v["r_prop"], "solution", 1.0),
                    ]
                    # forced 的 raw r_int=None：只保留 solution channel。非 forced
                    # 先在 raw reward 上做 z-score，再乘 lambda；lambda=0 直接关闭。
                    r_int = v.get("r_int")
                    weight = float(v.get("lambda_int", 1.0))
                    if r_int is not None and weight != 0.0:
                        chans.append((
                            ("proposer", sigma, False, "int", v.get("layer_key")),
                            r_int, "interaction", weight,
                        ))
                else:
                    # 旧 episode / 手工测试兼容：沿用 turn scalar = z(r_prop)+z(r_int_w)。
                    chans = [
                        (("proposer", sigma, False, "prop", None),
                         v["r_prop"], "legacy", 1.0),
                    ]
                    r_int_w = v.get("r_int_w")
                    if r_int_w is not None:
                        chans.append((
                            ("proposer", sigma, False, "int", v.get("layer_key")),
                            r_int_w, "legacy", 1.0,
                        ))
            else:
                chans = [
                    ((v["role"], sigma, is_resp, "rew", v.get("layer_key")),
                     v["reward"], "default", 1.0),
                ]
            for key, r, credit_name, weight in chans:
                dedup_key = (ep_idx, v.get("round", 0), v["role"], is_resp, r)
                anchor_groups[key].append(
                    (ep_idx, tid, r, dedup_key, credit_name, weight))

    # (ep_idx, tid) → {credit_name: weighted standardized advantage}
    step_channels: dict = defaultdict(dict)
    for key, entries in anchor_groups.items():
        # 代表样本：同 (episode, round, role, is_response, raw reward) 只保留一个。
        reps = {}
        for ep_idx, tid, r, dk, _name, _weight in entries:
            reps.setdefault(dk, r)
        rep_rewards = list(reps.values())
        if len(rep_rewards) < 2:
            continue
        mu, sig = _mean_std(rep_rewards)
        if sig <= delta:
            continue
        for ep_idx, tid, r, _dk, credit_name, weight in entries:
            value = weight * ((r - mu) / sig)   # lambda 在 z-score **之后**生效
            slot = step_channels[(ep_idx, tid)]
            slot[credit_name] = slot.get(credit_name, 0.0) + value

    # ── 汇总路由 ─────────────────────────────────────────────────────────────
    per_ep_adv = [{} for _ in range(N)]
    for ep_idx, td in enumerate(turn_data_list):
        for tid, v in td.items():
            if v["role"] == "controller":
                if ctrl_adv is not None:
                    per_ep_adv[ep_idx][tid] = ctrl_adv[ep_idx]
                continue
            channels = step_channels.get((ep_idx, tid))
            if not channels:
                continue
            if (ep_idx, tid) in token_credit_turns:
                # 只返回实际有比较信号的通道。forced 常只有 solution；若某一组
                # 零方差，另一组仍可独立训练。
                spec = {k: channels[k] for k in ("solution", "interaction")
                        if k in channels}
                if spec:
                    per_ep_adv[ep_idx][tid] = spec
            else:
                # 旧 turn 与单通道角色仍使用整段标量，保持向后兼容。
                per_ep_adv[ep_idx][tid] = sum(channels.values())
    return per_ep_adv
