"""RACA v2 Phase 3：两层优势计算（纯 Python，零 torch/numpy 依赖，可单测）。

- Layer 1（Controller）：episode 级组内归一化，零方差组丢弃（继承 v1.x Fix 7）
- Layer 2（Prop/Crit/Verif）：anchor key = (role, σ, is_response, channel, layer_key)
  - v2.3 双通道（§18）：proposer 主 turn 的优势 = z(r_prop) + z(λ·r_int)，
    两通道各自组内归一化后相加。r_prop 通道不按 p_t 分层（v2.1 的全量分层
    使 int_rate=0 后主 turn 组内零方差→整组被丢→解题能力停训，v2.2 首跑
    实测 acc 横盘）；r_int 通道保留 p_t 分层（v2.1 的隔离目的不变）。
    副作用修复：int_rate=0 时 r_int 通道自然失活，但 r_prop 通道仍供梯度
    ——主 turn 不再整组消失，int_rate=0 不再是吸收态。
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


def compute_raca_advantages(turn_data_list: list, delta: float = 1e-4) -> list:
    """对同一问题的 N 个 rollout 计算逐 turn 优势。

    turn_data_list: N 个 raca_turn_data（{tid: {role, round, sigma,
                    is_response, reward}}）。
    Returns: N 个 {tid: advantage} 字典（无优势的 turn 不出现）。
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
    # entry: (ep_idx, tid, reward, dedup_key)
    anchor_groups: dict = defaultdict(list)
    for ep_idx, td in enumerate(turn_data_list):
        for tid, v in td.items():
            if v["role"] == "controller":
                continue
            sigma   = v.get("sigma", "explore")
            is_resp = bool(v.get("is_response", False))
            if v["role"] == "proposer" and not is_resp and "r_prop" in v:
                # v2.3 双通道：r_prop 不分层，r_int 按 p_t 分层
                chans = [
                    (("proposer", sigma, False, "prop", None), v["r_prop"]),
                    (("proposer", sigma, False, "int", v.get("layer_key")),
                     v["r_int_w"]),
                ]
            else:
                chans = [((v["role"], sigma, is_resp, "rew",
                           v.get("layer_key")), v["reward"])]
            for key, r in chans:
                dedup_key = (ep_idx, v.get("round", 0), v["role"], is_resp, r)
                anchor_groups[key].append((ep_idx, tid, r, dedup_key))

    step_adv: dict = {}   # (ep_idx, tid) → advantage（多通道累加）
    for key, entries in anchor_groups.items():
        # 代表样本：同 (episode, round, role, is_response, reward) 只保留一个
        reps = {}
        for ep_idx, tid, r, dk in entries:
            reps.setdefault(dk, r)
        rep_rewards = list(reps.values())
        if len(rep_rewards) < 2:
            continue
        mu, sig = _mean_std(rep_rewards)
        if sig <= delta:
            continue   # 该通道组内全同分，无比较信号（不影响其它通道）
        for ep_idx, tid, r, _dk in entries:
            step_adv[(ep_idx, tid)] = (
                step_adv.get((ep_idx, tid), 0.0) + (r - mu) / sig
            )

    # ── 汇总路由 ─────────────────────────────────────────────────────────────
    per_ep_adv = [{} for _ in range(N)]
    for ep_idx, td in enumerate(turn_data_list):
        for tid, v in td.items():
            if v["role"] == "controller":
                if ctrl_adv is not None:
                    per_ep_adv[ep_idx][tid] = ctrl_adv[ep_idx]
            else:
                sa = step_adv.get((ep_idx, tid))
                if sa is not None:
                    per_ep_adv[ep_idx][tid] = sa
    return per_ep_adv
