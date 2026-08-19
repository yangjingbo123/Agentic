"""Rollout 行为指标聚合（纯 numpy，零 torch 依赖，可 CPU 单测）。

三类指标（标准 GRPO 看盘项 + RACA v2 证据指标）：
- 信号质量：全对组/全错组占比、组内 reward std —— GRPO 的命门，组内零方差
  ⇒ advantage 全零 ⇒ 该组无梯度；趋零即信号枯竭（DAPO 动态采样正为此而生）
- 行为/格式：答案可解析率（格式崩了 reward 再高也是假的）
- RACA v2 证据（§8）：交互率/有效率/选择性、stop 校准、闸门拦截数

注：熵、clip fraction、importance ratio 需要训练前向的完整 logits，
在 grpo_trainer._compute_loss 内采集（vLLM 只回 top-20 logprobs，算不出全分布熵）。
"""

from __future__ import annotations

import numpy as np


def rollout_metrics(batch_rollouts: list) -> dict:
    """batch_rollouts: list[list[episode_dict]]，按问题分组的全部 rollout。

    统计对象是**全部** rollout（含被优势过滤掉的 episode），因为这些指标
    描述的是采样行为本身，不应受下游梯度过滤影响。
    """
    eps = [ep for group in batch_rollouts for ep in group]
    rounds = [m for ep in eps for m in ep.get("raca_round_meta", [])]
    out: dict = {}

    # ── RACA v2 交互证据指标（§8） ───────────────────────────────────────────
    if rounds:
        us = [1.0 if m["u"] else 0.0 for m in rounds]
        ps = [1.0 if m["p_primary"] else 0.0 for m in rounds]
        out["int_rate"]     = float(np.mean(us))
        out["forced_rate"]  = float(np.mean([m["forced"] for m in rounds]))
        out["gate_blocked"] = int(sum(m["gate_blocked"] for m in rounds))
        # 交互有效率：P(轮末修对 | 自发求助且 primary 错)
        eff = [m["p_end"] for m in rounds if m["u"] and not m["p_primary"]]
        if eff:
            out["int_effectiveness"] = float(np.mean(eff))
        # 选择性：corr(u, p_primary)，预期随训练负相关增强（错的时候才求助）
        if np.std(us) > 0 and np.std(ps) > 0:
            out["int_selectivity"] = float(np.corrcoef(us, ps)[0, 1])
        # 答案可解析率：格式崩了 reward 再高也是假的
        out["parse_rate"] = float(np.mean(
            [1.0 if m.get("primary_parsed", True) else 0.0 for m in rounds]))
        # ── 修正漏斗（v2.2）：flag → correction → flip ──────────────────
        # v2.1 实测 eff≈0 的定位工具。计入全部轮（含 forced 注入）：
        # flag 高而 corr 低 = 修正跳断（hop 预算/机制）；
        # corr 高而 flip 低 = proposer 拿着反馈也修不对（反馈质量/能力上限）。
        out["funnel_flag"] = int(sum(m.get("n_flagged", 0) for m in rounds))
        out["funnel_corr"] = int(sum(m.get("n_corrections", 0) for m in rounds))
        out["funnel_flip"] = int(sum(1 for m in rounds if m.get("flip")))

    # ── stop 校准：P(correct | stop) vs P(correct | 耗尽轮次) ────────────────
    stopped   = [ep["is_correct"] for ep in eps if ep.get("stopped")]
    exhausted = [ep["is_correct"] for ep in eps if not ep.get("stopped")]
    if stopped:
        out["stop_acc"] = float(np.mean(stopped))
    if exhausted:
        out["exhaust_acc"] = float(np.mean(exhausted))
    if eps:
        out["stop_rate"] = float(np.mean(
            [1.0 if ep.get("stopped") else 0.0 for ep in eps]))

    # ── 信号质量（GRPO 命门） ────────────────────────────────────────────────
    # 以 episode 级正确性分组统计（controller 层 Layer 1 信号的直接代理）。
    all_pass = all_fail = 0
    stds = []
    for group in batch_rollouts:
        cs = [1.0 if ep["is_correct"] else 0.0 for ep in group]
        if not cs:
            continue
        if all(c == 1.0 for c in cs):
            all_pass += 1
        elif all(c == 0.0 for c in cs):
            all_fail += 1
        stds.append(float(np.std(cs)))
    if stds:
        n_g = len(stds)
        out["all_pass_frac"]    = all_pass / n_g   # 持续上升 = 题太简单，信号枯竭
        out["all_fail_frac"]    = all_fail / n_g   # 持续上升 = 题太难，同样无梯度
        out["group_reward_std"] = float(np.mean(stds))   # 趋零 = 信号枯竭
    return out
