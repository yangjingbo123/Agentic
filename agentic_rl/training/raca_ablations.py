"""RACA component ablations (Table `tab:raca-ablation`), additive and non-invasive.

This module does **not** modify `training/raca_adv.py`. It provides a drop-in
replacement for `compute_raca_advantages` that, given an ablation mode, removes
exactly one of RACA's three credit-assignment components while keeping the same
multi-agent architecture, turn-level rewards, and optimization protocol:

  - ``no_role``     : drop the role coordinate from the worker anchor key, so
                      rewards produced by different agent roles are normalized in
                      shared comparison groups (Role=No, Context=Yes, Channel=Yes).
  - ``no_context``  : drop the semantic-context coordinate ``sigma`` (ξ ∈
                      {explore, refine, verify}), so turns produced under
                      different blackboard states are compared together
                      (Role=Yes, Context=No, Channel=Yes).
  - ``no_channel``  : collapse the primary proposer's solution and interaction
                      channels into a single turn-level scalar advantage
                      r_pro = r_sol + λ_int·r_int (Eq. proposer-primary-reward),
                      broadcast over the whole turn rather than routed to
                      solution / interaction token spans
                      (Role=Yes, Context=Yes, Channel=No).

For ``ablation=None`` this delegates verbatim to the original estimator, so the
non-ablated RACA path is byte-for-byte unchanged. The channel ablation is also
reachable through the existing ``token_credit: false`` config switch; the
``no_channel`` mode here is the stricter, single-comparison-group variant that
matches Eq. (proposer-primary-reward) directly.
"""

from collections import defaultdict

from training.raca_adv import _mean_std, compute_raca_advantages as _orig

_ABLATIONS = ("no_role", "no_context", "no_channel")


def make_ablated_advantage_fn(ablation):
    """Return a function with the same signature as ``compute_raca_advantages``.

    Passing an unknown / falsy ``ablation`` returns the original estimator so the
    default RACA behavior is preserved exactly.
    """
    mode = (str(ablation).strip().lower() if ablation else "")
    if not mode:
        return _orig
    if mode not in _ABLATIONS:
        raise ValueError(
            "unknown RACA ablation %r; choose one of %s"
            % (ablation, ", ".join(_ABLATIONS)))

    def _fn(turn_data_list, delta=1e-4,
            balance_interaction_layers=False, interaction_balance_cap=2.0):
        return compute_raca_advantages_ablated(
            turn_data_list, ablation=mode, delta=delta,
            balance_interaction_layers=balance_interaction_layers,
            interaction_balance_cap=interaction_balance_cap)

    return _fn


def compute_raca_advantages_ablated(
    turn_data_list: list,
    ablation: str = "no_role",
    delta: float = 1e-4,
    balance_interaction_layers: bool = False,
    interaction_balance_cap: float = 2.0,
) -> list:
    """Ablated variant of :func:`training.raca_adv.compute_raca_advantages`.

    The controller (Layer 1) path is identical to RACA in every mode; only the
    worker (Layer 2) anchor keys / channels change according to ``ablation``.
    """
    mode = str(ablation).strip().lower()
    if not mode:
        return _orig(
            turn_data_list, delta=delta,
            balance_interaction_layers=balance_interaction_layers,
            interaction_balance_cap=interaction_balance_cap)
    if mode not in _ABLATIONS:
        raise ValueError("unknown RACA ablation %r" % (ablation,))

    drop_role = (mode == "no_role")
    drop_context = (mode == "no_context")
    merge_channel = (mode == "no_channel")

    def role_key(role):
        return "_all_" if drop_role else role

    def sigma_key(sigma):
        return "_all_" if drop_context else sigma

    N = len(turn_data_list)

    # ── Layer 1: controller episode 级（与 RACA 完全一致，不受消融影响）──────
    ctrl_rewards = []
    for td in turn_data_list:
        ctrl_rewards.append(
            sum(v["reward"] for v in td.values() if v["role"] == "controller")
        )
    mu_c, sig_c = _mean_std(ctrl_rewards)
    ctrl_adv = (
        [(r - mu_c) / sig_c for r in ctrl_rewards] if sig_c > delta else None
    )

    # ── Layer 2: anchor + 去重 ───────────────────────────────────────────────
    anchor_groups: dict = defaultdict(list)
    token_credit_turns = set()
    for ep_idx, td in enumerate(turn_data_list):
        for tid, v in td.items():
            if v["role"] == "controller":
                continue
            sigma = v.get("sigma", "explore")
            is_resp = bool(v.get("is_response", False))
            if v["role"] == "proposer" and not is_resp and "r_prop" in v:
                structured = bool(v.get("token_credit", False))
                r_int = v.get("r_int")
                weight = float(v.get("lambda_int", 1.0))
                if merge_channel:
                    # Channel ablation: one combined scalar reward, one group,
                    # broadcast over the whole turn (never a token_credit turn).
                    combined = float(v["r_prop"])
                    if r_int is not None and weight != 0.0:
                        combined += weight * float(r_int)
                    chans = [
                        ((role_key("proposer"), sigma_key(sigma), False,
                          "prop", None),
                         combined, "default", 1.0),
                    ]
                elif structured:
                    token_credit_turns.add((ep_idx, tid))
                    chans = [
                        ((role_key("proposer"), sigma_key(sigma), False,
                          "prop", None),
                         v["r_prop"], "solution", 1.0),
                    ]
                    if r_int is not None and weight != 0.0:
                        chans.append((
                            (role_key("proposer"), sigma_key(sigma), False,
                             "int", v.get("layer_key")),
                            r_int, "interaction", weight,
                        ))
                else:
                    chans = [
                        ((role_key("proposer"), sigma_key(sigma), False,
                          "prop", None),
                         v["r_prop"], "legacy", 1.0),
                    ]
                    r_int_w = v.get("r_int_w")
                    if r_int_w is not None:
                        chans.append((
                            (role_key("proposer"), sigma_key(sigma), False,
                             "int", v.get("layer_key")),
                            r_int_w, "legacy", 1.0,
                        ))
            else:
                chans = [
                    ((role_key(v["role"]), sigma_key(sigma), is_resp, "rew",
                      v.get("layer_key")),
                     v["reward"], "default", 1.0),
                ]
            for key, r, credit_name, weight in chans:
                dedup_key = (ep_idx, v.get("round", 0), v["role"], is_resp, r)
                anchor_groups[key].append(
                    (ep_idx, tid, r, dedup_key, credit_name, weight))

    group_stats = {}
    for key, entries in anchor_groups.items():
        reps = {}
        for _ep_idx, _tid, r, dk, _name, _weight in entries:
            reps.setdefault(dk, r)
        rep_rewards = list(reps.values())
        if len(rep_rewards) < 2:
            continue
        mu, sig = _mean_std(rep_rewards)
        if sig > delta:
            group_stats[key] = (mu, sig, len(entries))

    layer_weights = {key: 1.0 for key in group_stats}
    if balance_interaction_layers:
        cap = max(float(interaction_balance_cap), 1.0)
        by_context = defaultdict(list)
        for key, (_mu, _sig, count) in group_stats.items():
            role, sigma, is_resp, channel, layer_key = key
            if role == role_key("proposer") and not is_resp and channel == "int":
                by_context[(role, sigma, is_resp, channel)].append(
                    (key, count, layer_key))
        for layers in by_context.values():
            if len({layer_key for _key, _count, layer_key in layers}) < 2:
                continue
            target = sum(count for _key, count, _layer in layers) / len(layers)
            for key, count, _layer in layers:
                raw = target / max(count, 1)
                layer_weights[key] = min(cap, max(1.0 / cap, raw))

    step_channels: dict = defaultdict(dict)
    for key, entries in anchor_groups.items():
        if key not in group_stats:
            continue
        mu, sig, _count = group_stats[key]
        layer_weight = layer_weights.get(key, 1.0)
        for ep_idx, tid, r, _dk, credit_name, weight in entries:
            value = weight * layer_weight * ((r - mu) / sig)
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
                spec = {k: channels[k] for k in ("solution", "interaction")
                        if k in channels}
                if spec:
                    per_ep_adv[ep_idx][tid] = spec
            else:
                per_ep_adv[ep_idx][tid] = sum(channels.values())
    return per_ep_adv
