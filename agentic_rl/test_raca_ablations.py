"""CPU-only unit tests for RACA component ablations (Table `tab:raca-ablation`).

Verifies each ablation removes exactly one credit-assignment component and that
the non-ablated path is identical to the original estimator. No torch/GPU.
Run:  python -m pytest test_raca_ablations.py -q
"""

from training.raca_adv import compute_raca_advantages as raca_orig
from training.raca_ablations import (
    compute_raca_advantages_ablated,
    make_ablated_advantage_fn,
)


def _worker(role, reward, sigma="explore", is_response=False, rnd=0,
            layer_key=None):
    d = {"role": role, "reward": reward, "sigma": sigma,
         "is_response": is_response, "round": rnd}
    if layer_key is not None:
        d["layer_key"] = layer_key
    return d


def _proposer_primary(r_prop, r_int=None, sigma="explore", rnd=0,
                      layer_key=0, lambda_int=1.0):
    d = {"role": "proposer", "reward": r_prop, "sigma": sigma,
         "is_response": False, "round": rnd, "r_prop": r_prop,
         "token_credit": True, "lambda_int": lambda_int, "layer_key": layer_key}
    if r_int is not None:
        d["r_int"] = r_int
    return d


def _ctrl(reward):
    return {"role": "controller", "reward": reward, "round": 0}


# ── default path is unchanged ────────────────────────────────────────────────

def test_none_delegates_to_original():
    assert make_ablated_advantage_fn(None) is raca_orig
    assert make_ablated_advantage_fn("") is raca_orig


def test_default_path_byte_identical():
    # Two rollouts with critic + verifier response turns of different scales.
    ep = [
        {0: _ctrl(1.0),
         1: _worker("critic", 0.6, is_response=True, rnd=0),
         2: _worker("verifier", 0.9, is_response=True, rnd=0)},
        {0: _ctrl(-0.2),
         1: _worker("critic", -0.1, is_response=True, rnd=0),
         2: _worker("verifier", 0.3, is_response=True, rnd=0)},
    ]
    got = compute_raca_advantages_ablated(ep, ablation="", delta=1e-4)
    exp = raca_orig(ep, delta=1e-4)
    assert got == exp


# ── no_role: critic and verifier pooled into one comparison group ─────────────

def test_no_role_pools_roles():
    # Same reward for a critic and a verifier response under the same context.
    # With role conditioning they sit in singleton groups (dropped); without it
    # they share a group and both get a defined advantage.
    ep = [
        {0: _ctrl(1.0), 1: _worker("critic", 1.0, is_response=True)},
        {0: _ctrl(0.0), 2: _worker("verifier", 0.0, is_response=True)},
    ]
    raca = raca_orig(ep, delta=1e-4)
    # role-conditioned: each worker group has a single representative -> dropped
    assert 1 not in raca[0] and 2 not in raca[1]

    abl = compute_raca_advantages_ablated(ep, ablation="no_role", delta=1e-4)
    # pooled group {1.0, 0.0}: mean 0.5, std 0.5 -> advantages +1 / -1
    assert abs(abl[0][1] - 1.0) < 1e-9
    assert abs(abl[1][2] + 1.0) < 1e-9


# ── no_context: explore vs verify turns compared together ─────────────────────

def test_no_context_pools_contexts():
    ep = [
        {0: _ctrl(1.0), 1: _worker("critic", 1.0, sigma="explore",
                                    is_response=True)},
        {0: _ctrl(0.0), 2: _worker("critic", 0.0, sigma="verify",
                                    is_response=True)},
    ]
    raca = raca_orig(ep, delta=1e-4)
    # different sigma -> two singleton groups -> both dropped
    assert 1 not in raca[0] and 2 not in raca[1]

    abl = compute_raca_advantages_ablated(ep, ablation="no_context", delta=1e-4)
    assert abs(abl[0][1] - 1.0) < 1e-9
    assert abs(abl[1][2] + 1.0) < 1e-9


def test_no_context_keeps_role_separation():
    # Same context after ablation, but different roles+scales must NOT be pooled
    # (that would be the no_role ablation).  A critic 1.0 and verifier 1.0 with
    # no same-role partner each stay in a singleton group -> dropped.
    ep = [
        {0: _ctrl(1.0), 1: _worker("critic", 1.0, sigma="explore",
                                   is_response=True)},
        {0: _ctrl(0.0), 2: _worker("verifier", 0.0, sigma="verify",
                                   is_response=True)},
    ]
    abl = compute_raca_advantages_ablated(ep, ablation="no_context", delta=1e-4)
    assert 1 not in abl[0] and 2 not in abl[1]


# ── no_channel: proposer primary collapses to a single scalar advantage ───────

def test_no_channel_single_scalar_advantage():
    # Two proposer primary turns, same context, both with a solution and an
    # interaction reward.  RACA routes two channels (dict spec); the channel
    # ablation returns one scalar over the whole turn.
    ep = [
        {0: _ctrl(1.0), 1: _proposer_primary(r_prop=1.0, r_int=0.28,
                                             sigma="explore", layer_key=0)},
        {0: _ctrl(0.0), 1: _proposer_primary(r_prop=0.0, r_int=-0.10,
                                             sigma="explore", layer_key=0)},
    ]
    raca = raca_orig(ep, delta=1e-4)
    assert isinstance(raca[0][1], dict)  # RACA: structured solution/interaction

    abl = compute_raca_advantages_ablated(ep, ablation="no_channel", delta=1e-4)
    # combined rewards: 1.28 and -0.10 -> standardized to +1 / -1
    assert isinstance(abl[0][1], float)
    assert isinstance(abl[1][1], float)
    assert abs(abl[0][1] - 1.0) < 1e-9
    assert abs(abl[1][1] + 1.0) < 1e-9


def test_no_channel_respects_lambda_and_forced():
    # forced turn (r_int absent) keeps only r_prop in the combined reward.
    ep = [
        {0: _ctrl(1.0), 1: _proposer_primary(r_prop=1.0, r_int=0.5,
                                             lambda_int=2.0)},
        {0: _ctrl(0.0), 1: _proposer_primary(r_prop=0.0, r_int=None)},
    ]
    abl = compute_raca_advantages_ablated(ep, ablation="no_channel", delta=1e-4)
    # rewards: 1.0 + 2.0*0.5 = 2.0  vs  0.0 -> +1 / -1
    assert abs(abl[0][1] - 1.0) < 1e-9
    assert abs(abl[1][1] + 1.0) < 1e-9


def test_unknown_ablation_raises():
    import pytest
    with pytest.raises(ValueError):
        make_ablated_advantage_fn("no_such_mode")
    with pytest.raises(ValueError):
        compute_raca_advantages_ablated([], ablation="bogus")
