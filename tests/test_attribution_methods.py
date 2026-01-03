import pytest
import numpy as np
import pandas as pd

from sales_attribution import data
from sales_attribution.attribution import rule_based, markov, causal

try:
    from sales_attribution import data_controlled
    HAS_CONTROLLED = True
except Exception:
    HAS_CONTROLLED = False


def test_generate_journeys_and_agg():
    j = data.generate_user_journeys(n_users=200, max_steps=4)
    assert not j.empty
    assert set(['user_id', 'step', 'channel', 'converted']).issubset(j.columns)

    ts = data.generate_aggregate_timeseries(n_weeks=12)
    assert 'conversions' in ts.columns
    assert any(c.startswith('spend_') for c in ts.columns)


def test_rule_based_counts():
    j = data.generate_user_journeys(n_users=500)
    last = rule_based.last_touch_attribution(j)
    first = rule_based.first_touch_attribution(j)
    linear = rule_based.linear_attribution(j)
    # basic sanity checks
    assert isinstance(last, dict)
    assert isinstance(first, dict)
    assert isinstance(linear, dict)


def test_markov_transition_and_removal():
    j = data.generate_user_journeys(n_users=200)
    states, P = markov.transition_matrix(j)
    assert isinstance(states, list)
    assert P.shape[0] == P.shape[1]

    re = markov.removal_effect(j)
    # keys should include channels
    assert isinstance(re, dict)
    for ch, v in re.items():
        assert 'lost_conv' in v
        assert isinstance(v['lost_conv'], float)

    rf = markov.removal_effect_fraction(j, fraction=0.5)
    assert isinstance(rf, dict)


def test_causal_fallback_and_naive():
    j = data.generate_user_journeys(n_users=200)
    res = causal.estimate_incrementality_econml(j)
    assert isinstance(res, dict)
    # values should be numeric
    for v in res.values():
        assert isinstance(v, (int, float))

    naive = causal._naive_channel_ate(j)
    assert isinstance(naive, dict)


@pytest.mark.skipif(not HAS_CONTROLLED, reason='controlled data generator not available')
def test_controlled_detects_effects():
    effects = {'paid_search': 0.06, 'organic': 0.05}
    cj = data_controlled.generate_controlled_journeys(n_users=2000, channel_effects=effects)
    last = rule_based.last_touch_attribution(cj)
    # paid_search and organic should rank high
    paid = last.get('paid_search', 0)
    org = last.get('organic', 0)
    others = sum(v for k, v in last.items() if k not in ('paid_search', 'organic'))
    assert paid + org > others


def test_bayesian_interface_smoke():
    # Skip if pymc not installed to keep CI fast on minimal envs
    pytest.importorskip('pymc')
    from sales_attribution.attribution import bayesian_mmm
    ts = data.generate_aggregate_timeseries(n_weeks=20)
    trace, summary, ppc = bayesian_mmm.fit_bayesian_mmm(ts, draws=10, tune=10, fixed_decay=0.5)
    assert trace is not None
    assert summary is not None
