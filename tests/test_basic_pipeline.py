import pytest
from sales_attribution import data
from sales_attribution.attribution import rule_based


def test_generate_journeys():
    df = data.generate_user_journeys(n_users=500)
    assert not df.empty
    assert 'channel' in df.columns


def test_rule_based_counts():
    df = data.generate_user_journeys(n_users=500)
    last = rule_based.last_touch_attribution(df)
    first = rule_based.first_touch_attribution(df)
    linear = rule_based.linear_attribution(df)
    assert isinstance(last, dict)
    assert isinstance(first, dict)
    assert isinstance(linear, dict)
