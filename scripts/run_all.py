"""Run the full pipeline: generate data, run all attribution methods, and print comparisons."""
from sales_attribution import data
from sales_attribution.attribution import rule_based, markov, causal, bayesian_mmm
import pandas as pd


def run_demo():
    print("Generating synthetic journeys...")
    journeys = data.generate_user_journeys(n_users=20000)
    print("Running rule-based attribution...")
    last = rule_based.last_touch_attribution(journeys)
    first = rule_based.first_touch_attribution(journeys)
    linear = rule_based.linear_attribution(journeys)

    print("Running Markov attribution (removal effects)...")
    mr = markov.removal_effect(journeys)

    print("Running causal/incrementality estimator...")
    inc = causal.estimate_incrementality_econml(journeys)

    print("Generating aggregate timeseries for Bayesian MMM...")
    ts = data.generate_aggregate_timeseries()
    try:
        print("Fitting Bayesian MMM (this may take a bit)...")
        trace, summary = bayesian_mmm.fit_bayesian_mmm(ts, draws=300, tune=300)
    except Exception as e:
        trace, summary = None, None
        print("Bayesian fit failed (missing or heavy deps):", e)

    # assemble comparison table
    rows = []
    channels = sorted({r['channel'] if isinstance(r, dict) else None for r in []})
    # simplify: list channels from journeys
    channels = sorted(set(journeys['channel']))
    for ch in channels:
        rows.append({
            'channel': ch,
            'rule_last_touch': last.get(ch, 0),
            'rule_first_touch': first.get(ch, 0),
            'rule_linear': linear.get(ch, 0),
            'markov_lost_conv': mr.get(ch, {}).get('lost_conv', None),
            'incremental_conv': inc.get(ch, None),
        })
    comp = pd.DataFrame(rows).set_index('channel')
    print('\nComparison table (rows by channel):')
    print(comp)
    if summary is not None:
        print('\nBayesian MMM summary:')
        print(summary)

if __name__ == '__main__':
    run_demo()
