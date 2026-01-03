"""Synthetic data generators for attribution demos."""
from typing import List, Tuple
import numpy as np
import pandas as pd

CHANNELS = ["email", "organic", "paid_search", "display", "social"]


def generate_user_journeys(n_users=20000, max_steps=5, channels=CHANNELS, seed=42) -> pd.DataFrame:
    """Simulate user journeys as event sequences per user.

    Returns a DataFrame with columns ['user_id','step','channel','converted'] where converted is binary and only present on final step.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(n_users):
        k = rng.integers(1, max_steps + 1)
        path = rng.choice(channels, size=k, p=_channel_probs(len(channels)))
        # compute conversion probability biased by channels
        conv_prob = _path_conversion_prob(path)
        converted = rng.random() < conv_prob
        for step, ch in enumerate(path, 1):
            rows.append({"user_id": uid, "step": step, "channel": ch, "converted": int(converted)})
    return pd.DataFrame(rows)


def _channel_probs(m):
    # some arbitrary preference distribution
    base = np.array([0.15, 0.3, 0.25, 0.15, 0.15])
    return base[:m] / base[:m].sum()


def _path_conversion_prob(path: List[str]) -> float:
    # simple rule: organic has strong baseline, paid_search strong conversion, email small bump
    prob = 0.01
    for ch in path:
        if ch == "organic":
            prob += 0.03
        elif ch == "paid_search":
            prob += 0.04
        elif ch == "email":
            prob += 0.01
        elif ch == "display":
            prob += 0.008
        elif ch == "social":
            prob += 0.005
    # saturate
    return min(prob, 0.8)


def generate_aggregate_timeseries(n_weeks=104, channels=CHANNELS, seed=0) -> pd.DataFrame:
    """Generate weekly aggregated spend and conversions per channel for MMM.

    Returns a DataFrame indexed by week with columns for each channel's spend and the total conversions.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_weeks, freq="W")
    df = pd.DataFrame(index=dates)

    # simulate base spend patterns per channel
    base_spend = {
        "email": 2000,
        "organic": 1000,
        "paid_search": 8000,
        "display": 3000,
        "social": 2500,
    }

    for ch in channels:
        trend = np.linspace(0.8, 1.2, n_weeks)
        season = 1 + 0.15 * np.sin(np.linspace(0, 4 * np.pi, n_weeks) + rng.random())
        noise = rng.normal(0, 0.05, n_weeks)
        df[f"spend_{ch}"] = (base_spend[ch] * trend * season * (1 + noise)).clip(min=0)

    # conversions: apply adstock + log-elastic response
    adstocked = pd.DataFrame(index=df.index)
    decay = 0.5
    for ch in channels:
        spends = df[f"spend_{ch}"].values
        out = np.zeros_like(spends)
        s = 0
        for t, x in enumerate(spends):
            s = s * decay + x
            out[t] = s
        adstocked[ch] = out

    # true channel coefficients (unknown to models)
    true_coef = {"email": 0.0009, "organic": 0.0025, "paid_search": 0.004, "display": 0.001, "social": 0.0012}
    eps = rng.normal(0, 50, n_weeks)
    linear_pred = np.zeros(n_weeks)
    for ch in channels:
        linear_pred += true_coef[ch] * adstocked[ch].values
    # seasonality and baseline
    seasonality = 200 * (1 + 0.2 * np.sin(np.linspace(0, 6 * np.pi, n_weeks)))
    conversions = (linear_pred + seasonality + eps).clip(min=0)
    df["conversions"] = conversions.round().astype(int)
    return df
