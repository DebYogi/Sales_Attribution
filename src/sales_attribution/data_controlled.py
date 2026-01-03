"""Additional data generator to create controlled experiments for validating attribution methods."""
from typing import List, Dict
import numpy as np
import pandas as pd

CHANNELS = ["email", "organic", "paid_search", "display", "social"]


def generate_controlled_journeys(n_users=5000, channel_effects: Dict[str, float]=None, max_steps=5, seed=0) -> pd.DataFrame:
    """Simulate user journeys where specified channels increase conversion probability.

    channel_effects: dict mapping channel -> additional conversion probability if present in the path.
    Returns DataFrame with ['user_id','step','channel','converted'].
    """
    rng = np.random.default_rng(seed)
    if channel_effects is None:
        channel_effects = {ch: 0.0 for ch in CHANNELS}
    rows = []
    for uid in range(n_users):
        k = rng.integers(1, max_steps + 1)
        path = rng.choice(CHANNELS, size=k, p=_channel_probs(len(CHANNELS)))
        prob = 0.01
        for ch in path:
            prob += channel_effects.get(ch, 0.0)
        converted = rng.random() < min(prob, 0.95)
        for step, ch in enumerate(path, 1):
            rows.append({"user_id": uid, "step": step, "channel": ch, "converted": int(converted)})
    return pd.DataFrame(rows)


def _channel_probs(m):
    base = np.array([0.15, 0.3, 0.25, 0.15, 0.15])
    return base[:m] / base[:m].sum()
