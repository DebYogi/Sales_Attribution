"""Markov-chain attribution implementation (removal effect)."""
import pandas as pd
import numpy as np
from collections import defaultdict


def _build_transition_counts(journeys: pd.DataFrame):
    # For each user journey, produce transitions including start and conversion absorb state
    counts = defaultdict(lambda: defaultdict(int))
    for uid, grp in journeys.groupby("user_id"):
        path = grp.sort_values("step")["channel"].tolist()
        prev = "start"
        for ch in path:
            counts[prev][ch] += 1
            prev = ch
        counts[prev]["conversion"] += 1
    return counts


def transition_matrix(journeys: pd.DataFrame):
    counts = _build_transition_counts(journeys)
    states = sorted(set(k for k in counts.keys()) | {s for v in counts.values() for s in v.keys()})
    states = [s for s in states if s != "conversion"] + ["conversion"]
    idx = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    for i, s in enumerate(states):
        row = counts.get(s, {})
        total = sum(row.values())
        if total > 0:
            for t, c in row.items():
                j = idx[t]
                P[i, j] = c / total
        else:
            # absorbing conversion state
            P[i, idx[s]] = 1.0
    return states, P


def removal_effect(journeys: pd.DataFrame):
    """Compute removal effect: reduction in conversions when removing a channel.

    Returns a dict mapping channel -> (baseline_conv, after_removal_conv, lost_conv, removal_share)
    """
    states, P = transition_matrix(journeys)
    n = len(states)
    # identify conversion index
    conv_idx = states.index("conversion")
    # initial distribution is start state
    start_idx = states.index("start")
    # fundamental matrix approach for absorbing Markov chains
    # reorder so transient states first
    transient_idx = [i for i in range(n) if i != conv_idx]
    Q = P[np.ix_(transient_idx, transient_idx)]
    R = P[np.ix_(transient_idx, [conv_idx])]
    I = np.eye(Q.shape[0])
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)
    # baseline conversions per start
    # expected conversions starting from 'start' is (N*R)[start_pos,0]
    start_pos = transient_idx.index(start_idx)
    baseline_conv = (N @ R)[start_pos, 0]

    results = {}
    # for each removable channel, remove it from the chain by zeroing transitions INTO the channel and renormalizing rows
    for ch in [s for s in states if s not in ("start", "conversion")]:
        P2 = P.copy()
        jch = states.index(ch)
        # zero any incoming probability to the removed channel
        P2[:, jch] = 0
        # ensure conversion row remains absorbing
        P2[conv_idx, :] = 0
        P2[conv_idx, conv_idx] = 1.0
        # renormalize each non-conversion row to sum to 1 (if possible)
        for i in range(n):
            if i == conv_idx:
                continue
            s = P2[i, :].sum()
            if s > 0:
                P2[i, :] = P2[i, :] / s
            else:
                # if row has become empty, make it absorb to start to avoid losing mass
                P2[i, states.index("start")] = 1.0
        # recompute fundamental matrix for the modified chain
        transient_idx2 = [idx for idx in range(n) if idx != conv_idx]
        Q2 = P2[np.ix_(transient_idx2, transient_idx2)]
        R2 = P2[np.ix_(transient_idx2, [conv_idx])]
        I2 = np.eye(Q2.shape[0])
        try:
            N2 = np.linalg.inv(I2 - Q2)
        except np.linalg.LinAlgError:
            N2 = np.linalg.pinv(I2 - Q2)
        start_pos2 = transient_idx2.index(start_idx)
        after_conv = (N2 @ R2)[start_pos2, 0]
        lost = baseline_conv - after_conv
        results[ch] = {"baseline_conv": baseline_conv, "after_removal_conv": after_conv, "lost_conv": lost, "removal_share": lost / baseline_conv if baseline_conv>0 else 0}
    return results


def removal_effect_fraction(journeys: pd.DataFrame, fraction: float = 1.0):
    """Compute removal effect when removing a fraction of a channel's transitions.

    fraction = 1.0 means full removal (same as removal_effect), 0.5 means remove 50% of incoming probability to that channel.
    Returns dict channel -> same metrics as `removal_effect`.
    """
    if fraction <= 0:
        return {ch: {"baseline_conv": None, "after_removal_conv": None, "lost_conv": 0.0, "removal_share": 0.0} for ch in [s for s in transition_matrix(journeys)[0] if s not in ("start","conversion")]}

    states, P = transition_matrix(journeys)
    n = len(states)
    conv_idx = states.index("conversion")
    start_idx = states.index("start")

    transient_idx = [i for i in range(n) if i != conv_idx]
    Q = P[np.ix_(transient_idx, transient_idx)]
    R = P[np.ix_(transient_idx, [conv_idx])]
    I = np.eye(Q.shape[0])
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)
    start_pos = transient_idx.index(start_idx)
    baseline_conv = (N @ R)[start_pos, 0]

    results = {}
    for ch in [s for s in states if s not in ("start","conversion")]:
        P2 = P.copy()
        jch = states.index(ch)
        # scale down incoming probability to the channel by (1-fraction)
        removed_mass_per_row = P2[:, jch] * fraction
        P2[:, jch] = P2[:, jch] * (1.0 - fraction)
        # redistribute removed mass to start state to keep row sums at 1 where possible
        for i in range(n):
            if i == conv_idx:
                continue
            P2[i, start_idx] += removed_mass_per_row[i]
            # normalize row (exclude conversion row)
            s = P2[i, :].sum()
            if s > 0:
                P2[i, :] = P2[i, :] / s
            else:
                P2[i, start_idx] = 1.0
        # recompute absorbing matrix
        transient_idx2 = [idx for idx in range(n) if idx != conv_idx]
        Q2 = P2[np.ix_(transient_idx2, transient_idx2)]
        R2 = P2[np.ix_(transient_idx2, [conv_idx])]
        I2 = np.eye(Q2.shape[0])
        try:
            N2 = np.linalg.inv(I2 - Q2)
        except np.linalg.LinAlgError:
            N2 = np.linalg.pinv(I2 - Q2)
        start_pos2 = transient_idx2.index(start_idx)
        after_conv = (N2 @ R2)[start_pos2, 0]
        lost = baseline_conv - after_conv
        results[ch] = {"baseline_conv": baseline_conv, "after_removal_conv": after_conv, "lost_conv": lost, "removal_share": lost / baseline_conv if baseline_conv>0 else 0}
    return results
