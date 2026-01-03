"""Rule-based attribution methods."""
import pandas as pd
from typing import Dict


def last_touch_attribution(journeys: pd.DataFrame) -> Dict[str, int]:
    """Assign conversions to the last channel in each converting user's path."""
    conv_df = journeys[journeys["converted"] == 1]
    last = conv_df.groupby("user_id").tail(1)
    return last["channel"].value_counts().to_dict()


def first_touch_attribution(journeys: pd.DataFrame) -> Dict[str, int]:
    conv_df = journeys[journeys["converted"] == 1]
    first = conv_df.groupby("user_id").head(1)
    return first["channel"].value_counts().to_dict()


def linear_attribution(journeys: pd.DataFrame) -> Dict[str, float]:
    """Split credit equally across channels in each converting path."""
    conv_users = journeys[journeys["converted"] == 1]["user_id"].unique()
    credited = {}
    for uid in conv_users:
        path = journeys[journeys["user_id"] == uid]["channel"].tolist()
        share = 1 / len(path)
        for ch in path:
            credited[ch] = credited.get(ch, 0) + share
    return credited
