"""Causal / Incrementality estimators with EconML fallback logic."""
import numpy as np
import pandas as pd


def estimate_incrementality_econml(journeys: pd.DataFrame):
    """Estimate incremental effect per channel using EconML if available; fallback to naive ATE.

    For demonstration, we treat presence of channel in a user's path as a 'treatment' and estimate effect on conversion.
    Returns dict channel -> estimated incremental conversions.
    """
    try:
        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestRegressor
        HAS_ECONML = True
    except Exception as e:
        print("EconML not available or failed to import (numba/numpy conflict). Falling back to scikit-learn DML-style estimator.")
        HAS_ECONML = False

    results = {}
    # build user-level dataset
    df = journeys.groupby('user_id').agg({'converted':'max', 'channel': lambda x: list(x)}).reset_index()
    # create binary feature columns per channel indicating whether user was exposed
    channels = sorted({ch for lst in df['channel'] for ch in lst})
    for ch in channels:
        df[f"t_{ch}"] = df['channel'].apply(lambda lst: int(ch in lst))
    X = df[[f"t_{ch}" for ch in channels]]
    Y = df['converted']
    # No rich controls here; in practice include confounders. We'll use other channel exposures as controls.

    if HAS_ECONML:
        from sklearn.ensemble import RandomForestRegressor
        for i, ch in enumerate(channels):
            treat = X[[f"t_{ch}"]]
            other_t = X.drop(columns=[f"t_{ch}"])
            est = LinearDML(model_y=RandomForestRegressor(n_estimators=50, random_state=0), model_t=RandomForestRegressor(n_estimators=50, random_state=0))
            est.fit(Y, treat, X=other_t)
            te = est.effect(X=other_t).mean()
            n_exposed = (treat.values.flatten()==1).sum()
            results[ch] = float(te * n_exposed)
    else:
        # simple DML-style residualization using random forests from sklearn
        from sklearn.ensemble import RandomForestRegressor
        for ch in channels:
            t = df[f"t_{ch}"].values
            other = df[[c for c in X.columns if c != f"t_{ch}"]]
            # model t ~ other
            mt = RandomForestRegressor(n_estimators=50, random_state=0)
            mt.fit(other, t)
            t_hat = mt.predict(other)
            resid_t = t - t_hat
            # model y ~ other
            my = RandomForestRegressor(n_estimators=50, random_state=0)
            my.fit(other, Y)
            y_hat = my.predict(other)
            resid_y = Y - y_hat
            # estimate effect by regressing resid_y on resid_t (OLS closed-form)
            denom = (resid_t ** 2).sum()
            if denom == 0:
                coef = 0.0
            else:
                coef = (resid_t * resid_y).sum() / denom
            # scale to total users with exposure
            n_exposed = t.sum()
            results[ch] = float(coef * n_exposed)
    return results


def _naive_channel_ate(journeys: pd.DataFrame):
    df = journeys.groupby('user_id').agg({'converted':'max', 'channel': lambda x: list(x)}).reset_index()
    channels = sorted({ch for lst in df['channel'] for ch in lst})
    res = {}
    for ch in channels:
        treated = df[df['channel'].apply(lambda lst: ch in lst)]['converted']
        control = df[df['channel'].apply(lambda lst: ch not in lst)]['converted']
        res[ch] = float(treated.mean() - control.mean()) * len(df)
    return res
