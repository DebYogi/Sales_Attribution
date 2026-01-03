"""Bayesian Marketing Mix Modeling with PyMC."""
try:
    import pymc as pm
    HAS_PYMC = True
except Exception:
    pm = None
    HAS_PYMC = False
import pandas as pd
import numpy as np
try:
    import arviz as az
    HAS_ARVIZ = True
except Exception:
    az = None
    HAS_ARVIZ = False


def fit_bayesian_mmm(df: pd.DataFrame, channels=None, draws=1000, tune=1000, seed=0, fixed_decay=0.5):
    """Fit a simple Bayesian MMM with adstock and multiplicative (log) response.

    Returns the trace and summary. If PyMC is not available, returns (None, None).
    """
    if not HAS_PYMC:
        print("PyMC not available; skipping Bayesian MMM fit. Install pymc to enable this model.")
        return None, None

    if channels is None:
        channels = [c.replace('spend_', '') for c in df.columns if c.startswith('spend_')]
    spends = {ch: df[f"spend_{ch}"].values for ch in channels}
    conv = df['conversions'].values
    n = len(df)

    # Use a fixed decay value for adstock to avoid tensor-level set_subtensor issues
    decay = fixed_decay
    adstocks = {}
    for ch in channels:
        s = spends[ch]
        ad = _compute_adstock_np(s, decay)
        adstocks[ch] = np.log1p(ad)
    Xad = np.column_stack([adstocks[ch] for ch in channels])

    # Standardize adstocked predictors to improve sampling
    means = Xad.mean(axis=0)
    stds = Xad.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    Xstd = (Xad - means) / stds

    ppc = None
    with pm.Model() as model:
        sigma = pm.HalfNormal('sigma', 50)
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        sigma_beta = pm.HalfNormal('sigma_beta', 1.0)
        beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=len(channels))
        beta = pm.Deterministic('beta', beta_raw * sigma_beta)

        X_data = pm.Data('X_data', Xstd)
        mu = intercept + pm.math.dot(X_data, beta)
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=conv)
        try:
            trace = pm.sample(draws=draws, tune=tune, random_seed=seed, chains=4, cores=1, target_accept=0.95)
        except Exception as e:
            print('Sampling failed; attempting ADVI variational fit as fallback:', e)
            approx = pm.fit(n=5000, method='advi')
            trace = approx.sample(1000)

        # posterior predictive samples (ppc)
        try:
            ppc = pm.sample_posterior_predictive(trace, var_names=['obs'], model=model)
        except Exception:
            # API compatibility: try without model arg
            try:
                ppc = pm.sample_posterior_predictive(trace, var_names=['obs'])
            except Exception:
                ppc = None

    # Summarize posterior (beta is on standardized scale; convert to original adstock scale for interpretability)
    if HAS_ARVIZ:
        summary = az.summary(trace, var_names=['intercept', 'beta', 'sigma_beta'])
        # compute posterior mean betas and convert
        beta_means = summary.loc[[n for n in summary.index if n.startswith('beta')], 'mean'].values
        beta_orig_mean = beta_means / stds
        # attach to summary DataFrame
        bdf = pd.DataFrame({'beta_std_mean': beta_means, 'beta_orig_mean': beta_orig_mean}, index=channels)
        summary = {'arviz_summary': summary, 'beta_table': bdf}
    else:
        summary = None
    return trace, summary, ppc


def _compute_adstock_np(s, decay):
    out = np.zeros_like(s, dtype=float)
    prev = 0.0
    for t, x in enumerate(s):
        prev = decay * prev + x
        out[t] = prev
    return out
