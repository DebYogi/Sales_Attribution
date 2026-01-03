# Sales Attribution using Multiple method

This project demonstrates and compares five attribution approaches on a synthetic dataset:

- Rule-based attribution (first-touch, last-touch, linear)
- Markov-chain attribution (removal effect)
- Causal / Incrementality estimation (EconML-style / fallback)
- Bayesian Marketing Mix Model (PyMC)

Contents
- `src/sales_attribution` — library modules for data generation and each attribution method
- `notebooks/analysis.ipynb` — a notebook that runs the whole pipeline and compares outputs
- `scripts/run_all.py` — script to reproduce the analysis from the command line

Quick start
1. Create a conda/venv environment and install requirements:

   pip install -r requirements.txt

2. Run the demo notebook or the CLI script:

   python scripts/run_all.py

Notes
- Heavy dependencies (EconML, PyMC) are used for realistic examples; fallback implementations are provided when those libs are missing.
- The synthetic data generator creates journey-level logs and aggregated weekly time-series used for MMM.

## How to run

1. Install dependencies (recommended to use a virtual environment):

   pip install -r requirements.txt

2. If you want to run EconML's `LinearDML`, create a dedicated conda env with compatible NumPy/Numba (recommended):

   - Windows (run from a conda-enabled command prompt):

       scripts\setup_econml_env.bat

   - Mac/Linux:

       bash scripts/setup_econml_env.sh

   This will create a `econml-env` conda environment and add a Jupyter kernel named "Python (econml-env)" so you can run the notebook there.

3. Run the script to reproduce the full pipeline and print a comparison table:

   python scripts/run_all.py

4. Or open the Jupyter notebook `notebooks/analysis.ipynb` to step through the analysis interactively. If you've created the `econml-env` kernel, switch the notebook kernel to `Python (econml-env)` to run the EconML cells.

## Interpretation notes

- Rule-based methods are fast and intuitive but often mis-attribute conversions from assistive channels. Use them as quick baselines.
- Markov-chain attribution (removal effect) provides channel-level "lift" by estimating how many conversions are lost when each channel is removed. It captures path dynamics but ignores external confounding.
- Causal / Incrementality methods (EconML-style) attempt to estimate the causal uplift of exposure to a channel using double machine learning or uplift modeling; results depend strongly on available controls and assumptions.
- Bayesian MMM provides channel elasticities and uncertainty quantification at the aggregate level. It is well-suited for long-run budget allocation and modeling adstock/lagged effects.

## Next steps / caveats

- The synthetic data here is designed to illustrate differences; real data requires careful preprocessing, deduplication of users, and alignment of exposures and conversion windows.
- PyMC and EconML runs may be slow; tweak `draws` and `tune` in the Bayesian model for faster debugging.

---

If you'd like, I can (choose one):

- Run the notebook and generate a set of plots and an output report (PDF/HTML).
- Add example visualizations to the notebook comparing credit allocations across methods.
- Replace the PyMC model with a CmdStanPy model if you prefer CmdStan for speed.


License: MIT
