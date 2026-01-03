#!/bin/bash
# Create a conda environment for stable EconML (Linux/Mac)
# Requires conda to be available
conda create -n econml-env python=3.10 -y
conda activate econml-env
conda install -c conda-forge numpy=2.3 numba=0.57 scikit-learn -y
pip install econml arviz pymc

# Add kernel for Jupyter
python -m ipykernel install --user --name econml-env --display-name "Python (econml-env)"
echo "Done. Use the kernel 'Python (econml-env)' in Jupyter."