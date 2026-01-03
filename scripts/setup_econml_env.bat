@echo off
REM Create a conda environment for stable EconML (Windows)
REM Requires conda to be available on PATH
conda create -n econml-env python=3.10 -y
conda activate econml-env
conda install -c conda-forge numpy=2.3 numba=0.57 scikit-learn -y
pip install econml arviz pymc
echo\nEconML environment created. To use it in Jupyter, install ipykernel and add kernel:\n
python -m ipykernel install --user --name econml-env --display-name "Python (econml-env)"

echo Done.