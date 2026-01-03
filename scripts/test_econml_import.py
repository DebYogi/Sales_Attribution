"""Test EconML LinearDML import — intended to be run inside the econml-env kernel."""
try:
    from econml.dml import LinearDML
    print('LinearDML import OK')
except Exception as e:
    print('LinearDML import failed:', e)
    raise
