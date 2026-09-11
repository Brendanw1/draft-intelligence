#!/usr/bin/env python3
"""Check venv packages."""
import xgboost
import sklearn
import numpy
print(f"xgboost={xgboost.__version__} sklearn={sklearn.__version__} numpy={numpy.__version__}")
