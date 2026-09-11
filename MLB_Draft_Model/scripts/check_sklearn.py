from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
print('All imports OK')
print(f'numpy version: {np.__version__}')
print(f'sklearn available: True')
