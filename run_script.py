import pandas as pd, numpy as np, matplotlib.pyplot as plt
from Data.proc_us_census import preprocess_data 
from utils.linearHTE import linearHTE
from Model.rlearner import RLearner
from Visualization.experimentation import Experiment 

hcl_path = "/home/ubuntu/code/HCL/"

# Read US census csv
us_census = pd.read_csv(hcl_path + "data/USCensus1990.data.txt", delimiter=",")

# Check length and iFertil
# print(len(us_census))
# print(us_census['iFertil'].head(10))
"""
2458285
0    1
1    3
2    1
3    3
4    3
5    0
6    0
7    4
8    7
9    0
Name: iFertil, dtype: int64
"""

nX_tr, nX_va, nX_te, w_tr, w_va, w_te, values_tr, values_va, values_te, cost_tr, cost_va, cost_te = preprocess_data(hcl_path + 'data/USCensus1990.data.txt') 

l_hte = linearHTE()
value_tao, cost_tao = l_hte.fit_rlearner(nX_tr, values_tr, cost_tr, w_tr)

# Prediction
print(value_tao.predict(nX_va))
print(cost_tao.predict(nX_va))

ex = Experiment()

# Visualization
# Matrix: effectiveness score | incremental value | incremental cost

# aucc, percs, cpits, cpitcohorts = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
    
# )