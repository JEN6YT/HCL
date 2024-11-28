import pandas as pd, numpy as np, matplotlib.pyplot as plt
from Data.proc_us_census import preprocess_data 
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

rlearnermodel_O = RLearner()
z = np.zeros([len(values_tr), 1]) 
o = np.concatenate((np.reshape(values_tr, [-1, 1]), z), axis=1) 
o = np.concatenate((o, np.reshape(w_tr, [-1, 1])), axis=1)
rlearnermodel_O.fit(nX_tr, o) 

# Prediction
pred_values_va = rlearnermodel_O.tau_model.predict(nX_va)
print(pred_values_va)

ex = Experiment()

# Visualization
# Matrix: effectiveness score | incremental value | incremental cost

mplt, aucc, percs, cpits, cpitcohorts = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
    pred_values_va,
    values_va,
    w_va,
    cost_va,
    'r',
)




# note x forwarding is not working for pyplot.show()
mplt.savefig('test_aucc_plot.png')


