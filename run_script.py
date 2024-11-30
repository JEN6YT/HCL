import pandas as pd, numpy as np, matplotlib.pyplot as plt
from Data.proc_us_census import preprocess_data 
from Model.rlearner import RLearner
from Visualization.experimentation import Experiment 
import torch

hcl_path = "/home/ubuntu/code/HCL/"

# Read US census csv
# us_census = pd.read_csv(hcl_path + "data/USCensus1990.data.txt", delimiter=",")
us_census = pd.read_csv("/Users/jenniferzhang/Desktop/Research with Will/USCensus1990.data.txt")
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

# nX: user features
# w: T (whether user is treated or not)
# values: dIncome1 (reward)
# cost: iFertil (positive cost)

# nX_tr, nX_va, nX_te, w_tr, w_va, w_te, values_tr, values_va, values_te, cost_tr, cost_va, cost_te = preprocess_data(hcl_path + 'data/USCensus1990.data.txt') 
nX_tr, nX_va, nX_te, w_tr, w_va, w_te, values_tr, values_va, values_te, cost_tr, cost_va, cost_te = preprocess_data("/Users/jenniferzhang/Desktop/Research with Will/USCensus1990.data.txt") 


# ----- rlearner ----- # 
rlearnermodel_O = RLearner()
z = np.zeros([len(values_tr), 1]) 
o = np.concatenate((np.reshape(values_tr, [-1, 1]), z), axis=1) 
o = np.concatenate((o, np.reshape(w_tr, [-1, 1])), axis=1)
rlearnermodel_O.fit(nX_tr, o) 

# Prediction
pred_values_va = rlearnermodel_O.tau_model.predict(nX_va)
# print(pred_values_va)

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


# ----- DRM ----- # 

from Model.drm import *

drm_model = SimpleTCModelDNN(input_dim= 46, num_hidden= 92)

# Split data into treated and untreated
train_treat_index = np.where(w_tr==1)[0]
train_untreat_index = np.where(w_tr==0)[0]
treat_nX_tr = nX_tr[train_treat_index]
untreat_nX_tr = nX_tr[train_untreat_index]
treat_cost_tr = cost_tr[train_treat_index]
untreat_cost_tr = cost_tr[train_untreat_index]
treat_value_tr = values_tr[train_treat_index]
untreat_value_tr = values_tr[train_untreat_index]


val_treat_index = np.where(w_va==1)[0]
val_untreat_index = np.where(w_va==0)[0]
treat_nX_va= nX_va[val_treat_index]
untreat_nX_va = nX_va[val_untreat_index]
treat_cost_va = cost_va[val_treat_index]
untreat_cost_va = cost_va[val_untreat_index]
treat_value_va = values_va[val_treat_index]
untreat_value_va = values_va[val_untreat_index]


treat_nX_tr = torch.tensor(treat_nX_tr, dtype=torch.float32)
untreat_nX_tr = torch.tensor(untreat_nX_tr, dtype=torch.float32)
treat_cost_tr = torch.tensor(treat_cost_tr, dtype=torch.float32)
untreat_cost_tr = torch.tensor(untreat_cost_tr, dtype=torch.float32)
treat_value_tr = torch.tensor(treat_value_tr, dtype=torch.float32)
untreat_value_tr = torch.tensor(untreat_value_tr, dtype=torch.float32)
treat_nX_va= torch.tensor(treat_nX_va, dtype=torch.float32)
untreat_nX_va = torch.tensor(untreat_nX_va, dtype=torch.float32)
treat_cost_va = torch.tensor(treat_cost_va, dtype=torch.float32)
untreat_cost_va = torch.tensor(untreat_cost_va, dtype=torch.float32)
treat_value_va = torch.tensor(treat_value_va, dtype=torch.float32)
untreat_value_va = torch.tensor(untreat_value_va, dtype=torch.float32)

h_tre_rnkscore, h_unt_rnkscore = drm_model.forward(D_tre=treat_nX_tr, D_unt=untreat_nX_tr)

# Training
drm_epochs = 10
save_path="model.pth"

for epoch in range(drm_epochs):
    drm_obj = optimize_model(model=drm_model, 
                            D_tre=treat_nX_tr, 
                            D_unt=untreat_nX_tr, 
                            c_tre=treat_cost_tr, 
                            c_unt=untreat_cost_tr, 
                            o_tre=treat_value_tr, 
                            o_unt=untreat_value_tr)
    print(f"Epoch {epoch + 1}/{drm_epochs}, Objective: {drm_obj.item()}")
    
torch.save(drm_model.state_dict(), save_path)
print(f"Model saved to {save_path}")

drm_model.load_state_dict(torch.load("model.pth"))
drm_model.eval()

# Prediction
h_tre_rnkscore_val, h_unt_rnkscore_val = drm_model(D_tre=treat_nX_va, D_unt=untreat_nX_va)
combined_scores = np.zeros_like(w_va, dtype=np.float32)
combined_scores[val_treat_index] = h_tre_rnkscore_val.detach().numpy().squeeze()
combined_scores[val_untreat_index] = h_unt_rnkscore_val.detach().numpy().squeeze()

mplt_drm, aucc_drm, percs_drm, cpits_drm, cpitcohorts_drm = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
    combined_scores,
    values_va,
    w_va,
    cost_va,
    'b',
)

mplt_drm.savefig('test_aucc_plot_drm.png')