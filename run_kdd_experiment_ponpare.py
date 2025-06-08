import pandas as pd, numpy as np, matplotlib.pyplot as plt
#from Data.proc_us_census import preprocess_data 

from Data.proc_ponpare import processing_data

from Model.rlearner import RLearner
from Visualization.experimentation import Experiment 
import torch, pdb
import torch.nn.functional as F

ex = Experiment()
D_aucc = dict() 

# nX: user features
# w: T (whether user is treated or not)
# values: dIncome1 (reward)
# cost: iFertil (positive cost)

folder = '/ponpare_data/'
nX_tr, nX_va, nX_te, w_tr, w_va, w_te, values_tr, values_va, values_te, cost_tr, cost_va, cost_te, i_tr, i_va, i_te = processing_data(
    user_list_file = folder + 'user_list.csv',
    coupon_list_file = folder + 'coupon_list_train.csv',
    detail_file = folder + 'coupon_detail_train.csv'
)

# ----- rlearner ----- # 
rlearnermodel_O = RLearner(use_propensity=False) 
z = np.zeros([len(values_tr), 1]) 
o = np.concatenate((np.reshape(values_tr, [-1, 1]), z), axis=1) 
o = np.concatenate((o, np.reshape(w_tr, [-1, 1])), axis=1) 
rlearnermodel_O.fit(nX_tr, o) 

# Prediction
pred_values_va = rlearnermodel_O.tau_model.predict(nX_va)
# print(pred_values_va)

# Visualization
# Matrix: effectiveness score | incremental value | incremental cost

# [note - hack] plot random 
mplt, aucc_rnd, percs, cpits, cpitcohorts = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
    pred_values_va,
    values_va,
    w_va,
    -cost_va,
    'k',
    plot_random=True, ## this function only plots random and returns 0.5 aucc 
)
D_aucc['random'] = aucc_rnd 

print("random aucc: ", aucc_rnd)

# now plot r learner
mplt, aucc_rl, percs, cpits, cpitcohorts = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
    pred_values_va,
    values_va,
    w_va,
    -cost_va,
    'c',
)
D_aucc['rlearner'] = aucc_rl 

print("rlearner aucc: ", aucc_rl)
# note x forwarding is not working for pyplot.show()
# mplt.savefig('test_aucc_plot.png')

# Split data into treated and untreated
train_treat_index = np.where(w_tr==1)[0]
train_untreat_index = np.where(w_tr==0)[0]
treat_nX_tr = nX_tr[train_treat_index]
untreat_nX_tr = nX_tr[train_untreat_index]
treat_cost_tr = cost_tr[train_treat_index]
untreat_cost_tr = cost_tr[train_untreat_index]
treat_value_tr = values_tr[train_treat_index]
untreat_value_tr = values_tr[train_untreat_index]
treat_intensity_tr= i_tr[train_treat_index]
untreat_intensity_tr = i_tr[train_untreat_index]

val_treat_index = np.where(w_va==1)[0]
val_untreat_index = np.where(w_va==0)[0]
treat_nX_va= nX_va[val_treat_index]
untreat_nX_va = nX_va[val_untreat_index]
treat_cost_va = cost_va[val_treat_index]
untreat_cost_va = cost_va[val_untreat_index]
treat_value_va = values_va[val_treat_index]
untreat_value_va = values_va[val_untreat_index]
treat_intensity_va = i_va[val_treat_index]
untreat_intensity_va = i_va[val_untreat_index]

treat_nX_tr = torch.tensor(treat_nX_tr, dtype=torch.float32)
untreat_nX_tr = torch.tensor(untreat_nX_tr, dtype=torch.float32)
treat_cost_tr = torch.tensor(treat_cost_tr, dtype=torch.float32)
untreat_cost_tr = torch.tensor(untreat_cost_tr, dtype=torch.float32)
treat_value_tr = torch.tensor(treat_value_tr, dtype=torch.float32)
untreat_value_tr = torch.tensor(untreat_value_tr, dtype=torch.float32)
treat_intensity_tr= torch.tensor(treat_intensity_tr, dtype=torch.float32)
untreat_intensity_tr= torch.tensor(untreat_intensity_tr, dtype=torch.float32)

treat_nX_va= torch.tensor(treat_nX_va, dtype=torch.float32)
untreat_nX_va = torch.tensor(untreat_nX_va, dtype=torch.float32)
treat_cost_va = torch.tensor(treat_cost_va, dtype=torch.float32)
untreat_cost_va = torch.tensor(untreat_cost_va, dtype=torch.float32)
treat_value_va = torch.tensor(treat_value_va, dtype=torch.float32)
untreat_value_va = torch.tensor(untreat_value_va, dtype=torch.float32)
treat_intensity_va= torch.tensor(treat_intensity_va, dtype=torch.float32)
untreat_intensity_va= torch.tensor(untreat_intensity_va, dtype=torch.float32)

# ----- dual rlearner ----- # 
from Model.dual_rlearner_new import DualRLearner
drl = DualRLearner()

"""
# Selecting best lambda process

lambda_list = [0.001, 0.005, 0.01, 0.05]
colors = ['b', 'c', 'g', 'y']
result = drl.select_lambda(nX_tr, np.reshape(values_tr, [-1, 1]), np.reshape(cost_tr, [-1, 1]),  np.reshape(w_tr, [-1, 1]), lambda_list, nX_va)
labels = []

for i in range(len(result)):
    mplt_drl, aucc_drl, percs_drl, cpits_drl, cpitcohorts_drl = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
        result[i],
        values_va,
        w_va,
        cost_va,
        colors[i],
    )
    labels.append('lamd = ' + str(lambda_list[i]))

mplt_drl.legend(
    labels=labels,
    loc="upper right",  # Specify location of legend
    fontsize=10
)
# mplt_drl.savefig('selecting_lambda_dualrlearner_us.png')
"""


lmda = 5e-8 #0.00000005

# for covertype
# fitting_drl = drl.fit_dual(nX_tr, np.reshape(values_tr, [-1, 1]), np.reshape(cost_tr, [-1, 1]),  np.reshape(w_tr, [-1, 1]), lmda)
# for us census
fitting_drl = drl.fit_dual(nX_tr, np.reshape(values_tr, [-1, 1]), - np.reshape(cost_tr, [-1, 1]),  np.reshape(w_tr, [-1, 1]), lmda)
predicted_drl = fitting_drl.predict(nX_va)

mplt_drl, aucc_drl, percs_drl, cpits_drl, cpitcohorts_drl = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
    predicted_drl,
    values_va,
    w_va,
    -cost_va,
    'b',
)

# mplt_drl.savefig('dual_r.png')
D_aucc['duality_rlearer'] = aucc_drl 
print("duality aucc: ", aucc_drl)


# ----- DRM ----- # 

from Model.drm import *

drm_model = SimpleTCModelDNN(input_dim= 4, num_hidden= 32)

h_tre_rnkscore, h_unt_rnkscore = drm_model.forward(D_tre=treat_nX_tr, D_unt=untreat_nX_tr)

# Training
drm_epochs = 2000
save_path="model_drm.pth"

drm_obj = optimize_model(model=drm_model, 
                        D_tre=treat_nX_tr, 
                        D_unt=untreat_nX_tr, 
                        c_tre=treat_cost_tr, 
                        c_unt=untreat_cost_tr, 
                        o_tre=treat_value_tr, 
                        o_unt=untreat_value_tr,
                        epochs=drm_epochs)

torch.save(drm_model.state_dict(), save_path)
print(f"Model saved to {save_path}")

drm_model.load_state_dict(torch.load("model_drm.pth"))
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
    -cost_va,
    'mediumorchid',
)

D_aucc['drm'] = aucc_drm 
print("drm aucc: ", aucc_drm)
# mplt_drm.savefig('test_aucc_plot_drm.png')

# CTPM 
from Model import ctpm

ctpm_model = ctpm.CTPM(D_dim=4, num_hidden=32, temp=0, p_quantile=0, dropout_rate=0.0)

# Training
drm_epochs = 2000
save_path="model_ctpm.pth"

ctpm_obj = ctpm.optimize_ctpm_model(model=ctpm_model, 
                        D_tre=treat_nX_tr, 
                        D_unt=untreat_nX_tr, 
                        c_tre=treat_cost_tr, 
                        c_unt=untreat_cost_tr, 
                        o_tre=treat_value_tr, 
                        o_unt=untreat_value_tr,
                        i_tre=treat_intensity_tr, 
                        i_unt=untreat_intensity_tr,
                        epochs=drm_epochs)

torch.save(ctpm_model.state_dict(), save_path)
print(f"Model saved to {save_path}")

ctpm_model.load_state_dict(torch.load(save_path))
ctpm_model.eval()

# Prediction
_, _, _, h_tre_rnkscore_val, h_unt_rnkscore_val = ctpm_model(
    D_tre=treat_nX_va, 
    D_unt=untreat_nX_va, 
    o_tre=treat_value_va, 
    o_unt=untreat_value_va,
    c_tre=treat_cost_va,
    c_unt=untreat_cost_va,
    i_tre=treat_intensity_va,
    i_unt=untreat_intensity_va,
)
combined_scores = np.zeros_like(w_va, dtype=np.float32)
combined_scores[val_treat_index] = h_tre_rnkscore_val.detach().numpy().squeeze()
combined_scores[val_untreat_index] = h_unt_rnkscore_val.detach().numpy().squeeze()

mplt_ctpm, aucc_ctpm, percs_ctpm, cpits_ctpm, cpitcohorts_ctpm = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
    combined_scores,
    values_va,
    w_va,
    -cost_va,
    'r',
)

D_aucc['ctpm'] = aucc_ctpm 
print("ctpm aucc: ", aucc_ctpm)

mplt_ctpm.legend(
    labels=[
        "Random",
        "R-Learner",
        "Duality R-Learner",
        "Direct Ranking Model",
        "CTPM",
    ],
    loc="lower right",  # Specify location of legend
    fontsize=8
)
mplt_ctpm.savefig('ponpare_graphs.png')

print(D_aucc)
