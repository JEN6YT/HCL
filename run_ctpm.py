from Data.proc_ponpare import processing_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import results_to_df, auuc_metric, auqc_metric, kendall_metric, lift_h_metric
import os.path
import torch
from Model import ctpm

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    # Jobs dataset, args: 24, 200, 200.
    # IHDP dataset args: 24, 200 200

    dataset = 'ponpare'
    model_name = 'ctpm'

    set_random_seed(10) #10

    name = f"{dataset}_{model_name}"
    print(f"Output name: {name}")
    output_dir = '/home/ubuntu/HCL/output'
    path = os.path.join(output_dir, name)
    print('path', path)

    if not os.path.isdir(path):
        os.makedirs(path)


    folder = '/home/ubuntu/HCL/Data/coupon-purchase-prediction/'
    nX_tr, nX_va, nX_te, w_tr, w_va, w_te, values_tr, values_va, values_te, cost_tr, cost_va, cost_te, i_tr, i_va, i_te, nXc_tr, nXc_va, nXc_te = processing_data(
        user_list_file = folder + 'user_list.csv',
        coupon_list_file = folder + 'coupon_list_train.csv',
        detail_file = folder + 'coupon_detail_train.csv'
    )

    train_treat_index = np.where(w_tr==1)[0]
    train_untreat_index = np.where(w_tr==0)[0]
    treat_nX_tr = nX_tr[train_treat_index]
    untreat_nX_tr = nX_tr[train_untreat_index]
    treat_nXc_tr = nXc_tr[train_treat_index]
    untreat_nXc_tr = nXc_tr[train_untreat_index]
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
    treat_nXc_va = nXc_va[val_treat_index]
    untreat_nXc_va = nXc_va[val_untreat_index]
    treat_cost_va = cost_va[val_treat_index]
    untreat_cost_va = cost_va[val_untreat_index]
    treat_value_va = values_va[val_treat_index]
    untreat_value_va = values_va[val_untreat_index]
    treat_intensity_va = i_va[val_treat_index]
    untreat_intensity_va = i_va[val_untreat_index]

    treat_nX_tr = torch.tensor(treat_nX_tr, dtype=torch.float32)
    untreat_nX_tr = torch.tensor(untreat_nX_tr, dtype=torch.float32)
    treat_nXc_tr = torch.tensor(treat_nXc_tr, dtype=torch.float32)
    untreat_nXc_tr = torch.tensor(untreat_nXc_tr, dtype=torch.float32)
    treat_cost_tr = torch.tensor(treat_cost_tr, dtype=torch.float32)
    untreat_cost_tr = torch.tensor(untreat_cost_tr, dtype=torch.float32)
    treat_value_tr = torch.tensor(treat_value_tr, dtype=torch.float32)
    untreat_value_tr = torch.tensor(untreat_value_tr, dtype=torch.float32)
    treat_intensity_tr= torch.tensor(treat_intensity_tr, dtype=torch.float32)
    untreat_intensity_tr= torch.tensor(untreat_intensity_tr, dtype=torch.float32)

    treat_nX_va= torch.tensor(treat_nX_va, dtype=torch.float32)
    untreat_nX_va = torch.tensor(untreat_nX_va, dtype=torch.float32)
    treat_nXc_va = torch.tensor(treat_nXc_va, dtype=torch.float32)
    untreat_nXc_va = torch.tensor(untreat_nXc_va, dtype=torch.float32)
    treat_cost_va = torch.tensor(treat_cost_va, dtype=torch.float32)
    untreat_cost_va = torch.tensor(untreat_cost_va, dtype=torch.float32)
    treat_value_va = torch.tensor(treat_value_va, dtype=torch.float32)
    untreat_value_va = torch.tensor(untreat_value_va, dtype=torch.float32)
    treat_intensity_va= torch.tensor(treat_intensity_va, dtype=torch.float32)
    untreat_intensity_va= torch.tensor(untreat_intensity_va, dtype=torch.float32)
    # X_train = nX_tr
    # y_train = values_tr 
    # t_train = w_tr 

    # X_test = nX_te 
    # y_test = values_te 
    # t_test = w_te 

    ctpm_model = ctpm.CTPM(Da_dim=4, Db_dim=16, num_hidden=32, temp=0, p_quantile=0, dropout_rate=0.0)

    # Training
    drm_epochs = 1000
    save_path="model_ctpm.pth"

    ctpm_obj = ctpm.optimize_ctpm_model(model=ctpm_model, 
                            Da_tre=treat_nX_tr,
                            Da_unt=untreat_nX_tr,
                            Db_tre=treat_nXc_tr,
                            Db_unt=untreat_nXc_tr,
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

    _, _, result, _, _ = ctpm_model(
        Da_tre=treat_nX_tr,
        Da_unt=untreat_nX_tr,
        Db_tre=treat_nXc_tr,
        Db_unt=untreat_nXc_tr,
        o_tre=treat_value_va, 
        o_unt=untreat_value_va,
        c_tre=treat_cost_va,
        c_unt=untreat_cost_va,
        i_tre=treat_intensity_va,
        i_unt=untreat_intensity_va,
    )

    label_feature = ['spend']
    treatment_feature = ['CASE']
    test_df = pd.DataFrame(
        {
            'spend': np.squeeze(values_te), 
            'CASE' : np.squeeze(w_te),
            'target_dif': np.squeeze(result), 
        }
    )

    # AUUC (Area under Uplift Curve)
    auuc = auuc_metric(test_df,'target_dif', 100, treatment_feature, label_feature, path)

    # lift@30%
    lift_h = lift_h_metric(test_df,'target_dif', 100, treatment_feature, label_feature, h=0.3)

    # AUQC (Area under Qini Curve)
    auqc = auqc_metric(test_df,'target_dif', 100, treatment_feature, label_feature, path)

    # KRCC (Kendall Rank Correlation Coefficient)
    krcc = kendall_metric(test_df,'target_dif', 100, treatment_feature, label_feature)

    print('===========================  Test Results Summary ====================')
    print('auuc', auuc)
    print('lift_h', lift_h)
    print('auqc', auqc)
    print('krcc', krcc)