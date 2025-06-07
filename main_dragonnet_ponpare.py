from dragonnet import DragonNet
from proc_ponpare import processing_data
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from utils import results_to_df, auuc_metric, auqc_metric, kendall_metric, lift_h_metric
import os.path
import torch

road='/home/ubuntu/code/revenue_uplift/datasets/Hillstrom'# 数据存放的地址

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    # Jobs dataset, args: 24, 200, 200.
    # IHDP dataset args: 24, 200 200

    dataset = 'Hillstrom_wom'
    model_name = 'dragonnets'
    
    set_random_seed(10) #10

    name = f"{dataset}_{model_name}"
    print(f"Output name: {name}")
    output_dir = '/home/ubuntu/code/revenue_uplift/dragonnets/output'
    path = os.path.join(output_dir, name)
    print('path', path)
    
    if not os.path.isdir(path):
        os.makedirs(path)

    """
    train_df =  pd.read_pickle(road + '/dt_binary_wom_tv_train.pkl')
    test_df =  pd.read_pickle(road + '/dt_binary_wom_test.pkl')
    
    in_features = ['recency', 'history_segment', 'mens', 'womens',
       'zip_code', 'newbie', 'channel_Multichannel', 'channel_Phone', 'channel_Web']
    label_feature = ['spend']
    treatment_feature = ['CASE']
    
    X_train = train_df[in_features].values.astype(float)
    y_train = train_df[label_feature].values.astype(float)
    t_train = train_df[treatment_feature].values.astype(float)

    X_test = test_df[in_features].values.astype(float)
    y_test = test_df[label_feature].values.astype(float)
    t_test = test_df[treatment_feature].values.astype(float)
    """
    folder = '/home/ubuntu/code/HCL/Data/coupon-purchase-prediction/'
    nX_tr, nX_va, nX_te, w_tr, w_va, w_te, values_tr, values_va, values_te, cost_tr, cost_va, cost_te, i_tr, i_va, i_te, _, _, _ = processing_data(
        user_list_file = folder + 'user_list.csv',
        coupon_list_file = folder + 'coupon_list_train.csv',
        detail_file = folder + 'coupon_detail_train.csv'
    )
    X_train = nX_tr
    y_train = values_tr 
    t_train = w_tr 

    X_test = nX_te 
    y_test = values_te 
    t_test = w_te 

    model = DragonNet(X_train.shape[1], epochs=25)
    model.fit(X_train, y_train, t_train)
    y0_pred, y1_pred, t_pred, _ = model.predict(X_test)
    
    label_feature = ['spend']
    treatment_feature = ['CASE']
    test_df = pd.DataFrame(
        {
            'spend': np.squeeze(values_te), 
            'CASE' : np.squeeze(w_te),
            'target_dif': np.squeeze(y1_pred - y0_pred), 
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
    