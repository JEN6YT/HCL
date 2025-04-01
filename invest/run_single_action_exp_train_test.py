import torch, pdb, os
from train_single_step_model import train_single_step_model

exp_id = 'mar31_alleval_v1'
os.system('mkdir /home/ubuntu/code/HCL/invest/data/'+exp_id+'/')

data_list_f = open('/home/ubuntu/code/HCL/invest/data/data_list_2025_03_31_tr360d_bs4d_monthlyinterval.txt', 'r')
l = data_list_f.readline()
cnt = 1
while l:
    print('-->training model with: ' + l)
    train_single_step_model(
        exp_id,
        l.strip(),
        dropout_ratio = 0.0,
        obj_use_mean_return = False,
        steps = 750,
        lr = 0.001,
    )
    l = data_list_f.readline()
    cnt += 1

