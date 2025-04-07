import pandas as pd, numpy as np, matplotlib.pyplot as plt
from Data.proc_us_census import preprocess_data 
from Data.proc_covtype import process_data
from Model.rlearner import RLearner
from Visualization.experimentation import Experiment 
import torch
import torch.nn.functional as F

from Model.rlearner import RLearner
from Model.drm_propensity_comparison import SimpleTCModelDNN

import argparse

def main():
    parser = argparse.ArgumentParser(description='Run the causal inference models')
    parser.add_argument('--data', type=str, choices=["us_census", "covtype"], help='Dataset to use')
    args = parser.parse_args()

    if args.data == "us_census":
        data_path = 'USCensus1990.data.txt'
        nX_tr, nX_va, nX_te, w_tr, w_va, w_te, values_tr, values_va, values_te, cost_tr, cost_va, cost_te = preprocess_data(data_path) 
        input_dim = 46
        number_of_hidden = 92
        save_fig_path = "us_census_graphs.png"
        file_path_o = "results_uscensus/causal_forest_grf_test_set_results_O_numtrees50_alpha0.2_min_node_size3_sample_fraction0.5.csv"
        file_path_c = "results_uscensus/causal_forest_grf_test_set_results_C_numtrees50_alpha0.2_min_node_size3_sample_fraction0.5.csv"
    else:
        data_path = "covtype.csv"
        nX_tr, nX_va, nX_te, w_tr, w_va, w_te, values_tr, values_va, values_te, cost_tr, cost_va, cost_te = process_data(data_path) 
        input_dim = 51
        number_of_hidden = 100
        save_fig_path = "covtype_graphs.png"
        file_path_o = "results_covtype/causal_forest_grf_test_set_results_O_numtrees60_alpha0.2_min_node_size4_sample_fraction0.5.csv"
        file_path_c = "results_covtype/causal_forest_grf_test_set_results_C_numtrees60_alpha0.2_min_node_size4_sample_fraction0.5.csv"

    ex = Experiment()

    # Define parameters for new objective
    beta = 1.3
    alpha = 1.5
    top_k_percent = [0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]

    # nX: user features
    # w: T (whether user is treated or not)
    # values: dIncome1 (reward)
    # cost: iFertil (positive cost)

    # # ----- rlearner ----- # 

    print("Training rlearner without propensity")

    # Rlearner for O
    rlearnermodel_O = RLearner(use_propensity=False)
    z = np.zeros([len(values_tr), 1]) 
    o = np.concatenate((np.reshape(values_tr, [-1, 1]), z), axis=1) 
    o = np.concatenate((o, np.reshape(w_tr, [-1, 1])), axis=1)
    rlearnermodel_O.fit(nX_tr, o) 

    # Rlearner for C
    c = np.concatenate((np.reshape(cost_tr, [-1, 1]), z), axis=1)
    c = np.concatenate((c, np.reshape(w_tr, [-1, 1])), axis=1)
    rlearnermodel_C = RLearner(use_propensity=False)
    rlearnermodel_C.fit(nX_tr, c)

    # Prediction

    e_hat = len(np.where(w_va==1)[0]) / (len(np.where(w_va==0)[0]) + len(np.where(w_va==1)[0]))
    # print(e_hat)
    e_x_o = rlearnermodel_O._fit_predict_p_hat(nX_va, w_va)
    e_x_c = rlearnermodel_C._fit_predict_p_hat(nX_va, w_va)

    values_va_weighted_tr = np.multiply(np.divide(e_hat, e_x_o), values_va)
    values_va_weighted_un = np.multiply(np.divide(1-e_hat, 1-e_x_o), values_va)
    cost_va_weighted_tr = np.multiply(np.divide(e_hat, e_x_c), cost_va)
    cost_va_weighted_un = np.multiply(np.divide(1-e_hat, 1-e_x_c), cost_va)

    values_va_weighted_tr = np.asarray(values_va_weighted_tr)
    values_va_weighted_un = np.asarray(values_va_weighted_un)
    cost_va_weighted_tr = np.asarray(cost_va_weighted_tr)
    cost_va_weighted_un = np.asarray(cost_va_weighted_un)

    all_indices = torch.arange(len(w_va)) # Get all indices

    for k in top_k_percent:
        pred_values_va = rlearnermodel_O.tau_model.predict(nX_va) - beta * rlearnermodel_C.tau_model.predict(nX_va)
        # Get top-k indices
        # Ensure it's a PyTorch tensor
        pred_values_va = torch.tensor(pred_values_va)
        k = max(1, int(len(pred_values_va) * (k / 100)))
        # Get top-k values
        top_k_values, top_k_indices = torch.topk(pred_values_va, k)
        
        threshold = top_k_values[-1]  # Get lowest score in top-k
        cut_indices = torch.where(pred_values_va >= threshold)[0]

        # Get treated indices (where cut_indices and w_va == 1)
        treated_selected_ind = cut_indices[(w_va[cut_indices] == 1)]

        untreated_selected_ind = torch.tensor(np.setdiff1d(all_indices.numpy(), treated_selected_ind.numpy()))

        prob_tr = F.softmax(pred_values_va[treated_selected_ind], dim=0)
        prob_un = F.softmax(pred_values_va[untreated_selected_ind], dim=0)
        prob_tr = np.asarray(prob_tr)
        prob_un = np.asarray(prob_un)
        obj = np.multiply(values_va_weighted_tr[treated_selected_ind], prob_tr).mean() \
                - np.multiply(values_va_weighted_un[untreated_selected_ind], prob_un).mean() \
                - beta * (np.multiply(cost_va_weighted_tr[treated_selected_ind], prob_tr).mean() \
                            - np.multiply(cost_va_weighted_un[untreated_selected_ind], prob_un).mean())

        print(obj)

    # ----- rlearner with propensity ----- #

    rlearnermodel_o_propensity = RLearner(use_propensity=True)
    rlearnermodel_o_propensity.fit(nX_tr, o)

    rlearnermodel_c_propensity = RLearner(use_propensity=True)
    rlearnermodel_c_propensity.fit(nX_tr, c)

    print("Rlearner with propensity")

    e_x_o = rlearnermodel_o_propensity._fit_predict_p_hat(nX_va, w_va)
    e_x_c = rlearnermodel_c_propensity._fit_predict_p_hat(nX_va, w_va)

    values_va_weighted_tr = np.multiply(np.divide(e_hat, e_x_o), values_va)
    values_va_weighted_un = np.multiply(np.divide(1-e_hat, 1-e_x_o), values_va)
    cost_va_weighted_tr = np.multiply(np.divide(e_hat, e_x_c), cost_va)
    cost_va_weighted_un = np.multiply(np.divide(1-e_hat, 1-e_x_c), cost_va)

    values_va_weighted_tr = np.asarray(values_va_weighted_tr)
    values_va_weighted_un = np.asarray(values_va_weighted_un)
    cost_va_weighted_tr = np.asarray(cost_va_weighted_tr)
    cost_va_weighted_un = np.asarray(cost_va_weighted_un)

    for k in top_k_percent:
        # print(rlearnermodel_o_propensity.tau_model.predict(nX_va))
        # print(rlearnermodel_c_propensity.tau_model.predict(nX_va))
        pred_values_va_pro = rlearnermodel_o_propensity.tau_model.predict(nX_va) - beta * rlearnermodel_c_propensity.tau_model.predict(nX_va)
        # Get top-k indices
        # Ensure it's a PyTorch tensor
        pred_values_va_pro = torch.tensor(pred_values_va_pro)

        # Get top-k values
        k = max(1, int(len(pred_values_va_pro) * (k / 100)))
        top_k_values, top_k_indices = torch.topk(pred_values_va_pro, k)

        threshold = top_k_values[-1]  # Get lowest score in top-k
        cut_indices = torch.where(pred_values_va_pro >= threshold)[0]

        # Get treated indices (where cut_indices and w_va == 1)
        treated_selected_ind = cut_indices[(w_va[cut_indices] == 1)]

        untreated_selected_ind = torch.tensor(np.setdiff1d(all_indices.numpy(), treated_selected_ind.numpy()))

        prob_tr = F.softmax(pred_values_va_pro[treated_selected_ind], dim=0)
        prob_un = F.softmax(pred_values_va_pro[untreated_selected_ind], dim=0)
        prob_tr = np.asarray(prob_tr)
        prob_un = np.asarray(prob_un)
        obj = np.multiply(values_va_weighted_tr[treated_selected_ind], prob_tr).mean() \
                - np.multiply(values_va_weighted_un[untreated_selected_ind], prob_un).mean() \
                - beta * (np.multiply(cost_va_weighted_tr[treated_selected_ind], prob_tr).mean() \
                            - np.multiply(cost_va_weighted_un[untreated_selected_ind], prob_un).mean())

        print(obj)



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


    # ----- DRM ----- # 

    drm_model = SimpleTCModelDNN(input_dim= input_dim, num_hidden= number_of_hidden)

    # Training
    drm_epochs = 1500
    save_path="model_drm.pth"

    drm_obj = drm_model.optimize_model(X=nX_tr,
                                        w=w_tr, 
                                        D_tre=treat_nX_tr, 
                                        D_unt=untreat_nX_tr, 
                                        c_tre=treat_cost_tr, 
                                        c_unt=untreat_cost_tr, 
                                        o_tre=treat_value_tr, 
                                        o_unt=untreat_value_tr,
                                        lr=0.0001,
                                        epochs=drm_epochs,
                                        alpha=alpha,
                                        use_propensity=False,
                                        top_k_percent=1.0)

    torch.save(drm_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    drm_model.load_state_dict(torch.load("model_drm.pth"))
    drm_model.eval()

    # Prediction
    for k in top_k_percent:
        drm_obj = drm_model.optimize_model(X=nX_va,
                                            w=w_va,
                                            D_tre=treat_nX_va, 
                                            D_unt=untreat_nX_va, 
                                            c_tre=treat_cost_va, 
                                            c_unt=untreat_cost_va, 
                                            o_tre=treat_value_va, 
                                            o_unt=untreat_value_va,
                                            lr=0.0001,
                                            epochs=1,
                                            alpha=alpha,
                                            use_propensity=True,
                                            top_k_percent=k)
        print(f"{k}: ", drm_obj)


    # ----- DRM with propensity----- # 

    drm_model_pro = SimpleTCModelDNN(input_dim= input_dim, num_hidden= number_of_hidden)

    # Training
    drm_epochs = 1500
    save_path="model_drm_pro.pth"

    drm_obj_pro = drm_model_pro.optimize_model(X = nX_tr,
                                                w = w_tr,
                                                D_tre=treat_nX_tr, 
                                                D_unt=untreat_nX_tr, 
                                                c_tre=treat_cost_tr, 
                                                c_unt=untreat_cost_tr, 
                                                o_tre=treat_value_tr, 
                                                o_unt=untreat_value_tr,
                                                lr=0.0001,
                                                epochs=drm_epochs,
                                                alpha=alpha,
                                                use_propensity=True,
                                                top_k_percent=1.0)

    torch.save(drm_model_pro.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    drm_model_pro.load_state_dict(torch.load("model_drm_pro.pth"))
    drm_model_pro.eval()


    # Prediction
    for k in top_k_percent:
        drm_pro_obj = drm_model_pro.optimize_model(X=nX_va,
                                                    w=w_va,
                                                    D_tre=treat_nX_va, 
                                                    D_unt=untreat_nX_va, 
                                                    c_tre=treat_cost_va, 
                                                    c_unt=untreat_cost_va, 
                                                    o_tre=treat_value_va, 
                                                    o_unt=untreat_value_va,
                                                    lr=0.0001,
                                                    epochs=1,
                                                    alpha=alpha,
                                                    use_propensity=True,
                                                    top_k_percent=k)
        print(f"{k}: ", drm_pro_obj)


if __name__ == "__main__":
    main()