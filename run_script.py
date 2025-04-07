import pandas as pd, numpy as np, matplotlib.pyplot as plt
from Data.proc_us_census import preprocess_data 
from Data.proc_covtype import process_data
from Model.rlearner import RLearner
from Visualization.experimentation import Experiment 
import torch
import torch.nn.functional as F

from Model.drm import *
from Model.percentile_barrier import *
from Model.dual_rlearner_new import DualRLearner
from Model.rlearner_mlp import mlprlearner
# from Model.rlearner_propensity import RLearner_propensity
# from Model.drm_propensity import *

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

    # nX: user features
    # w: T (whether user is treated or not)
    # values: dIncome1 (reward)
    # cost: iFertil (positive cost)

    # ----- random ----- #
    x_values = np.arange(0, 1.1, 0.1)
    y_values = x_values  # Since it's a diagonal line y = x

    # Plot the line with markers at every 0.1 interval
    plt.plot(x_values, y_values, color='k', linestyle='-', linewidth=3, marker='o', markersize=12)

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

    mplt, aucc, percs, cpits, cpitcohorts = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
        pred_values_va,
        values_va,
        w_va,
        cost_va,
        'c',
    )

    print("rlearner aucc: ", aucc)

    # note x forwarding is not working for pyplot.show()
    # mplt.savefig('test_aucc_plot.png')

    # # ----- rlearner with propensity ----- #

    # rlearnermodel_propensity = RLearner_propensity()
    # rlearnermodel_propensity.fit(nX_tr, o)
    # pred_values_va_propensity = rlearnermodel_propensity.tau_model.predict(nX_va)

    # mplt_propensity, aucc_propensity, percs_propensity, cpits_propensity, cpitcohorts_propensity = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
    #     pred_values_va_propensity,
    #     values_va,
    #     w_va,
    #     cost_va,
    #     'orange',
    # )

    # print("rlearner with propensity aucc: ", aucc_propensity)
    # mplt_propensity.savefig('test_aucc_plot_propensity.png')

    # ----- dual rlearner ----- # 
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


    lmda = 0.05

    # for covertype
    # fitting_drl = drl.fit_dual(nX_tr, np.reshape(values_tr, [-1, 1]), np.reshape(cost_tr, [-1, 1]),  np.reshape(w_tr, [-1, 1]), lmda)
    # for us census
    fitting_drl = drl.fit_dual(nX_tr, np.reshape(values_tr, [-1, 1]), - np.reshape(cost_tr, [-1, 1]),  np.reshape(w_tr, [-1, 1]), lmda)
    predicted_drl = fitting_drl.predict(nX_va)

    mplt_drl, aucc_drl, percs_drl, cpits_drl, cpitcohorts_drl = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
        predicted_drl,
        values_va,
        w_va,
        cost_va,
        'b',
    )

    # mplt_drl.savefig('dual_r.png')
    print("duality aucc: ", aucc_drl)

    # ----- Causal Tree ----- # 
    # file_path_o = "results_covtype/causal_forest_grf_test_set_results_O_numtrees60_alpha0.2_min_node_size4_sample_fraction0.5.csv"
    predicted_o = pd.read_csv(file_path_o)
    predicted_o = predicted_o.transpose()
    # Remove the header row by resetting the index and dropping the first row
    predicted_o = predicted_o.iloc[1:].reset_index(drop=True)
    predicted_o = predicted_o.to_numpy()
    predicted_o = predicted_o.astype(float)
    # Flatten the DataFrame to convert it into a Series
    # predicted_o = predicted_o.squeeze()

    # file_path_c = "results_covtype/causal_forest_grf_test_set_results_C_numtrees60_alpha0.2_min_node_size4_sample_fraction0.5.csv"
    predicted_c = pd.read_csv(file_path_c)
    predicted_c = predicted_c.transpose()
    predicted_c = predicted_c.iloc[1:].reset_index(drop=True)
    predicted_c = predicted_c.to_numpy()
    predicted_c = predicted_c.astype(float)
    # print(predicted_c)
    # print(predicted_c.dtype)

    mplt_ct, aucc_ct, percs_ct, cpits_ct, cpitcohorts_ct = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
        F.relu(torch.from_numpy(predicted_o))/(F.relu(torch.from_numpy(predicted_c))+1e-5),
        values_va,
        w_va,
        cost_va,
        'g',
    )
    print("causal tree aucc: ", aucc_ct)

    # mplt_ct.savefig('test_ct.png')


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



    # ----- Percentil Barrier Model ----- # 

    p_quantile = torch.tensor(0.4, dtype=torch.float32)
    initial_temperature = torch.tensor(3, dtype=torch.float32)

    pb_model = percentile_barrier_model(input_dim=input_dim, hidden_dim=number_of_hidden, initial_temp=initial_temperature, p_quantile=p_quantile)

    h_tre_rnkscore_pb, h_unt_rnkscore_pb  = pb_model.forward(D_tre=treat_nX_tr, D_unt=untreat_nX_tr)

    # Training
    pb_epochs = 1500
    save_path_pb="model_pb.pth"

    pb_obj = optimize_model_pb(model=pb_model, 
                                D_tre=treat_nX_tr, 
                                D_unt=untreat_nX_tr, 
                                c_tre=treat_cost_tr, 
                                c_unt=untreat_cost_tr, 
                                o_tre=treat_value_tr, 
                                o_unt=untreat_value_tr,
                                epochs=pb_epochs)
        
    torch.save(pb_model.state_dict(), save_path_pb)
    print(f"Model saved to {save_path_pb}")

    pb_model.load_state_dict(torch.load("model_pb.pth"))
    pb_model.eval()

    # Prediction
    h_tre_rnkscore_val_pb, h_unt_rnkscore_val_pb = pb_model(D_tre=treat_nX_va, D_unt=untreat_nX_va)
    combined_scores_pb = np.zeros_like(w_va, dtype=np.float32)
    combined_scores_pb[val_treat_index] = h_tre_rnkscore_val_pb.detach().numpy().squeeze()
    combined_scores_pb[val_untreat_index] = h_unt_rnkscore_val_pb.detach().numpy().squeeze()

    mplt_pb, aucc_pb, percs_pb, cpits_pb, cpitcohorts_pb = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
        combined_scores_pb,
        values_va,
        w_va,
        cost_va,
        'r',
    )
    print("percentile barrier aucc: ", aucc_pb)

    # mplt_pb.savefig('test_aucc_plot_pb.png')

    # ----- DRM ----- # 

    drm_model = SimpleTCModelDNN(input_dim= input_dim, num_hidden= number_of_hidden)

    # Training
    drm_epochs = 1500
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
        cost_va,
        'mediumorchid',
    )

    print("drm aucc: ", aucc_drm)
    # mplt_drm.savefig('test_aucc_plot_drm.png')


    # ----- DRM with propensity----- # 

    # drm_model_pro = SimpleTCModelDNN_propensity(input_dim= input_dim, num_hidden= number_of_hidden)

    # # Training
    # drm_epochs = 1500
    # save_path="model_drm_pro.pth"

    # drm_obj_pro = drm_model_pro.optimize_model_pro(model=drm_model_pro, 
    #                                                 X = nX_tr,
    #                                                 w = w_tr,
    #                                                 D_tre=treat_nX_tr, 
    #                                                 D_unt=untreat_nX_tr, 
    #                                                 c_tre=treat_cost_tr, 
    #                                                 c_unt=untreat_cost_tr, 
    #                                                 o_tre=treat_value_tr, 
    #                                                 o_unt=untreat_value_tr,
    #                                                 epochs=drm_epochs)

    # torch.save(drm_model_pro.state_dict(), save_path)
    # print(f"Model saved to {save_path}")

    # drm_model_pro.load_state_dict(torch.load("model_drm_pro.pth"))
    # drm_model_pro.eval()

    # # Prediction
    # h_tre_rnkscore_val_pro, h_unt_rnkscore_val_pro = drm_model_pro(D_tre=treat_nX_va, D_unt=untreat_nX_va)
    # combined_scores_pro = np.zeros_like(w_va, dtype=np.float32)
    # combined_scores_pro[val_treat_index] = h_tre_rnkscore_val_pro.detach().numpy().squeeze()
    # combined_scores_pro[val_untreat_index] = h_unt_rnkscore_val_pro.detach().numpy().squeeze()

    # mplt_drm_pro, aucc_drm_pro, percs_drm_pro, cpits_drm_pro, cpitcohorts_drm_pro = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
    #     combined_scores_pro,
    #     values_va,
    #     w_va,
    #     cost_va,
    #     'pink',
    # )

    # print("drm propensity aucc: ", aucc_drm_pro)
    # mplt_drm_pro.savefig('test_aucc_plot_drm_pro.png')


    # ----- rlearner 2layer mlp ----- # 

    rlearnermodel = mlprlearner()
    z = np.zeros([len(values_tr), 1])
    o = np.concatenate((np.reshape(values_tr, [-1, 1]), z), axis=1) 
    o = np.concatenate((o, np.reshape(w_tr, [-1, 1])), axis=1)
    rlearnermodel.fit(nX_tr, o)
    predicted_rlearner_mlp = rlearnermodel.predict(nX_va)

    mplt_mlp, aucc_mlp, percs_mlp, cpits_mlp, cpitcohorts_mlp = ex.AUC_cpit_cost_curve_deciles_cohort_vis(
        predicted_rlearner_mlp,
        values_va,
        w_va,
        cost_va,
        'y',
    )

    # mplt_drl.savefig('rlearner_mlp.png')
    print("rlearner mlp aucc: ", aucc_mlp)


    mplt_mlp.legend(
        labels=[
            "Random",
            "R-Learner on Incremental Gain",
            "Duality R-Learner",
            "Causal Forest",
            "Percentile Barrier Model",
            "Direct Ranking Model",
            # "Percentile Barrier Model Annealing",
            "R-Learner 2 Layer MLP"
        ],
        loc="lower right",  # Specify location of legend
        fontsize=8
    )
    mplt_mlp.savefig(save_fig_path)


if __name__ == "__main__":
    main()