import numpy as np
import pandas as pd
import os

def process_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at: {path}")
    D = pd.read_csv(path, header=None)
    ### note integer column names for no-header direct csv reads 
    D1 = D[D[54] == 1] ## cover_type = Spruce-Fir 
    D2 = D[D[54] == 2] ## cover_type = Lodgepole Pine 

    D = pd.concat([D1, D2]) 
    e_median = np.median(D[0].values) 
    D = D[D[0] > e_median] 
    D = D.sample(frac=1.0) 

    cohort_column_name = 3 ## column name for distance to hydrology (meters) 
    treatment_indicator_value = 1.0 
    control_indicator_value = 0.0 

    w_median = np.median(D[cohort_column_name].values) 

    # get intensity
    intensity_col = D[cohort_column_name].values

    D[cohort_column_name] = D[cohort_column_name].apply(lambda x: 1.0 if x < w_median else 0.0) 

    ## feature take out 4 due to vert. hydro 
    feature_list = [ i for i in range(3)] + [i for i in range(5, 9)] + [i for i in range(10, 54)] 

    label_list = [ 
        9, ## distance to wild fire ignition points 
        54 ## Pine (2) vs Fir (1) 
        #3 ## distance to hydrology (meters) 
    ] 
    o_median = np.median(D[label_list[0]].values)
    D[label_list[0]] = D[label_list[0]].apply(lambda x: 1.0 if x < o_median else 0.0) #(lambda x: -1.0 * x / 100.0) # the reward is near wild fire starting points in hundres of meters 
    D[label_list[1]] = D[label_list[1]].apply(lambda x: 1.0 if x == 1 else 0.0) #(lambda x: 100.0 if x == 1 else 0.0) ## the cost is pine vs fir 
    
    # Standardize features and labels
    for l in feature_list: 
        D[l] = pd.to_numeric(D[l], errors='coerce') 
        D[l] = (D[l] - D[l].mean()) / D[l].std() 
        D[l][pd.isnull(D[l])] = 0.0 

    for l in label_list: 
        D[l] = pd.to_numeric(D[l], errors='coerce') 
        D[l][pd.isnull(D[l])] = 0.0 

    # Compute the treatment and control group statistics (cpit)
    treated_entries = D[D[cohort_column_name] == treatment_indicator_value] 
    untreated_entries = D[D[cohort_column_name] == control_indicator_value] 

    rpu_treated = float(treated_entries[label_list[0]].sum()) / len(treated_entries) 
    cipu_treated = float(treated_entries[label_list[1]].sum()) / len(treated_entries) 
    rpu_untreated = float(untreated_entries[label_list[0]].sum()) / len(untreated_entries) 
    cipu_untreated = float(untreated_entries[label_list[1]].sum()) / len(untreated_entries) 

    cpit = (cipu_treated - cipu_untreated) / (rpu_treated - rpu_untreated) 

    # Prepare matrices
    len_tr = len(D) // 5 * 3 
    len_va = len(D) // 5 

    nX = D[feature_list].to_numpy()
    w = D[cohort_column_name].apply(lambda x: 1.0 if x == treatment_indicator_value else 0.0).to_numpy()
    values = D[label_list[0]].to_numpy()
    negcost = D[label_list[1]].to_numpy() * -1.0

    # Split train/val/test sets
    nX_tr = nX[:len_tr, :] 
    nX_va = nX[len_tr:len_tr + len_va, :] 
    nX_te = nX[len_tr + len_va:, :] 

    w_tr = w[:len_tr]
    w_va = w[len_tr:len_tr + len_va] 
    w_te = w[len_tr + len_va:] 

    values_tr = values[:len_tr] 
    values_va = values[len_tr:len_tr + len_va] 
    values_te = values[len_tr + len_va:] 

    negcost_tr = negcost[:len_tr] 
    negcost_va = negcost[len_tr:len_tr + len_va] 
    negcost_te = negcost[len_tr + len_va:] 

    i_tr = intensity_col[:len_tr]
    i_va = intensity_col[len_tr:len_tr + len_va]
    i_te = intensity_col[len_tr + len_va:]

    # Return the matrices
    return nX_tr, nX_va, nX_te, w_tr, w_va, w_te, values_tr, values_va, values_te, negcost_tr, negcost_va, negcost_te, i_tr, i_va, i_te

