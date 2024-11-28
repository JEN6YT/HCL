import pandas as pd
import numpy as np
import pickle as pkl

def preprocess_data(file_path):
    # Read the dataset
    D = pd.read_csv(file_path) 

    cohort_column_name = 'dHour89'
    treatment_indicator_value = 1.0 
    control_indicator_value = 0.0 

    # Data cleaning and filtering
    D = D[D['iFertil'] >= 1.5] 
    D = D[D['dAge'] < 5] 
    D = D[D['iCitizen'] == 0] 

    # Median transformation for the cohort column
    w_median = np.median(D[cohort_column_name].values) 
    D[cohort_column_name] = D[cohort_column_name].apply(lambda x: 1.0 if x > w_median else 0.0) 

    # Define feature and label lists
    feature_list = [ 
        'iAvail', 'iCitizen', 'iClass', 'dDepart', 'iDisabl1', 'iDisabl2', 'iEnglish', 
        'iFeb55', 'dIndustry', 'iKorean', 'iLang1', 'iLooking', 'iMay75880', 'iMeans', 
        'iMilitary', 'iMobility', 'iMobillim', 'dOccup', 'iOthrserv', 'iPerscare', 'dPOB', 
        'dPoverty', 'dPwgt1', 'dRearning', 'iRelat1', 'iRelat2', 'iRemplpar', 'iRiders', 
        'iRlabor', 'iRPOB', 'iRvetserv', 'iSchool', 'iSept80', 'iSex', 'iSubfam1', 'iSubfam2', 
        'iTmpabsnt', 'dTravtime', 'iVietnam', 'dWeek89', 'iWork89', 'iWorklwk', 'iWWII', 
        'iYearsch', 'iYearwrk', 'dYrsserv' 
    ]

    label_list = ['dIncome1', 'iFertil'] 

    # Process labels
    o_median = np.median(D[label_list[0]].values) 
    D[label_list[0]] = D[label_list[0]].apply(lambda x: float(x)) 
    D[label_list[1]] = D[label_list[1]].apply(lambda x: -1.0 * (x - 2.0)) 

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

    # Return the matrices
    return nX_tr, nX_va, nX_te, w_tr, w_va, w_te, values_tr, values_va, values_te, negcost_tr, negcost_va, negcost_te
