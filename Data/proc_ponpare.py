import pandas as pd
import numpy as np
import pickle as pkl

def processing_data(user_list_file, coupon_list_file, detail_file):
    # Read the dataset
    user_pd = pd.read_csv(user_list_file) 
    coupon_pd = pd.read_csv(coupon_list_file)
    detail_pd = pd.read_csv(detail_file)
    # Merge datasets
    D = pd.merge(detail_pd, user_pd, how='left', on='USER_ID_hash')
    D = pd.merge(D, coupon_pd, how='left', on='COUPON_ID_hash')
    D = D.drop_duplicates(subset=['USER_ID_hash', 'COUPON_ID_hash'], keep='first')

    # Data cleaning and filtering
    D = D[['REG_DATE','SEX_ID','AGE','PREF_NAME', 'ITEM_COUNT', 'CATALOG_PRICE','DISCOUNT_PRICE','COUPON_ID_hash','USER_ID_hash']+['CAPSULE_TEXT','GENRE_NAME','DISPPERIOD','VALIDPERIOD','USABLE_DATE_MON','USABLE_DATE_TUE','USABLE_DATE_WED','USABLE_DATE_THU','USABLE_DATE_FRI','USABLE_DATE_SAT','USABLE_DATE_SUN','USABLE_DATE_HOLIDAY','USABLE_DATE_BEFORE_HOLIDAY','large_area_name','ken_name','small_area_name']]
    
    # Reg_name: given a global datatime range, normalize the data
    D['REG_DATE'] = pd.to_datetime(D['REG_DATE'])
    D['REG_DATE'] = (D['REG_DATE'] - D['REG_DATE'].min()) / (D['REG_DATE'].max() - D['REG_DATE'].min())
    # Pref_name: one-hot encoding
    pref_map = {name: i for i, name in enumerate(D['PREF_NAME'].unique())}
    D['PREF_NAME'] = D['PREF_NAME'].map(pref_map)
    # sex_id: one-hot encoding
    sex_map = {'f': 0, 'm': 1}
    D['SEX_ID'] = D['SEX_ID'].map(sex_map)


    # get Treatment column
    D['TREATMENT'] = (1 - D['DISCOUNT_PRICE'] / D['CATALOG_PRICE']) * D['ITEM_COUNT']
    D['TREATMENT'] = D['TREATMENT'] / D['TREATMENT'].max()
    #D['TREATMENT'] = (D['CATALOG_PRICE'] - D['DISCOUNT_PRICE'])*D['ITEM_COUNT']

    # 1 - Discount_price / catalog_price

    cohort_column_name = 'TREATMENT' ## column name 
    treatment_indicator_value = 1.0 
    control_indicator_value = 0.0 

    intensity_col = D[cohort_column_name].values

    w_median = np.median(D[cohort_column_name].values) 
    D[cohort_column_name] = D[cohort_column_name].apply(lambda x: 1.0 if x > w_median else 0.0) 

    # Define feature and label lists
    feature_list = D.keys()[0:4].tolist()
    coupon_feature_list = ['CAPSULE_TEXT','GENRE_NAME','DISPPERIOD','VALIDPERIOD','USABLE_DATE_MON','USABLE_DATE_TUE','USABLE_DATE_WED','USABLE_DATE_THU','USABLE_DATE_FRI','USABLE_DATE_SAT','USABLE_DATE_SUN','USABLE_DATE_HOLIDAY','USABLE_DATE_BEFORE_HOLIDAY','large_area_name','ken_name','small_area_name']
    to_map_feature_list = ['CAPSULE_TEXT','GENRE_NAME','large_area_name','ken_name','small_area_name']
    
    for to_map_feature in to_map_feature_list:
        cur_map = {name: i for i, name in enumerate(D[to_map_feature].unique())}
        D[to_map_feature] = D[to_map_feature].map(pref_map)

    # Value
    D['Reward'] = D['ITEM_COUNT'] #* D['DISCOUNT_PRICE']
    # Cost
    D['Cost'] = (0.9 * D['CATALOG_PRICE']-D['DISCOUNT_PRICE']) * D['ITEM_COUNT'] 

    # Remove rows with COST < 0
    D = D[D['Cost'] > 0]

    label_list = ['Reward', 'Cost'] 

    D = D.fillna(-0.0) ## fill all Nan's with 0.0 

    # Standardize features and labels
    for l in feature_list + coupon_feature_list: 
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
    nXc = D[coupon_feature_list].to_numpy()

    w = D[cohort_column_name].apply(lambda x: 1.0 if x == treatment_indicator_value else 0.0).to_numpy()
    values = D[label_list[0]].to_numpy()
    negcost = D[label_list[1]].to_numpy()

    # Split train/val/test sets
    nX_tr = nX[:len_tr, :] # user features
    nX_va = nX[len_tr:len_tr + len_va, :] 
    nX_te = nX[len_tr + len_va:, :] 

    nXc_tr = nXc[:len_tr, :]
    nXc_va = nXc[len_tr:len_tr + len_va, :]
    nXc_te = nXc[len_tr + len_va:, :]

    w_tr = w[:len_tr]
    w_va = w[len_tr:len_tr + len_va] 
    w_te = w[len_tr + len_va:] 

    values_tr = values[:len_tr] 
    values_va = values[len_tr:len_tr + len_va] 
    values_te = values[len_tr + len_va:] 

    negcost_tr = negcost[:len_tr] # number of items (a * catelogue price - discount price), a = 75%
    negcost_va = negcost[len_tr:len_tr + len_va] 
    negcost_te = negcost[len_tr + len_va:] 

    i_tr = intensity_col[:len_tr]
    i_va = intensity_col[len_tr:len_tr + len_va]
    i_te = intensity_col[len_tr + len_va:]
    
    # # Save everything in a pickle file
    # with open('ponpare_data.pkl', 'wb') as f:
    #     pkl.dump({
    #         'nX_tr': nX_tr,
    #         'nX_va': nX_va,
    #         'nX_te': nX_te,
    #         'nXc_tr': nXc_tr,
    #         'nXc_va': nXc_va,
    #         'nXc_te': nXc_te,
    #         'w_tr': w_tr,
    #         'w_va': w_va,
    #         'w_te': w_te,
    #         'values_tr': values_tr,
    #         'values_va': values_va,
    #         'values_te': values_te,
    #         'negcost_tr': negcost_tr,
    #         'negcost_va': negcost_va,
    #         'negcost_te': negcost_te,
    #         'i_tr': i_tr,
    #         'i_va': i_va,
    #         'i_te': i_te
    #     }, f)

    # Return the matrices
    return nX_tr, nX_va, nX_te, w_tr, w_va, w_te, values_tr, values_va, values_te, negcost_tr, negcost_va, negcost_te, i_tr, i_va, i_te, nXc_tr, nXc_va, nXc_te