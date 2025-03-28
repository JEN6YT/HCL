import torch, pdb
from datetime import datetime, timedelta
from utils import find_closest_datetime

def doubleFilter(trainFeatures, dummy_tickers, all_tickers):

    revised_trainFeatures = torch.zeros((1, trainFeatures.shape[1]))
    revised_all_tickers = []
    for i  in range(len(all_tickers)): 
        ticker = all_tickers[i]
        if ticker not in dummy_tickers:
            continue
        revised_all_tickers.append(ticker)
        revised_trainFeatures = torch.cat([revised_trainFeatures, torch.unsqueeze(trainFeatures[i, :], 0)], dim=0)
    
    revised_trainFeatures = revised_trainFeatures[1:, :]
    return revised_trainFeatures, revised_all_tickers

def concat_features_from_exchange(D, start_training_date_str, end_training_date_str, sample_feature, trainFeatures, all_tickers, check_tickers=None):
    for ticker in D: 
        #print(ticker)
        if start_training_date_str not in D[ticker]['prices']._bD.keys() or end_training_date_str not in D[ticker]['prices']._bD.keys():
            continue
        cur_feature = D[ticker]['prices'].return_ranged_value_list_from_keys(start_training_date_str, end_training_date_str) 
        if len(cur_feature) != len(sample_feature):
            continue
        if check_tickers is not None and ticker not in check_tickers:
            continue
        trainFeatures = torch.cat([trainFeatures, torch.unsqueeze(torch.tensor(cur_feature).float(), 0)], dim=0)
        all_tickers.append(ticker)
    return trainFeatures, all_tickers

def get_model_data(Dnyse, Dnasdaq, data_start_date, training_time_length, buy_sell_time_length):
    date_format = "%Y-%m-%d" 

    start_training_date = datetime.strptime(data_start_date, date_format) 
    end_training_date = start_training_date+ training_time_length

    #datetime_object = datetime.datetime.strptime(start_training_date, date_format)
    #print(datetime_object)

    all_str_dates = Dnasdaq['AAPL']['prices']._bD.keys()
    all_datetime_dates = [datetime.strptime(date, date_format) for date in all_str_dates]

    #fact = Dnasdaq['AAPL']['prices']._bD.keys() == Dnyse['BRK-B']['prices']._bD.keys()
    #print(fact)

    start_training_date_str = find_closest_datetime(all_datetime_dates, start_training_date).strftime(date_format)[:11]
    end_training_date_closest = find_closest_datetime(all_datetime_dates, end_training_date)
    end_training_date_str = end_training_date_closest.strftime(date_format)[:11]
    buy_date_str = end_training_date_str
    sell_date_str = find_closest_datetime(all_datetime_dates, end_training_date_closest + buy_sell_time_length).strftime(date_format)[:11]

    print(start_training_date_str)
    print(end_training_date_str)
    print(buy_date_str)
    print(sell_date_str)
    
    ### ---- get price series as features ---- ###
    sample_feature = Dnasdaq['AAPL']['prices'].return_ranged_value_list_from_keys(start_training_date_str, end_training_date_str) 
    trainFeatures = torch.zeros((1, len(sample_feature)))
    all_train_tickers = []

    trainFeatures, all_train_tickers = concat_features_from_exchange(Dnasdaq, start_training_date_str, end_training_date_str, sample_feature, trainFeatures, all_train_tickers)
    trainFeatures, all_train_tickers = concat_features_from_exchange(Dnyse, start_training_date_str, end_training_date_str, sample_feature, trainFeatures, all_train_tickers)

    trainFeatures = trainFeatures[1:, :]

    sample_feature = Dnasdaq['AAPL']['prices'].return_ranged_value_list_from_keys(buy_date_str, sell_date_str)
    in_portfolio_series = torch.zeros((1, len(sample_feature)))
    dummy_tickers = []

    in_portfolio_series, dummy_tickers = concat_features_from_exchange(Dnasdaq, buy_date_str, sell_date_str, sample_feature, in_portfolio_series, dummy_tickers, all_train_tickers)
    in_portfolio_series, dummy_tickers = concat_features_from_exchange(Dnyse, buy_date_str, sell_date_str, sample_feature, in_portfolio_series, dummy_tickers, all_train_tickers)

    in_portfolio_series = in_portfolio_series[1:, :]

    trainFeatures, all_train_tickers = doubleFilter(trainFeatures, dummy_tickers, all_train_tickers)

    return trainFeatures, in_portfolio_series, all_train_tickers