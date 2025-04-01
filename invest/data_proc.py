import torch, pdb, pickle
from datetime import datetime, timedelta
from utils import find_closest_datetime, read_json_file

def doubleFilter(trainFeatures, dummy_tickers, all_tickers):

    """
    Function used to double filter the features and all tickers with 'dummy_tickers'
    it makes sure the revised features and tickers is a set that is overlap of dummy_tickers and all_tickers
    """
    
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
    
    """
    with D as keyed dictionary with tickers, values as pricing data 
    The function extracts features per ticker, using start/end training date strings 
    and concatenate them into torch tensor, this tensor is used for model training/eval 
    """
    
    for ticker in D: 
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

def get_single_action_model_data(Dnyse, Dnasdaq, data_start_date, training_time_length, buy_sell_time_length):
    
    """
    Function to extract feature/label-series data given ticker-keyed Dictionary of features 
    the features are cut-out and built into tensors by calling the above concat_features_... function 
    and double filtered using the above doubleFilter function 
    """
    
    date_format = "%Y-%m-%d" 

    start_training_date = datetime.strptime(data_start_date.strip(), date_format) 
    end_training_date = start_training_date+ training_time_length

    #datetime_object = datetime.datetime.strptime(start_training_date, date_format)
    #print(datetime_object)

    all_str_dates = Dnasdaq['AAPL']['prices']._bD.keys()
    all_datetime_dates = [datetime.strptime(date.strip(), date_format) for date in all_str_dates]

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


def get_single_action_model_train_test_data(
        training_time_length_days: int,
        buy_sell_time_length_days: int,
        training_data_start_date: str,
        test_data_start_date: str,
        data_list_file: str,
):
    """
    Function to extract training/test data for single action model 
    
    The model takes in features as the price data for the length given in 'training_time_length_days'
    and use as label, the price data for the length given in 'buy_sell_time_legnth_days'
    
    Trainining data is extracted starting from 'training_data_start_date'
    Test data is extracted starting from 'test_data_start_date' 
    
    Calls the above get_single_action_model_data for respective dates and for training/testing 
    """
    ## --- load price data queried from financial modeling prep --- 
    Dnyse = pickle.load(open('data/nyse_daily_price_volume_data.pkl', 'rb'))
    Dnasdaq = pickle.load(open('data/nasdaq_daily_price_volume_data.pkl', 'rb'))
    
    ## --- all assets us equity and active that's tradeable with fractional shares on alpaca markets ---
    alpaca_list = read_json_file('data/alpaca_trading_us_equity_active.txt')

    ## --- simple filtering to produce a set that's tradeable on alpaca --- 
    print('length of alpaca_list:', len(alpaca_list))
    fractional_active_tradeable_list = []
    fractional_active_tradeable_keys = dict()
    for i in range(len(alpaca_list)):
        ticker = alpaca_list[i]
        if ticker['status'] == 'active' and \
           ticker['tradable'] == True and \
           ticker['fractionable'] == True and \
           (ticker['exchange'] == 'NYSE' or ticker['exchange'] == 'NASDAQ'): 
            fractional_active_tradeable_list.append(ticker)
            fractional_active_tradeable_keys[ticker['symbol']] = True
    print(len(fractional_active_tradeable_keys))

    Dnasdaq_new = dict()
    Dnyse_new = dict()
    for k in Dnasdaq.keys():
        if k in fractional_active_tradeable_keys:
            Dnasdaq_new[k] = Dnasdaq[k]
    for k in Dnyse.keys():
        if k in fractional_active_tradeable_keys:
            Dnyse_new[k] = Dnyse[k]
    
    print('pringint lengths of Dxxx:')
    print(len(Dnasdaq_new))
    print(len(Dnyse_new))
    print(len(Dnasdaq))
    print(len(Dnyse))
    
    ### --- specify the training vs label lengths --- 
    training_time_length = timedelta(days=training_time_length_days) #365 - 5 
    buy_sell_time_length = timedelta(days=buy_sell_time_length_days) #4
    
    ### --- obtain training data --- 
    #training_data_start_date = training_data_start_date #'2023-07-15' # v3: 2023-10-15 #v4: 2023-08-15
    print('processing data for training...')
    trainFeature, train_in_portfolio_series, all_train_tickers = get_single_action_model_data(Dnyse, Dnasdaq, training_data_start_date, training_time_length, buy_sell_time_length)
    print('resulting shapes:')
    print(trainFeature.shape)
    print(train_in_portfolio_series.shape)
    print(len(all_train_tickers))
    
    ### --- obtain test data --- 
    #test_data_start_date = test_data_start_date #'2023-10-01' # v3: 2024-01-01 #v4: 2023-11-01
    print('processing data for testing...')
    testFeature, test_in_portfolio_series, all_test_tickers = get_single_action_model_data(Dnyse_new, Dnasdaq_new, test_data_start_date, training_time_length, buy_sell_time_length)
    print('resulting shapes:')
    print(testFeature.shape)
    print(test_in_portfolio_series.shape)
    print(len(all_test_tickers))
    
    filename = f"/home/ubuntu/code/HCL/invest/data/model_data_single_step_trainingtimelength{str(training_time_length_days)}d_buyselltimelength{str(buy_sell_time_length_days)}d_training_data_start_date_{training_data_start_date.strip().replace('-', '_')}_test_data_start_date_{test_data_start_date.strip().replace('-', '_')}_alpacafracfiltered.pkl"
    print('saving to ... ' + filename)
    list_f = open(data_list_file, 'a')
    list_f.write(filename+'\n')
    list_f.close()
    
    pickle.dump({"trainFeature":trainFeature, "train_in_portfolio_series":train_in_portfolio_series, "all_train_tickers":all_train_tickers, "testFeature":testFeature, "test_in_portfolio_series":test_in_portfolio_series, "all_test_tickers":all_test_tickers}, open(filename, 'wb'))
