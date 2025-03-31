import torch, pickle, pdb
from datetime import datetime, timedelta
from utils import read_json_file
from data_proc import get_model_data

Dnyse = pickle.load(open('data/nyse_daily_price_volume_data.pkl', 'rb'))
Dnasdaq = pickle.load(open('data/nasdaq_daily_price_volume_data.pkl', 'rb'))

alpaca_list = read_json_file('data/alpaca_trading_us_equity_active.txt')
#print(alpaca_list)
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
Dnasdaq = Dnasdaq_new
Dnyse = Dnyse_new

training_time_length = timedelta(days=360) #365 - 5 
buy_sell_time_length = timedelta(days=60) 

data_start_date = '2023-10-15'
print('processing data for training...')
trainFeature, train_in_portfolio_series, all_train_tickers = get_model_data(Dnyse, Dnasdaq, data_start_date, training_time_length, buy_sell_time_length)
print('resulting shapes:')
print(trainFeature.shape)
print(train_in_portfolio_series.shape)
print(len(all_train_tickers))

data_start_date = '2024-01-01'
print('processing data for testing...')
testFeature, test_in_portfolio_series, all_test_tickers = get_model_data(Dnyse, Dnasdaq, data_start_date, training_time_length, buy_sell_time_length)
print('resulting shapes:')
print(testFeature.shape)
print(test_in_portfolio_series.shape)
print(len(all_test_tickers))

pickle.dump({"trainFeature":trainFeature, "train_in_portfolio_series":train_in_portfolio_series, "all_train_tickers":all_train_tickers, "testFeature":testFeature, "test_in_portfolio_series":test_in_portfolio_series, "all_test_tickers":all_test_tickers}, open('data/model_data_single_step_v3_alpacafracfiltered.pkl', 'wb'))
