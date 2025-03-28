import torch, pickle, pdb
from datetime import datetime, timedelta
from utils import find_closest_datetime
from data_proc import get_model_data

Dnyse = pickle.load(open('data/nyse_daily_price_volume_data.pkl', 'rb'))
Dnasdaq = pickle.load(open('data/nasdaq_daily_price_volume_data.pkl', 'rb'))

training_time_length = timedelta(days=360) #365 - 5 
buy_sell_time_length = timedelta(days=60)

data_start_date = '2020-03-21'
print('processing data for training...')
trainFeature, train_in_portfolio_series, all_train_tickers = get_model_data(Dnyse, Dnasdaq, data_start_date, training_time_length, buy_sell_time_length)
print('resulting shapes:')
print(trainFeature.shape)
print(train_in_portfolio_series.shape)
print(len(all_train_tickers))

data_start_date = '2021-05-25'
print('processing data for testing...')
testFeature, test_in_portfolio_series, all_test_tickers = get_model_data(Dnyse, Dnasdaq, data_start_date, training_time_length, buy_sell_time_length)
print('resulting shapes:')
print(testFeature.shape)
print(test_in_portfolio_series.shape)
print(len(all_test_tickers))

pickle.dump({"trainFeature":trainFeature, "train_in_portfolio_series":train_in_portfolio_series, "testFeature":testFeature, "test_in_portfolio_series":test_in_portfolio_series}, open('data/model_data_single_step_v1.pkl', 'wb'))
