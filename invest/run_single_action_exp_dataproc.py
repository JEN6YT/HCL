import torch, pickle, pdb 
from datetime import datetime, timedelta 
from data_proc import get_single_action_model_train_test_data 

date_format = "%Y-%m-%d"
datetime_format = "%Y-%m-%d %H:%M:%S"
data_list_file = "/home/ubuntu/code/HCL/invest/data/data_list_2025_03_31_tr360d_bs25d_monthlyinterval.txt"

abs_start_date = '2020-03-25' 
abs_stop_date = datetime.strptime('2025-03-20', date_format)

training_time_length_days = 360
buy_sell_time_length_days = 25

monthly_interval = timedelta(days=30)
yearly_interval = timedelta(days=360)
buy_sell_nonoverlap_interval = timedelta(days=30)

training_data_start_date = abs_start_date
test_data_start_date = (datetime.strptime(training_data_start_date, date_format) + buy_sell_nonoverlap_interval).strftime(datetime_format)[:11]

while datetime.strptime(test_data_start_date.strip(), date_format) + timedelta(days=training_time_length_days) + buy_sell_nonoverlap_interval < abs_stop_date:
    print("--------- processing dates: ---------")
    print("training_data_start_date: "+training_data_start_date)
    print("test_data_start_date: "+test_data_start_date)
    get_single_action_model_train_test_data(
        training_time_length_days,
        buy_sell_time_length_days,
        training_data_start_date,
        test_data_start_date,
        data_list_file,
    )
    training_data_start_date = (datetime.strptime(training_data_start_date.strip(), date_format) + monthly_interval).strftime(datetime_format)[:11].strip()
    test_data_start_date = (datetime.strptime(training_data_start_date.strip(), date_format) + buy_sell_nonoverlap_interval).strftime(datetime_format)[:11].strip()
    
