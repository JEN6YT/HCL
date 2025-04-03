import torch, pickle, hashlib
from datetime import datetime, timedelta
from utils import get_finance_api_data
from ts_data_struct import BiHashList

def combined_update():
    update_price_history_data()
    update_train_test_model_4d()    
    update_train_test_model_25d()
    update_predictions()

def update_price_history_data():
    ## determine start date for the update
    Dnasdaq = pickle.load(open('/home/ubuntu/code/HCL/invest/data/nasdaq_daily_price_volume_data.pkl', 'rb'))
    Dnyse = pickle.load(open('/home/ubuntu/code/HCL/invest/data/nyse_daily_price_volume_data.pkl', 'rb'))

    end_date_nasdaq = Dnasdaq['AAPL']['prices']._bD.inv[len(Dnasdaq['AAPL']['prices']._bD) - 1]
    end_date_nyse = Dnyse['GM']['prices']._bD.inv[len(Dnyse['GM']['prices']._bD) - 1]

    assert(end_date_nasdaq == end_date_nyse)
    start_date_update = datetime.strptime(end_date_nasdaq, "%Y-%m-%d") + timedelta(days=1)
    start_date_update = datetime.strftime(start_date_update, "%Y-%m-%d")[:10].strip()

    ## determine end date for the update
    end_date_update = datetime.strftime(datetime.now(), "%Y-%m-%d")[:10].strip()

    print('start_date_update: ' + start_date_update) 
    print('end_date_update: ' + end_date_update) 
    
    ## request data
    url_str = 'historical-price-eod/light'

    #nasdaq
    count = 0
    full_count = len(Dnasdaq)
    for symbol in Dnasdaq:
        count += 1
        print(f'{count}/{str(full_count)} Requesting price/volume data for {url_str} chart from {start_date_update} to {end_date_update} for {symbol} ...')
        url = f'https://financialmodelingprep.com/stable/{url_str}?symbol={symbol}&from={start_date_update}&to={end_date_update}'
        res = get_finance_api_data(url=url)
        if not res:
            print(f"Failed to get data for {symbol}")
            continue
        res.reverse()
        prices = BiHashList()
        volumes = BiHashList()
        for item in res:
            price_key = 'price' if 'light in url_str' else 'close'
            Dnasdaq[symbol]['prices'].append(item['date'], item[price_key])
            Dnasdaq[symbol]['volumes'].append(item['date'], item['volume'])

    #nyse
    count = 0
    full_count = len(Dnyse)
    for symbol in Dnyse:
        count += 1
        print(f'{count}/{str(full_count)} Requesting price/volume data for {url_str} chart from {start_date_update} to {end_date_update} for {symbol} ...')
        url = f'https://financialmodelingprep.com/stable/{url_str}?symbol={symbol}&from={start_date_update}&to={end_date_update}'
        res = get_finance_api_data(url=url)
        if not res:
            print(f"Failed to get data for {symbol}")
            continue
        res.reverse()
        prices = BiHashList()
        volumes = BiHashList()
        for item in res:
            price_key = 'price' if 'light in url_str' else 'close'
            Dnyse[symbol]['prices'].append(item['date'], item[price_key])
            Dnyse[symbol]['volumes'].append(item['date'], item['volume'])
    
    ## save data
    pdb.set_trace()
    pickle.dump(Dnasdaq, open('/home/ubuntu/code/HCL/invest/data/nasdaq_daily_price_volume_data.pkl', 'wb'))
    pickle.dump(Dnyse, open('/home/ubuntu/code/HCL/invest/data/nyse_daily_price_volume_data.pkl', 'wb'))
    

def update_train_test_model_4d():
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    h = hashlib.sha256(timestamp.encode()).hexdigest()
    data_list_file = "/home/ubuntu/code/HCL/invest/data/prod/data_list_tr360d_bs4d_prod_"+h+".txt"
    training_time_length_days = 360
    buy_sell_time_length_days = 4
    
    training_data_start_date = datetime.now() - timedelta(days=(training_time_length_days+buy_sell_time_length_days + 1))
    test_data_start_date = datetime.now() - timedelta(days=(training_time_length_days))
    
    print("--------- [prod update - 4d model] processing data for dates: ---------")
    print("training_data_start_date: "+training_data_start_date)
    print("test_data_start_date: "+test_data_start_date)
    get_single_action_model_train_test_data(
        training_time_length_days,
        buy_sell_time_length_days,
        training_data_start_date,
        test_data_start_date,
        data_list_file,
    )
    
    exp_id = 'prod_4d_'+h[:10]
    os.system('mkdir /home/ubuntu/code/HCL/invest/data/'+exp_id+'/')
    
    data_list_f = open(data_list_file, 'r')
    l = data_list_f.readline()
    print('-->training prod 4d model with: ' + l)
    train_single_step_model(
        exp_id,
        l.strip(),
        dropout_ratio = 0.0,
        obj_use_mean_return = True,
        steps = 750,
        lr = 0.001,
    )
    
def update_train_test_model_25d():
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    h = hashlib.sha256(timestamp.encode()).hexdigest()    
    data_list_file = "/home/ubuntu/code/HCL/invest/data/prod/data_list_tr360d_bs25d_prod_"+h+".txt"
    training_time_length_days = 360
    buy_sell_time_length_days = 25
    
    training_data_start_date = datetime.now() - timedelta(days=(training_time_length_days+buy_sell_time_length_days + 1))
    test_data_start_date = datetime.now() - timedelta(days=(training_time_length_days))
    
    print("--------- [prod update - 25d model] processing data for dates: ---------")
    print("training_data_start_date: "+training_data_start_date)
    print("test_data_start_date: "+test_data_start_date)
    get_single_action_model_train_test_data(
        training_time_length_days,
        buy_sell_time_length_days,
        training_data_start_date,
        test_data_start_date,
        data_list_file,
    )
    
    exp_id = 'prod_25d_'+h[:10]
    os.system('mkdir /home/ubuntu/code/HCL/invest/data/'+exp_id+'/')
    
    data_list_f = open(data_list_file, 'r')
    l = data_list_f.readline()
    print('-->training prod 4d model with: ' + l)
    train_single_step_model(
        exp_id,
        l.strip(),
        dropout_ratio = 0.0,
        obj_use_mean_return = True,
        steps = 750,
        lr = 0.001,
    )

def update_predictions():
    
