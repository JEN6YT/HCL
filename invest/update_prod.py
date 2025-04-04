import pickle, hashlib, os, pdb, torch
from train_single_step_model import train_single_step_model 
from datetime import datetime, timedelta 
from utils import get_finance_api_data, find_file_in_dir
from ts_data_struct import BiHashList 
from data_proc import get_single_action_model_train_test_data 
from model.iimodel import IIMODEL 

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
    if datetime.strptime(start_date_update, "%Y-%m-%d") >= datetime.strptime(end_date_update, "%Y-%m-%d"): 
        return
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
    pickle.dump(Dnasdaq, open('/home/ubuntu/code/HCL/invest/data/nasdaq_daily_price_volume_data.pkl', 'wb'))
    pickle.dump(Dnyse, open('/home/ubuntu/code/HCL/invest/data/nyse_daily_price_volume_data.pkl', 'wb'))
    

def update_train_test_model_4d():
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    #h = hashlib.sha256(timestamp.encode()).hexdigest()
    data_list_file = "/home/ubuntu/code/HCL/invest/data/prod/data_list_tr360d_bs4d_prod.txt"
    training_time_length_days = 360
    buy_sell_time_length_days = 4
    
    training_data_start_date = datetime.now() - timedelta(days=(training_time_length_days+buy_sell_time_length_days + 1))
    test_data_start_date = datetime.now() - timedelta(days=(training_time_length_days))
    training_data_start_date = datetime.strftime(training_data_start_date, "%Y-%m-%d")[:10].strip()
    test_data_start_date = datetime.strftime(test_data_start_date, "%Y-%m-%d")[:10].strip()

    print("--------- [prod update - 4d model] processing data for dates: ---------")
    print("training_data_start_date: " + training_data_start_date)
    print("test_data_start_date: " + test_data_start_date)
    get_single_action_model_train_test_data(
        training_time_length_days,
        buy_sell_time_length_days,
        training_data_start_date,
        test_data_start_date,
        data_list_file,
        is_prod = True,
    )
    
    exp_id = 'prod_4d_models'
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
        log_interval=250, 
        eval_interval=250,
        is_prod=True,
    )
    
def update_train_test_model_25d():
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    # h = hashlib.sha256(timestamp.encode()).hexdigest()    
    data_list_file = "/home/ubuntu/code/HCL/invest/data/prod/data_list_tr360d_bs25d_prod.txt"
    training_time_length_days = 360
    buy_sell_time_length_days = 25
    
    training_data_start_date = datetime.now() - timedelta(days=(training_time_length_days+buy_sell_time_length_days + 1))
    test_data_start_date = datetime.now() - timedelta(days=(training_time_length_days))
    training_data_start_date = datetime.strftime(training_data_start_date, "%Y-%m-%d")[:10].strip()
    test_data_start_date = datetime.strftime(test_data_start_date, "%Y-%m-%d")[:10].strip()

    print("--------- [prod update - 25d model] processing data for dates: ---------")
    print("training_data_start_date: " + training_data_start_date)
    print("test_data_start_date: " + test_data_start_date)
    
    get_single_action_model_train_test_data(
        training_time_length_days,
        buy_sell_time_length_days,
        training_data_start_date,
        test_data_start_date,
        data_list_file,
        is_prod = True,
    )
    
    exp_id = 'prod_25d_models'
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
        log_interval=250, 
        eval_interval=250,
        is_prod=True,
    )

def update_predictions():
    model = IIMODEL()
    data_list_file = "/home/ubuntu/code/HCL/invest/data/prod/data_list_tr360d_bs4d_prod.txt"
    f = open(data_list_file, 'r')
    data_filename = f.readline().strip()

    recent_test_data = pickle.load(open(data_filename, 'rb')) 
    test_features = recent_test_data['testFeature']
    test_tickers = recent_test_data['all_test_tickers']

    exp_id_4d = 'prod_4d_models'
    model_4d_filename = f'/home/ubuntu/code/HCL/invest/data/{exp_id_4d}/single_action_m_{exp_id_4d}_dropout0.0_objmeanretTrue_steps750_lr0.001_step750.pt'
    checkpoint = torch.load(model_4d_filename)
    model.load_state_dict(checkpoint)
    model.eval()
    D = dict()
    D['scores'] = model(test_features).squeeze().tolist()
    D['tickers'] = test_tickers
    pickle.dump(D, open('/home/ubuntu/code/HCL/invest/data/prod/prod_4d_model_prediction.pkl', 'wb'))


    exp_id_25d = 'prod_25d_models'
    model_25d_filename = f'/home/ubuntu/code/HCL/invest/data/{exp_id_25d}/single_action_m_{exp_id_25d}_dropout0.0_objmeanretTrue_steps750_lr0.001_step750.pt'
    checkpoint = torch.load(model_25d_filename)
    model.load_state_dict(checkpoint)
    model.eval()
    D = dict()
    D['scores'] = model(test_features).squeeze().tolist()
    D['tickers'] = test_tickers
    pickle.dump(D, open('/home/ubuntu/code/HCL/invest/data/prod/prod_25d_model_prediction.pkl', 'wb'))


if __name__ == '__main__':
    update_price_history_data()
    update_train_test_model_4d()    
    update_train_test_model_25d()
    update_predictions()
