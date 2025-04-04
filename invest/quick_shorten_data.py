import pickle 

Do = pickle.load(open('data/nasdaq_daily_price_volume_data_orig.pkl', 'rb'))
D = dict()
cnt = 0
D['AAPL'] = Do['AAPL']
for k in Do.keys():
    D[k] = Do[k]
    cnt += 1 
    if cnt == 22: 
        break

pickle.dump(D, open('data/nasdaq_daily_price_volume_data.pkl', 'wb'))

Do = pickle.load(open('data/nyse_daily_price_volume_data_orig.pkl', 'rb'))
D = dict()
cnt = 0
D['GM'] = Do['GM']
for k in Do.keys():
    D[k] = Do[k]
    cnt += 1 
    if cnt == 22: 
        break
pickle.dump(D, open('data/nyse_daily_price_volume_data.pkl', 'wb'))

