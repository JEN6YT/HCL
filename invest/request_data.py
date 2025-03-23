import pickle, pdb
from utils import get_finance_api_data, build_price_volume_chart_data

url = 'https://financialmodelingprep.com/stable/company-screener?marketCapMoreThan=10000000&exchange=NYSE&country=US&isEtf=false&limit=1000000000'
nyse_list = get_finance_api_data(url=url)

url = 'https://financialmodelingprep.com/stable/company-screener?marketCapMoreThan=10000000&exchange=NASDAQ&country=US&isEtf=false&isActiveTrading=True&limit=1000000000'
nasdaq_list = get_finance_api_data(url=url)

pickle.dump({"nyse_list":nyse_list, "nasdaq_list":nasdaq_list}, open('data/equity_lists.pkl', 'wb'))

D = pickle.load(open('data/equity_lists.pkl', 'rb'))
nyse_list = D['nyse_list']
nasdaq_list = D['nasdaq_list']

fiveyrs_start_date = '2020-03-21'
fiveyrs_end_date = '2025-03-21'

##### request all price data in the past 5 years 

#### request all price data for daily chart in the past 5 years 

nasdaq_list = nasdaq_list[:10]
print('Requesting all price data for daily chart in the past 5 years for NASDAQ...')
#url_str = 'historical-chart/15min'
url_str = 'historical-price-eod/light'
pickle.dump(build_price_volume_chart_data(nasdaq_list, fiveyrs_start_date, fiveyrs_end_date, url_str), open('data/nasdaq_daily_price_volume_data.pkl', 'wb'))

### request all price data for daily chart in the past 5 years for NYSE 
"""
print('Requesting all price data for 15 min chart in the past 5 years for NYSE...')
url_str = 'historical-chart/15min'
pickle.dump(build_price_volume_chart_data(nyse_list, fiveyrs_start_date, fiveyrs_end_date, url_str), open('data/nasdaq_15min_price_volume_data.pkl', 'wb'))

"""


"""
url = 'https://financialmodelingprep.com/stable/historical-chart/15min?symbol=AAPL&from=2025-03-01&to=2025-03-21'
res = get_finance_api_data(url=url)
res.reverse()

prices = BiHashList()
volumes = BiHashList()

for item in res:
    prices.append(item['date'], item['close'])
    volumes.append(item['date'], item['volume'])

#print(prices['2025-03-06 09:30:00'])
#print(volumes['2025-03-06 09:30:00'])

print(prices.return_ranged_value_list_from_keys('2025-03-06 09:30:00', '2025-03-21 12:30:00'))

"""
