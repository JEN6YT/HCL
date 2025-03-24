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

print('Requesting all price data for daily chart in the past 5 years for NYSE...')
url_str = 'historical-price-eod/light'
pickle.dump(build_price_volume_chart_data(nyse_list, fiveyrs_start_date, fiveyrs_end_date, url_str), open('data/nyse_daily_price_volume_data.pkl', 'wb'))

print('Requesting all price data for daily chart in the past 5 years for NASDAQ...')
#url_str = 'historical-chart/15min'
url_str = 'historical-price-eod/light'
pickle.dump(build_price_volume_chart_data(nasdaq_list, fiveyrs_start_date, fiveyrs_end_date, url_str), open('data/nasdaq_daily_price_volume_data.pkl', 'wb'))
