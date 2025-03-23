import pickle
from utils import get_finance_api_data
from ts_data_struct import BiHashList

"""
url = 'https://financialmodelingprep.com/stable/company-screener?marketCapMoreThan=10000000&exchange=NYSE&country=US&isEtf=false&limit=1000000000'
nyse_list = get_finance_api_data(url=url)

url = 'https://financialmodelingprep.com/stable/company-screener?marketCapMoreThan=10000000&exchange=NASDAQ&country=US&isEtf=false&isActiveTrading=True&limit=1000000000'
nasdaq_list = get_finance_api_data(url=url)

pickle.dump({"nyse_list":nyse_list, "nasdaq_list":nasdaq_list}, open('data/equity_lists.pkl', 'wb'))
"""

# price data

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
