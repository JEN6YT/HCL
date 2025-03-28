import pickle
from utils import get_finance_api_data
#from ts_data_struct import BiHashList

"""
url = 'https://financialmodelingprep.com/stable/company-screener?marketCapMoreThan=10000000&exchange=NYSE&country=US&isEtf=false&limit=1000000000'
nyse_list = get_finance_api_data(url=url)

url = 'https://financialmodelingprep.com/stable/company-screener?marketCapMoreThan=10000000&exchange=NASDAQ&country=US&isEtf=false&isActiveTrading=True&limit=1000000000'
nasdaq_list = get_finance_api_data(url=url)

pickle.dump({"nyse_list":nyse_list, "nasdaq_list":nasdaq_list}, open('data/equity_lists.pkl', 'wb'))
"""

# price data

#url = 'https://financialmodelingprep.com/stable/historical-chart/15min?symbol=QQQ&from=2025-03-01&to=2025-03-21'
url = 'https://financialmodelingprep.com/stable/historical-chart/15min?symbol=VOO&from=2025-03-01&to=2025-03-21'
res = get_finance_api_data(url=url)
res.reverse()

print(res)

# SPY also S&P 500 ETF
