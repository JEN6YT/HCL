# Please change the following to your own PAPER api key and secret
# or set them as environment variables (ALPACA_API_KEY, ALPACA_SECRET_KEY).
# You can get them from https://alpaca.markets/

api_key_25d = 'PK44IV49E4VZF809A5TU'
secret_key_25d = 'kgaLpj8OHinMn8ZpVEWQJdLATBLKB25FxKkcbmwc'

api_key_4d = 'PKC1SLBE8IF8QZM8XLEO'
secret_key_4d = 'uGVS94NLjmAjNao3i6stO7luJAdvITf8vU2ik5pB'

#### We use paper environment for this example ####
paper = True # Please do not modify this. This example is for paper trading only.
####

# Below are the variables for development this documents
# Please do not change these variables
trade_api_url = None
trade_api_wss = None
data_api_url = None
stream_data_wss = None

import alpaca, pickle, pdb

from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream

from alpaca.data.requests import (
    CorporateActionsRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetAssetsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import (
    AssetExchange,
    AssetStatus,
    OrderClass,
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
)

# to run async code in jupyter notebook
import nest_asyncio
from datetime import datetime 

def make_trade(symbol, shares_amount, trade_client, latest_price_on_record):

    # simple, market order, fractional qty
    # Alpaca trading platform support fractional trading by default
    # you can specify:
    # fractional qty (e.g. 0.01 qty) in the order request (which is shown in this example)
    # or notional value (e.g. 100 USD) (which is in the next example)
    #
    # If you have an error of `qty must be integer`,
    # please try to `Reset Account` of your paper account via the Alpaca Trading API dashboard
    req = MarketOrderRequest(
        symbol = symbol,
        qty = shares_amount,
        side = OrderSide.BUY,
        type = OrderType.MARKET,
        time_in_force = TimeInForce.DAY,
    )
    print(f"buying {symbol}, for {str(latest_price_on_record)} per share, buying {str(shares_amount)} amount of shares, and buying ${str(shares_amount * latest_price_on_record)} worth of shares.")
    res = trade_client.submit_order(req); print(res)

    
def make_portfolio_buy_25d(latest_price_D, trade_client, total_portfolio_usd_amount=10000): 
    D25d = pickle.load(open('/home/ubuntu/code/HCL/invest/data/prod/prod_25d_model_prediction.pkl', 'rb')) 
    scores = D25d['scores'] 
    tickers = D25d['tickers'] 
    trade_record = dict() 
    trade_record['shares_amount'] = dict()
    trade_record['usd_amount'] = dict()
    trade_record['exceptions'] = dict()
    for i in range(len(tickers)): 
        ticker = tickers[i] 
        latest_price = latest_price_D[ticker] 
        score = scores[i] 
        usd_amount = total_portfolio_usd_amount * score 
        if usd_amount < 1.0: ## this is changed to 1.0 as of 12:13pm 4.4.2025 
            trade_record['shares_amount'][ticker] = 0.0
            trade_record['usd_amount'][ticker] = 0.0
            trade_record['exceptions'][ticker] = 'usd amount smaller than $1.0'
            continue 
        shares_amount = usd_amount / latest_price 
        trade_record['shares_amount'][ticker] = shares_amount 
        trade_record['usd_amount'][ticker] = usd_amount
        try: 
            make_trade(ticker, shares_amount, trade_client, latest_price)
        except alpaca.common.exceptions.APIError as e:
            trade_record['exceptions'][ticker] = e
            print(f"--> ]Trade Error]: {e}")
        except Exception as e:
            trade_record['exceptions'][ticker] = e
            print(f"--> ]Trade Error]: {e}")

    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    record_filename = 'trade_record_25d_'+timestamp.replace(' ', '_')+'.pkl'
    pickle.dump(trade_record, open('/home/ubuntu/code/HCL/invest/data/prod/trades/'+record_filename, 'wb'))

def make_portfolio_buy_4d(latest_price_D, trade_client, total_portfolio_usd_amount=10000): 
    D4d = pickle.load(open('/home/ubuntu/code/HCL/invest/data/prod/prod_4d_model_prediction.pkl', 'rb')) 
    scores = D4d['scores'] 
    tickers = D4d['tickers'] 
    trade_record = dict() 
    trade_record['shares_amount'] = dict()
    trade_record['usd_amount'] = dict()
    for i in range(len(tickers)): 
        ticker = tickers[i] 
        latest_price = latest_price_D[ticker] 
        score = scores[i] 
        usd_amount = total_portfolio_usd_amount * score 
        if usd_amount < 1.0: ## this is changed to 1.0 as of 12:13pm 4.4.2025 
            trade_record['shares_amount'][ticker] = 0.0
            trade_record['usd_amount'][ticker] = 0.0
            trade_record['exceptions'][ticker] = 'usd amount smaller than $1.0'
            continue 
        shares_amount = usd_amount / latest_price 
        trade_record['shares_amount'][ticker] = shares_amount 
        trade_record['usd_amount'][ticker] = usd_amount
        try: 
            make_trade(ticker, shares_amount, trade_client, latest_price)
        except alpaca.common.exceptions.APIError as e:
            trade_record['exceptions'][ticker] = e
            print(f"--> ]Trade Error]: {e}")
        except Exception as e:
            trade_record['exceptions'][ticker] = e
            print(f"--> ]Trade Error]: {e}")
    
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    record_filename = 'trade_record_4d_'+timestamp.replace(' ', '_')+'.pkl'
    pickle.dump(trade_record, open('/home/ubuntu/code/HCL/invest/data/prod/trades/'+record_filename, 'wb'))
        

if __name__ == "__main__":
    total_portfolio_usd_amount_4d = 10000
    total_portfolio_usd_amount_25d = 10000
    nest_asyncio.apply()
    trade_client_25d = TradingClient(api_key=api_key_25d, secret_key=secret_key_25d, paper=paper, url_override=trade_api_url)
    trade_client_4d = TradingClient(api_key=api_key_4d, secret_key=secret_key_4d, paper=paper, url_override=trade_api_url)

    ## load and process latest prices 
    latest_price_D = dict() 
    Dnasdaq = pickle.load(open('/home/ubuntu/code/HCL/invest/data/nasdaq_daily_price_volume_data.pkl', 'rb'))
    Dnyse = pickle.load(open('/home/ubuntu/code/HCL/invest/data/nyse_daily_price_volume_data.pkl', 'rb'))
    for k in Dnasdaq: 
        latest_price_D[k] = Dnasdaq[k]['prices'][Dnasdaq[k]['prices']._bD.inv[len(Dnasdaq[k]['prices']._bD) - 1]]
        
    for k in Dnyse:
        if len(Dnyse[k]['prices']._bD) > 0:
            latest_price_D[k] = Dnyse[k]['prices'][Dnyse[k]['prices']._bD.inv[len(Dnyse[k]['prices']._bD) - 1]]
    
    #make_portfolio_buy_4d(latest_price_D, trade_client_4d, total_portfolio_usd_amount_4d)
    make_portfolio_buy_25d(latest_price_D, trade_client_25d, total_portfolio_usd_amount_25d)