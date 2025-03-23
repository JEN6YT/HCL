import requests, pdb
from datetime import datetime
from ts_data_struct import BiHashList

FINANCIAL_KEY = "1e347f859bc1eaa56334ad8c5dc10924"

def find_closest_datetime(datetime_list, target_datetime):
    """
    Finds the closest datetime to a target datetime in a list.

    Args:
        datetime_list: A list of datetime objects.
        target_datetime: The datetime object to find the closest to.

    Returns:
        The closest datetime object in the list to the target datetime, or None if the list is empty.
    """
    """
    # Example usage:
    dates = [
        datetime(2025, 3, 20),
        datetime(2025, 3, 22),
        datetime(2025, 3, 25),
        datetime(2025, 3, 28),
    ]
    target_date = datetime(2025, 3, 23)

    closest_date = find_closest_datetime(dates, target_date)
    print(f"The closest date to {target_date} is {closest_date}") # Output: The closest date to 2025-03-23 00:00:00 is 2025-03-22 00:00:00"
    """

    if not datetime_list:
        return None

    return min(datetime_list, key=lambda x: abs(x - target_datetime))

def get_finance_api_data(url, max_retries=3, wait_time=5):
    retries = 0

    while retries < max_retries:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(f"{url}&apikey={FINANCIAL_KEY}",headers=headers)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit hit! Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2
                retries += 1
            else:
                print(f"HTTP Error {e.response.status_code}: {e.response.reason}")
                break

        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            break

    return None

def build_price_volume_chart_data(stock_list, start_date, end_date, url_str):
    D = {}
    for item in stock_list:
        symbol = item['symbol']
        print(f'Requesting price/volume data for {url_str} chart from {start_date} to {end_date} for {symbol} ...')
        url = f'https://financialmodelingprep.com/stable/{url_str}?symbol={symbol}&from={start_date}&to={end_date}'
        res = get_finance_api_data(url=url)
        res.reverse()
        prices = BiHashList()
        volumes = BiHashList()
        for item in res:
            price_key = 'price' if 'light in url_str' else 'close'
            prices.append(item['date'], item[price_key])
            volumes.append(item['date'], item['volume'])
        D[symbol] = {'prices':prices, 'volumes':volumes}
    return D