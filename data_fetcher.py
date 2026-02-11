# data_fetcher.py
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_data(coin_id, days=365):
    """
    Fetches historical price data for a given cryptocurrency from CoinGecko.

    Args:
        coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum').
        days (int): Number of days of historical data to fetch.

    Returns:
        pandas.DataFrame: DataFrame with 'Date' and 'Price' columns, or None if error.
    """
    # CoinGecko API endpoint for historical market data
    # 'vs_currency': default to 'usd'
    # 'days': number of days from today
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"

    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Extract prices
        prices = data.get('prices', [])
        if not prices:
            print(f"No price data found for {coin_id}.")
            return None

        # Convert timestamps to datetime objects and create DataFrame
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms') # Convert milliseconds to datetime
        df['Price'] = df['price']
        df = df[['Date', 'Price']]
        df.set_index('Date', inplace=True)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {coin_id}: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing JSON data for {coin_id}: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    btc_data = fetch_historical_data('bitcoin', days=30)
    if btc_data is not None:
        print("Bitcoin Historical Data (last 30 days):")
        print(btc_data.head())

    eth_data = fetch_historical_data('ethereum', days=90)
    if eth_data is not None:
        print("\nEthereum Historical Data (last 90 days):")
        print(eth_data.head())