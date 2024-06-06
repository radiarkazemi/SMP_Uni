import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
from pymongo import MongoClient
import time
from stock_code_scrapper import get_important_stocks


async def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for a single ticker asynchronously.

    Args:
        ticker (str): Stock ticker.
        start_date (str): Start date for fetching data.
        end_date (str): End date for fetching data.

    Returns:
        pd.DataFrame: DataFrame with historical stock data for the ticker.
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if not stock_data.empty:
            stock_data['ticker'] = ticker
            return stock_data
    except Exception as e:
        print(f"Failed to get data for {ticker}: {e}")
    return None


async def get_stock_data(tickers, days_to_go_back):
    """
    Fetches historical stock data for a list of tickers asynchronously.

    Args:
        tickers (list): List of dictionaries with 'ticker' and 'company' keys.
        days_to_go_back (int): Number of days of historical data to retrieve.

    Returns:
        list: A list of dictionaries containing ticker, company, and its stock data.
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_to_go_back)
                  ).strftime('%Y-%m-%d')

    tasks = [fetch_stock_data(ticker['ticker'], start_date, end_date)
             for ticker in tickers]
    company_data_list = await asyncio.gather(*tasks)

    # Filter out None values
    company_data_list = [{'ticker': tickers[i]['ticker'], 'company': tickers[i]['company'],
                          'data': data} for i, data in enumerate(company_data_list) if data is not None]

    return company_data_list


def save_to_mongodb(stock_data_list, db_name='SMP_uni'):
    """
    Saves the stock data to MongoDB with each ticker as its own collection.

    Args:
        stock_data_list (list): A list of dictionaries containing ticker, company, and its stock data.
        db_name (str): The name of the MongoDB database.
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client[db_name]

    for item in stock_data_list:
        ticker = item['ticker']
        company = item['company']
        data = item['data']
        collection = db[ticker]

        data_records = []
        for idx, row in data.iterrows():
            record = {
                '_id': idx.strftime('%Y-%m-%d'),  # Unique ID based on date
                'company_name': company,
                'data': {
                    'Time': {'$date': int(time.mktime(idx.timetuple()) * 1000)},
                    'Open': row['Open'],
                    'High': row['High'],
                    'Low': row['Low'],
                    'Close': row['Close'],
                    'Volume': row['Volume'],
                    'ticker': ticker
                }
            }
            data_records.append(record)

        # Insert data into MongoDB
        for record in data_records:
            existing_document = collection.find_one({'_id': record['_id']})
            if existing_document:
                collection.update_one({'_id': record['_id']}, {'$set': record})
            else:
                collection.insert_one(record)

    print(f"Data saved to MongoDB database '{db_name}'")


if __name__ == "__main__":
    tickers_info = get_important_stocks()
    days_to_go_back = 600  # Define the number of days to go back

    loop = asyncio.get_event_loop()
    stock_data_list = loop.run_until_complete(
        get_stock_data(tickers_info, days_to_go_back))

    # Save the data to MongoDB
    save_to_mongodb(stock_data_list)

    print("Stock data saved to MongoDB")
