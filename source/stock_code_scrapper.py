import requests
from bs4 import BeautifulSoup


def get_important_stocks():
    """
    Fetches a list of 10 important S&P 500 stock tickers and company names.

    Returns:
        list: A list of dictionaries with 'ticker' and 'company' keys.
    """
    important_stocks = [
        {'ticker': 'AAPL', 'company': 'Apple Inc.'},
        {'ticker': 'MSFT', 'company': 'Microsoft Corporation'},
        {'ticker': 'AMZN', 'company': 'Amazon.com Inc.'},
        {'ticker': 'GOOGL', 'company': 'Alphabet Inc. (Class A)'},
        {'ticker': 'META', 'company': 'Meta Platforms, Inc.'},
        {'ticker': 'TSLA', 'company': 'Tesla, Inc.'},
        # Corrected symbol
        {'ticker': 'BRK-B', 'company': 'Berkshire Hathaway Inc. (Class B)'},
        {'ticker': 'JNJ', 'company': 'Johnson & Johnson'},
        {'ticker': 'V', 'company': 'Visa Inc.'},
        {'ticker': 'JPM', 'company': 'JPMorgan Chase & Co.'}
    ]
    return important_stocks
