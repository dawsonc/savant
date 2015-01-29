"""Module for retrieving historical stock data from Yahoo Fianance"""

import pandas

BASE_URL = "https://ichart.yahoo.com/table.csv?s="


def get_historical_data(symbol):
    """Gets all available historical data for the given symbol

    Takes a stock ticker symbol in the form of a string
    Returns a pandas.DataFrame of all historical price data for that symbol

    Raises ValueError if the given symbol is invalid
    """
    request_url = BASE_URL + symbol
    return pandas.read_csv(request_url)
