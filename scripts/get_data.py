"""Script for retrieving and saving historical data from Yahoo Finance

Also performs train/cross-validation/test split
"""

import numpy as np
import scipy.io

from datetime import datetime

# Needed to access sibling modules
import sys
sys.path.append('/Users/Charles/progs/ml/final/savant')

from circus import spy
from circus import tailor

# Get the most recent data from the spy (for the SPY ETF, coincidentally)
print("Retrieving historical data...")
raw_data = spy.get_historical_data('SPY')

# Pass the data to the tailor to trim it
raw_data = tailor.reduce_to_column(raw_data, 'Adj Close')

# Only use the last 200 days' data
raw_data = raw_data[:50]

# Then construct feature vectors from the data (using last 100 days)
print("Constructing feature vectors...")
data = np.array(
    [tailor.get_partial_feature_vector(raw_data, i, 26)
     for i, x in enumerate(raw_data)])

# Drop most recent day b/c you can't have a label from the future :P
# . & filter out days for which enough data doesn't exist
data = np.array([x for x in data[1:] if x[0] is not None])
# Now `data` contains tuples of (indicators, price)

# Convert prices to a (1, 3) vector with the shape [Buy, Hold, Sell]
# . where a 1 represents a positive recommendation for that category


def recommendations(x):
    """Returns a (3,) Buy/Hold/Sell recommendation vector for the given data

    A recommendation of "Buy" is given for price changes of >= 0.5%
    A recommendation of "Hold" is given for price changes of -0.5% to 0.5%
    A recommendation of "Sell" is given for price changes of <= -0.5%
    """
    future_price = x[1]
    most_recent_price = x[0][0]

    pct_change = (future_price - most_recent_price) / most_recent_price

    recommendation = []
    if pct_change >= 0.5 / 100:
        recommendation = [1, 0, 0]
    elif abs(pct_change) < 0.5 / 100:
        recommendation = [0, 1, 0]
    elif pct_change <= -0.5 / 100:
        recommendation = [0, 0, 1]

    return np.array(recommendation)


def simple_rec(x):
    """Returns a simple recomendation of 1 for buy and 0 for sell"""
    future_price = x[1]
    most_recent_price = x[0][0]
    pct_change = (future_price - most_recent_price) / most_recent_price
    if pct_change >= 0:
        return 1
    if pct_change < 0:
        return 0


print("Generating Buy/Hold/Sell ratings...")
prepped_data = np.array(list(map(lambda x: (x[0], simple_rec(x)), data)))

print("Shuffling data...")
np.random.shuffle(prepped_data)
train_proportion = 0.6
cv_proportion = 0.2
test_proportion = 1 - (train_proportion + cv_proportion)

train_size = round(prepped_data.shape[0] * train_proportion)
cv_size = round(prepped_data.shape[0] * cv_proportion)

training_data = prepped_data[:train_size]
cv_data = prepped_data[train_size:(train_size + cv_size)]
test_data = prepped_data[(train_size + cv_size):]

print("Saving data...")
scipy.io.savemat("data/stock_data.mat", {"training_data": training_data,
                                         "cv_data": cv_data,
                                         "test_data": test_data
                                         }, appendmat=False)

print("Saving logfile...")
with open('data/log.txt', 'a') as logfile:
    logfile.write("Retrieved & shuffled on %s\n" % (datetime.now()))
