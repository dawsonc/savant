"""Main module for connecting tailor, soldier, & spy (get it?).

This is where the magic happens. I'll walk you through it below
"""

import numpy as np
import pandas

import tailor
import soldier
import spy

### Tuning constants

## (for stock data)
PRIMARY_SYMBOL = "SPY"
# The number of previous days to include in the feature vector
NUM_PREV_DAYS = 30
# A list of tuples giving other indices to track and how many days of each
# . to include in the feature vector
# .. e.g. [('AAPL', 10), ('GOOG', 20)] to include the past 10 days of AAPL
#                                      and the past 20 days of GOOG
ADDITIONAL_SYMBOLS = [] # NOT YET SUPPORTED
# The relevant column in the Yahoo Finance data
COLNAME = 'Adj Close'

## (for neural network)
NUM_HIDDEN_UNITS = 25
NUM_OUTPUTS = 1

## (for the sake of science)
CV_SPLIT = 0.2 # proportion of data set aside for cross-validation
TEST_SPLIT = 0.2  # proportion of data set aside for testing

# Get the most recent data from the spy (for the SPY ETF, coincidentally)
print("Retrieving historical data...")
raw_data = spy.get_historical_data(PRIMARY_SYMBOL)

# Pass the data to the tailor to trim it
raw_data = tailor.reduce_to_column(raw_data, COLNAME)

# Smooth with Holt-Winters
raw_data = pandas.Series(tailor.holt_winters_ewma(raw_data, 10, 0.3, 1)[:4629])

# Convert to % change
# First reverse the data so that the average goes forward in time
raw_data = pandas.Series(list(reversed(raw_data)))
# Compute % change
raw_data = raw_data.pct_change()
# Back into original format
raw_data = pandas.Series(list(reversed(raw_data)))
# Drop last element (which is NaN)
raw_data = raw_data[:-1]

# Then construct feature vectors from the data
# First get primary symbol data
print("Constructing feature vectors...")
data = np.array(
        [tailor.get_partial_feature_vector(raw_data, i, NUM_PREV_DAYS)
            for i, x in enumerate(raw_data)])

# . Drop most recent day b/c you can't have a label from the future :P
# . & filter out days for which enough data doesn't exist
data = np.array([x for x in data[1:] if x[0] is not None])
# Now `data` contains tuples of (indicators, price)

# Now generate Buy/Sell ratings for each day to use as output labels
# 1 for buy, 0 for sell
prepped_data = np.array([(x[0], 1) if x[1] > 0 else (x[0], 0) for x in data])

### WATCH THIS SPACE: SECONDARY SYMBOL SUPPORT COMING SOON ###

# Shuffle data for training, cross-validation, & test
print("Shuffling data...")
np.random.shuffle(prepped_data)
# Calculate splits
training_split = 1 - (CV_SPLIT + TEST_SPLIT)
training_size = round(prepped_data.shape[0] * training_split)
cv_size = round(prepped_data.shape[0] * CV_SPLIT)\
# Split the data
training_data = prepped_data[:training_size]
cv_data = prepped_data[training_size:(training_size + cv_size)]
test_data = prepped_data[(training_size + cv_size):]

# Instantiate the neural network
# . note that the output function is sigmoid as we are attempting to classify
# .. days as 1 (increase) or 0 (decrease)
print("Instantiating neural network...")
input_layer_size = NUM_PREV_DAYS
hidden_layer_size = NUM_HIDDEN_UNITS
num_labels = NUM_OUTPUTS
reg = 0
output_func = soldier.sigmoid
output_func_gradient = soldier.sigmoid_gradient
nn = soldier.NeuralNetwork(input_layer_size, hidden_layer_size, num_labels,
                           reg,
                           soldier.sigmoid, soldier.sigmoid_gradient,
                           output_func, output_func_gradient)

# Normalize & Train
print("Normalizing training data...")

print("Training neural network...")
training_indicators = np.array([day[0] for day in training_data])
training_labels = np.array([day[1] for day in training_data])
nn.train(training_indicators, training_labels)
print("Mean of training labels: ", training_labels.mean())
