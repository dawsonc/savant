import numpy as np
import pandas

# Needed to access sibling modules
import sys
sys.path.append('/Users/Charles/progs/ml/final/savant')

from circus import tinker
from circus import tailor
from circus import soldier
from circus import spy

# Get the most recent stock data
raw_data = spy.get_historical_data('SPY')

# Drop the date column, keeping open, high, low, & adjusted close
raw_data = raw_data.drop("Date", axis=1).drop("Close", axis=1)

# Convert to np array
data = raw_data.as_matrix()

# Only use data from past 100 weeks
data = data[:100 * 5]


def calc_indicators(ts):
    return np.array([tinker.slow_stochastic_d(ts),
                     tinker.stochastic_d(ts, 0),
                     # tinker.five_day_ma(ts),
                     tinker.relative_strength_index(ts),
                     tinker.macd_hist(ts)])

indicators = []
num_days = 26
for i in range(1, data.shape[0] - num_days):
    indicators.append(data[i:i + num_days, 4])
indicators = np.vstack([calc_indicators(x) for x in indicators])

# Normalize indicators
indicators_mean = np.mean(indicators, axis=0)
indicators_range = np.ptp(indicators, axis=0)
indicators = (indicators - indicators_mean) / indicators_range

# Generate labels
pct_changes = - np.diff(data[:, 4]) / data[:-1, 4]
# Normalize
label_mean = np.mean(pct_changes)
label_range = np.ptp(pct_changes)
pct_changes = (pct_changes - label_mean) / label_range

# Generate exemplar tuples
exemplars = np.vstack(list(zip(indicators, pct_changes)))

# Shuffle exemplars and separate into training & cv
np.random.shuffle(exemplars)

m = exemplars.shape[0]
training_exemplars = exemplars[:m * 0.6]
cv_exemplars = exemplars[m * 0.6:]

# Start the music
# Separate training indicators and labels
training_indicators = np.vstack([x[0] for x in training_exemplars])
training_labels = np.array([x[1] for x in training_exemplars])

# Instantiate network and train
input_layer_size = training_indicators.shape[1]
hidden_layer_size = input_layer_size * 2 + 1
num_outputs = 1
reg = 1
activation_function = soldier.tanh_sigmoid
activation_function_gradient = soldier.tanh_sigmoid_gradient
output_activation_func = soldier.tanh_sigmoid
output_activation_func_gradient = soldier.tanh_sigmoid_gradient

rmses = []
for i in range(10):
    nn = soldier.NeuralNetwork(input_layer_size, hidden_layer_size,
                               num_outputs, reg, activation_function,
                               activation_function_gradient,
                               output_activation_func,
                               output_activation_func_gradient)

    nn.train(training_indicators, training_labels)

    # Calculate training error (scaling back up to get real numbers afterwards)
    _, training_predictions = nn.predict(training_indicators)
    training_rmse = tinker.rmse(
        tinker.denormalize(training_predictions, label_mean, label_range),
        tinker.denormalize(training_labels, label_mean, label_range))

    # Now calculate cross validation error
    # . (scaling back up to get real numbers afterwards)
    cv_indicators = np.vstack([x[0] for x in cv_exemplars])
    cv_labels = np.array([x[1] for x in cv_exemplars])
    _, cv_predictions = nn.predict(cv_indicators)
    cv_rmse = tinker.rmse(
        tinker.denormalize(cv_predictions, label_mean, label_range),
        tinker.denormalize(cv_labels, label_mean, label_range))

    rmses.append((training_rmse, cv_rmse))

    predicted = 100 * tinker.denormalize(cv_predictions[0],
                                         label_mean,
                                         label_range)
    actual = 100 * tinker.denormalize(cv_labels[0],
                                      label_mean,
                                      label_range)
    print("Predict: %0.4f Actual: %0.4f" % (predicted, actual))

print("Mean of training set: %0.4f" %
      np.mean(tinker.denormalize(training_labels,
                                 label_mean,
                                 label_range) * 100))

min_training_rmses = min(rmses, key=lambda x: x[0])
print("Minimum Training RMSE: ", min_training_rmses[0])
print("      Minimum CV RMSE: ", min_training_rmses[1])
