import numpy as np
import pandas

from sklearn import svm

# Needed to access sibling modules
import sys
sys.path.append('/Users/Charles/progs/ml/final/savant')

from circus import tinker
from circus import tailor
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
# Generate indicators
indicators = []
num_days = 26
for i in range(1, data.shape[0] - num_days):
    indicators.append(data[i:i + num_days, 4])
indicators = np.vstack([calc_indicators(x) for x in indicators])

# Generate labels
pct_changes = - np.diff(data[:, 4]) / data[:-1, 4]
labels = np.hstack([1 if change >= 0 else 0 for change in pct_changes])

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
cv_indicators = np.vstack([x[0] for x in cv_exemplars])
cv_labels = np.array([x[1] for x in cv_exemplars])

# Instantiate the SVM Classifier
classifier = svm.SVC()

# Train it
classifier.fit(training_indicators, training_labels)

# Calculate % accuracy for Up/Down predictions on both training and cv sets
training_predictions = classifier.predict(training_indicators)
cv_predictions = classifier.predict(cv_indicators)

training_scorecard = (training_predictions == training_labels).astype(int)
training_accuracy = np.mean(training_scorecard) * 100
cv_scorecard = (cv_predictions == cv_labels).astype(int)
cv_accuracy = np.mean(cv_scorecard) * 100

print("Training accuracy: %0.2f" % training_accuracy)
print("CV accuracy: %0.2f" % cv_accuracy)
