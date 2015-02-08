"""Script for running the experiments in tinker.py"""

import numpy as np

# Needed to access sibling modules
import sys
sys.path.append('/Users/Charles/progs/ml/final/savant')

from circus import tinker

print("Test error is not shown so as to avoid isolate it from the dev process")
t = tinker.Tinker("data/stock_data.mat")

HIDDEN_LAYER_SIZES = [1, 3, 10, 30, 100, 300]
REGS = [0, 1, 3, 10, 30, 100, 300, 1000]

# Disable error logging for overlows (which can sometimes occur while
# . caluluating the sigmoid function)
np.seterr(over='ignore')

# Baseline (always predict buy)
print("-----------------------------------------------------------")
print("\t* Baseline (always buy):")
baseline_errors = t.baseline()
print("\t\tTraining error: %.2f%%" % (baseline_errors[0]))
print("\t\tCV error: %.2f%%" % (baseline_errors[1]))

print("-----------------------------------------------------------")
print("\t* Using only time series:")
num_days_choices = [1, 3, 10, 30, 100]
# Iterate over all possible arrays to find the errors
ts_errors = np.zeros((len(HIDDEN_LAYER_SIZES),
                      len(REGS),
                      len(num_days_choices),
                      3))
for hls_index, hls in enumerate(HIDDEN_LAYER_SIZES):
    for reg_index, reg in enumerate(REGS):
        for num_days_index, num_days in enumerate(num_days_choices):
            print("\t\t\t(%s, %s, %s)" % (hls, reg, num_days))
            error = t.with_time_series(num_days, hls, reg)
            ts_errors[hls_index][reg_index][num_days_index] = error
# Find the min error
ts_min_coords = np.unravel_index(
    ts_errors.argmax(), ts_errors.shape)
print("\t\tTraining Error: %0.2f%%, CV Error: %0.2f%%" %
      (ts_errors[0], ts_errors[ts_min_coords][1]))
print("\t\tParameters\n\t\t\tHidden Layer Size: %s" % ts_min_coords[0])
print("\t\t\tRegularization: %s\n\t\t\tNumber of Days Used: %s" %
      ts_min_coords[1:])
