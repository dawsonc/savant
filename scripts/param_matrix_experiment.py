"""Script for running the experiments in tinker.py"""

import numpy as np

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

# Needed to access sibling modules
import sys
sys.path.append('/Users/Charles/progs/ml/final/savant')

from circus import tinker

print("Test accuracy is not shown so as to isolate it from the dev process")
t = tinker.Tinker("data/stock_data.mat")

HIDDEN_LAYER_SIZES = [10, 30, 100, 300]
REGS = [0, 1, 3, 10, 30, 100, 300, 1000]

# Disable error logging for overlows (which can sometimes occur while
# . caluluating the sigmoid function)
np.seterr(over='ignore')

# Baseline 1 (always predict [1, 1, 1])
print("-----------------------------------------------------------")
print("\t* Baseline 2 (always predict [1, 1, 1]):")
baseline1_accuracies = t.baseline1()
print("\t\tTraining Accuracy: %.2f%%" % (baseline1_accuracies[0]))
print("\t\tCV Accuracy: %.2f%%" % (baseline1_accuracies[1]))

# Baseline 2 (always predict buy)
print("-----------------------------------------------------------")
print("\t* Baseline (always buy):")
baseline2_accuracies = t.baseline2()
print("\t\tTraining Accuracy: %.2f%%" % (baseline2_accuracies[0]))
print("\t\tCV Accuracy: %.2f%%" % (baseline2_accuracies[1]))

print("-----------------------------------------------------------")
print("\t* Using only time series:")
num_days_choices = [10, 30, 100]

# If the optimal parameters are known, enter them here
# . (hidden layer size, regularization, num days)
# If the optimal parameters are unknown, enter None
optimal_ts_params = (10, 0, 30)
# Not really "optimal", b/c a number of combinations give same max accuracy
# . (33%)
if optimal_ts_params:
    accuracies = t.with_time_series(
        optimal_ts_params[0], optimal_ts_params[1], optimal_ts_params[2])
    print("\t\tTraining accuracy: %0.2f%%, CV accuracy: %0.2f%%" %
          (accuracies[0], accuracies[1]))
    print("\t\tParameters\n\t\t\tHidden Layer Size: %s" % optimal_ts_params[0])
    print("\t\t\tRegularization: %s\n\t\t\tNumber of Days Used: %s" %
          (optimal_ts_params[1], optimal_ts_params[2]))
else:
    # Parallelized implementation:
    # . map a list of parameter combinations to a list of dicts containing
    # .. parameters along with training, cv, and test accuracies

    # Construct list of parameter combos
    parameter_combos = []
    for hls in HIDDEN_LAYER_SIZES:
        for reg in REGS:
            for num_days in num_days_choices:
                parameter_combos.append("%s/%s/%s" % (hls, reg, num_days))

    # Define function for mapping parameter combos to accuracies
    def compute_accuracies(parameter_combo):
        # Extract parameters
        params = parameter_combo.split('/')
        hls = int(params[0])
        reg = int(params[1])
        num_days = int(params[2])

        # Compute accuracies
        accuracies = t.with_time_series(num_days, hls, reg)

        # Return dict of parameter combo & all three accuracies
        return {"params": parameter_combo,
                "training_accuracy": accuracies[0],
                "cv_accuracy": accuracies[1],
                "test_accuracy": accuracies[2]}

    # Create worker pool
    pool = Pool(16)

    # Hi Ho! Hi Ho! It's off to work we go
    results = pool.map(compute_accuracies, parameter_combos)

    # Quittin' time
    pool.close()
    pool.join()

    # Now we just have to find the result with the max cv accuracy
    best_result = max(results, key=lambda x: x['cv_accuracy'])
    best_params = best_result['params'].split('/')

    # And display our findings
    print("\t\tTraining Accuracy: %0.2f%%, CV Accuracy: %0.2f%%" %
          (best_result['training_accuracy'], best_result['cv_accuracy']))
    print("\t\tOptimal Parameters\n\t\t\tHidden Layer Size: %s" %
          (best_params[0]))
    print("\t\t\tRegularization: %s\n\t\t\tNumber of Days Used: %s" %
          (best_params[1], best_params[2]))

    optimal_ts_params = (int(best_params[0]),
                         int(best_params[1]),
                         int(best_params[2]))

print("-----------------------------------------------------------")
print("\t* Using only technical indicators:")

# If the optimal parameters are known, enter them here
# . (hidden layer size, regularization, num days)
# If the optimal parameters are unknown, enter None
optimal_ti_params = None
# Not really "optimal", b/c a number of combinations give same max accuracy
# . (33%)
if optimal_ti_params:
    accuracies = t.with_time_series(
        optimal_ti_params[0], optimal_ti_params[1], optimal_ti_params[2])
    print("\t\tTraining accuracy: %0.2f%%, CV accuracy: %0.2f%%" %
          (accuracies[0], accuracies[1]))
    print("\t\tParameters\n\t\t\tHidden Layer Size: %s" % optimal_ti_params[0])
    print("\t\t\tRegularization: %s" %
          (optimal_ti_params[1]))
else:
    # Parallelized implementation:
    # . map a list of parameter combinations to a list of dicts containing
    # .. parameters along with training, cv, and test accuracies

    # Construct list of parameter combos
    parameter_combos = []
    for hls in HIDDEN_LAYER_SIZES:
        for reg in REGS:
            parameter_combos.append("%s/%s" % (hls, reg))

    # Define function for mapping parameter combos to accuracies
    def compute_accuracies(parameter_combo):
        # Extract parameters
        params = parameter_combo.split('/')
        hls = int(params[0])
        reg = int(params[1])

        # Compute accuracies
        accuracies = t.with_indicators(hls, reg)

        # Return dict of parameter combo & all three accuracies
        return {"params": parameter_combo,
                "training_accuracy": accuracies[0],
                "cv_accuracy": accuracies[1],
                "test_accuracy": accuracies[2]}

    # Create worker pool
    pool = ThreadPool(16)

    # Hi Ho! Hi Ho! It's off to work we go
    # Parallelization not working atm. Will fix later
    results = list(map(compute_accuracies, parameter_combos))

    # Quittin' time
    pool.close()
    pool.join()

    # Now we just have to find the result with the max cv accuracy
    best_result = max(results, key=lambda x: x['cv_accuracy'])
    best_params = best_result['params'].split('/')

    # And display our findings
    print("\t\tTraining Accuracy: %0.2f%%, CV Accuracy: %0.2f%%" %
          (best_result['training_accuracy'], best_result['cv_accuracy']))
    print("\t\tOptimal Parameters\n\t\t\tHidden Layer Size: ",
          best_params[0])
    print("\t\t\tRegularization: ", best_params[1])

    optimal_ti_params = (int(best_params[0]),
                         int(best_params[1]))
