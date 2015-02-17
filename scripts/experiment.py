"""Script for running the experiments in tinker.py"""

import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

# Needed to access sibling modules
import sys
sys.path.append('/Users/Charles/progs/ml/final/savant')

from circus import tinker

# Get data
import get_data

print("Test errors are not shown so as to isolate it from the dev process")
t = tinker.Tinker("data/stock_data.mat")

# Disable error logging for overlows (which can sometimes occur while
# . caluluating the sigmoid function)
np.seterr(over='ignore')

# Baseline 2 (always predict buy)
print("-----------------------------------------------------------")
print("\t* Baseline (always buy):")
baseline2_errors = t.baseline()
print("\t\tTraining error: %.2f%%" % (baseline2_errors[0]))
print("\t\tCV error: %.2f%%" % (baseline2_errors[1]))

print("-----------------------------------------------------------")
print("\t* Using only technical indicators:")
hidden_layer_size = 20
reg = 0
print("\t\tParameters\n\t\t\tHidden Layer Size: %s" % hidden_layer_size)
print("\t\t\tregularization: %s" %
      (reg))
indicator_errors, _ = t.with_indicators(hidden_layer_size, reg)
print("\t\tTraining Error: %0.2f%%" % indicator_errors[0])
print("\t\tCV Error: %0.2f%%" % indicator_errors[1])
