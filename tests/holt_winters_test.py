import numpy as np
import matplotlib.pyplot as plt

# Needed to access top-level stuff
import sys
sys.path.append('/Users/Charles/progs/ml/final/savant')

from circus import tailor

# make a hat function, and add noise
x = np.linspace(0, 1, 100)
x = np.hstack((x, x[::-1]))
x += np.random.normal(loc=0, scale=0.1, size=200) + 3.0
plt.plot(x, alpha=0.4, label='Raw')

# holt winters second order ewma
plt.plot(tailor.holt_winters_ewma(x, 15, 0.3, 1), 'b', label='Holt-Winters')

plt.title('Holt-Winters')
plt.legend(loc=8)

plt.savefig('holt_winters.png', fmt='png', dpi=100)
