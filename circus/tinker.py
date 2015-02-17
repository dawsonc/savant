"""Module for computing errors and technical indicators"""

import scipy.io
import numpy as np

from circus import soldier


def rmse(predictions, labels):
    """Computes the Root Mean Square Error for the given predictions"""
    square_diff = (predictions - labels) ** 2
    mean_diff = np.mean(square_diff)
    return np.sqrt(mean_diff)


def denormalize(data, means, ranges):
    """Undoes feature scaling and mean normalization"""
    return data * ranges + means


def stochastic_k(ts, n):
    """Computes the Stochastic K technical indicator"""
    return 100 * (ts[n] - np.min(ts[n:n + 14])) / \
        (np.max(ts[n:n + 10]) - np.min(ts[n:n + 10]))


def stochastic_d(ts, n):
    """Computes the Stochastic D technical indicator

    Stochastic D is the 3-day SMA of Stochastic K
    """
    return (stochastic_k(ts, n) +
            stochastic_k(ts, n + 1) +
            stochastic_k(ts, n + 2)) / 3


def macd(ts, n):
    """Computes the moving average convergence-divergence (MACD)"""
    alpha = 0.1
    ema12 = 1
    for price in reversed(ts[n:n + 12 + 1]):
        ema12 = alpha * (price - ema12) + ema12
    ema26 = 1
    for price in reversed(ts[n:n + 26 + 1]):
        ema26 = alpha * (price - ema26) + ema26

    return ema12 - ema26


def macd_hist(ts):
    """Computes the histogram of the MACD

    (the difference b/t the MACD and its 9-day SMA)
    """
    alpha = 0.2
    signal_line = 1  # 9-day EMA of MACD
    for i in range(9):
        signal_line = alpha * (macd(ts, i) - signal_line) + signal_line

    return macd(ts, 0) - signal_line


def relative_strength_index(ts):
    """Computes the relative strength index for the given time series"""
    # This is the most complicated indicator to calculate
    # . so watch carefully

    # Calculate interday changes
    deltas = ts[:-1] - ts[1:]

    # Calculate ups and downs
    ups = np.hstack([x if x > 0 else 0 for x in deltas])
    downs = np.hstack([abs(x) if x < 0 else 0 for x in deltas])

    # Find the 10-day exponential moving average
    # . for both the ups and the downs
    alpha = 2.0 / (1.0 + 20.0)
    ups_ema = 1.0
    for price in reversed(ups):
        ups_ema = (price * alpha) + (ups_ema * (1 - alpha))
    downs_ema = 1.0
    for price in reversed(downs):
        downs_ema = (price * alpha) + (downs_ema * (1 - alpha))

    # Compute relative strength factor
    rs = ups_ema / downs_ema

    # Return relative strength index
    return 100 - 100 / (1 + rs)


def slow_stochastic_d(ts):
    """Slow stochastic d is the 3-day moving average of stochastic d"""
    return (stochastic_d(ts, 0) +
            stochastic_d(ts, 1) +
            stochastic_d(ts, 2)) / 3


def five_day_ma(ts):
    """Computes the 5-day SMA of the given time series"""
    return np.mean(ts[0:6])
