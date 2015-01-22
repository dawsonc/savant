"""Module for reshaping the historical data, & constructing feature matrices"""

import numpy as np


def reduce_to_column(data, column_key):
    """Reduces the csv data by removing all columns except the given column

    pandas.DataFrame, String -> pandas.Series
    """
    try:
        return data[column_key]
    except KeyError:
        raise ValueError("Data must contain the column %s" % column_key)


def get_partial_feature_vector(data, index, k=30):
    """Given historical data, constructs a partial feature vector for \
the day indicated by `index`

    Arguments:
        data  --- pandas.Series containing historical price data for \
a particular security
        index --- integer indicating which row for which to construct a \
partial feature vector
                    Note: 0 is the most recent row, higher indices are further\
in the past
        k     --- integer indicating the total number of rows to include \
in the feature vector, starting at the specified index

    Returns the tuple (features, result) such that:
        features is a numpy array containing the prices for days `index + k` \
through `index` or None if there is not enough data
        result is the price for the day `index`
    """
    starting_index = index + 1
    end_index = index + k

    if end_index < data.size:
        features = data[starting_index:end_index + 1].as_matrix()
    else:
        features = None

    return (features, data[index])


def construct_full_feature_vector(feature_vectors):
    """Composite the given tuple of partial feature vectors into a full \
feature vector

    Note: the ordering of the partial feature vectors in `feature_vectors` \
must be kept consistent across all calls to this function, or else the \
training data will be inconsistently formatted, and thus rendered meaningless
    """
    return np.concatenate(feature_vectors)


def normalize(data, means, stds):
    """Applies mean normalization and feature scaling to the given data"""
    data_norm = data
    num_features = data.shape[1]
    for i in range(num_features):
        data_norm[:, i] = (data[:, i] - means[i]) / stds[i]
    return data_norm
