"""Module for reshaping the historical data, & constructing feature vectors/matrices"""

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
    """Given historical data, constructs a partial feature vector for the day indicated by `index`

    Arguments:
        data  --- pandas.Series containing historical price data for a particular security
        index --- integer indicating which row for which to construct a partial feature vector
                    Note: 0 is the most recent row, higher indices are further in the past
        k     --- integer indicating the number of preceeding rows to include in the feature vector

    Returns the tuple (features, result) such that:
        features is a numpy array containing the prices for days `index + 30` through `index + 1`
            Note: as the final feature vector contains historical data for multiple indicators,
                  . `features` is only a partial feature vector and needs to be composited with
                  .. the partial feature vectors for the other indicators in order to construct
                  ... the full feature vector
        result is an double containing the price for day `index`
    """
    result = data[index]

    starting_index = index + 1
    end_index = index + k
    features = data[starting_index:end_index+1].as_matrix()

    return (features, result)

def construct_full_feature_vector(feature_vectors):
    """Composite the given tuple of partial feature vectors into a full feature vector

    Note: the ordering of the partial feature vectors in `feature_vectors` must be
          . kept consistent across all calls to this function, or else the training data
          .. will be inconsistently formatted, and thus rendered meaningless
    """
    return np.concatenate(feature_vectors)
